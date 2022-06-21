"""
Cited from https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
and https://github.com/tkipf/c-swm
"""
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn, tensor

from utils import utils
from utils.utils import truncated_normal_


class DSSL(nn.Module):
    """
    Construct a MultiVAE estimator.
    Parameters
    ----------
    p_dims
        decoder MLP layer's dim. [100, 600, n_items]
    q_dims
        encoder MLP layer's dim. [n_items, 600, 100]
    keep_prob
        Dropout regularization parameter (default: 1.0)
    """
    def __init__(
            self,
            state_net: str,
            p_dims: List[int],
            q_dims: Optional[List[int]],
            keep_prob: float,
            feature_dim: int,
            copy_feature: bool = True,
    ) -> None:
        super(DSSL, self).__init__()
        self._TGAE = _DSSL(state_net=state_net,
                           embedding_dim=q_dims[-1],
                           hidden_dim=q_dims[-1],
                           feature_dim=feature_dim,
                           copy_feature=copy_feature)

        self.p_dims = p_dims

        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            self.q_dims = q_dims

        self.keep_prob = keep_prob

        self.weights_q, self.biases_q = nn.ParameterList(), nn.ParameterList()

        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance, respectively.
                d_out *= 2
            self.weights_q.append(
                nn.Parameter(nn.init.xavier_normal_(torch.empty(d_in, d_out)))
            )
            self.biases_q.append(
                nn.Parameter(truncated_normal_(torch.empty(d_out), std=0.001))
            )

        self.weights_p, self.biases_p = nn.ParameterList(), nn.ParameterList()

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            self.weights_p.append(
                nn.Parameter(nn.init.xavier_normal_(torch.empty(d_in, d_out)))
            )
            self.biases_p.append(
                nn.Parameter(truncated_normal_(torch.empty(d_out), std=0.001))
            )

    def q_graph(self, input):
        mu_q, std_q, kl = None, None, None

        h = F.normalize(input, dim=1, eps=1e-6)
        h = F.dropout(h, p=self.keep_prob, training=self.training, inplace=False)

        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = torch.matmul(h, w) + b

            if i != len(self.weights_q) - 1:
                h = torch.tanh(h)
            else:
                mu_q = h[:, :, :self.q_dims[-1]]
                logvar_q = h[:, :, self.q_dims[-1]:]

                std_q = torch.exp(0.5 * logvar_q)
                kl = torch.mean(torch.sum(
                    0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q ** 2 - 1), dim=1))
        return mu_q, std_q, kl

    def p_graph(self, z):
        h = z

        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = torch.matmul(h, w) + b

            if i != len(self.weights_p) - 1:
                h = torch.tanh(h)
        return h

    def forward_pass(self, obs, feature):
        # q-network
        mu_q, std_q, kl = self.q_graph(obs)
        epsilon = torch.normal(mean=0., std=torch.ones(std_q.shape, device=mu_q.device))
        sampled_z = mu_q + self.training * epsilon * std_q

        next_state = self._TGAE(sampled_z, feature)
        pred_val = self.p_graph(next_state)

        # p-network
        logits = self.p_graph(sampled_z)
        return logits, pred_val, kl

    def compute_loss(self, obs, feature, next_obs, anneal):
        logits, pred_val, kl = self.forward_pass(obs, feature)
        log_softmax_var = F.log_softmax(logits, dim=-1)
        neg_ll = -torch.mean(torch.sum(log_softmax_var * obs, dim=-1))
        log_softmax_var1 = F.log_softmax(pred_val, dim=-1)
        neg_ll1 = -torch.mean(torch.sum(log_softmax_var1 * next_obs, dim=-1))
        neg_ELBO = neg_ll + neg_ll1 + anneal * kl
        return neg_ELBO

    def forward(self, obs, feature):
        mu_q, std_q, kl = self.q_graph(obs)
        epsilon = torch.normal(mean=0., std=torch.ones(std_q.shape, device=mu_q.device))
        sampled_z = mu_q + self.training * epsilon * std_q

        next_state = self._TGAE(sampled_z, feature)
        pred_val = self.p_graph(next_state)

        return pred_val


class _DSSL(nn.Module):
    def __init__(self,
                 state_net,
                 embedding_dim,
                 hidden_dim,
                 feature_dim,
                 copy_feature=False):
        super(_DSSL, self).__init__()
        if state_net== 'linear':
            self.state_net = StateLinearNet(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                feature_dim=feature_dim,
                copy_feature=copy_feature)
        elif state_net== 'gnn':
            self.state_net = StateGnnNet(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                feature_dim=feature_dim,
                copy_feature=copy_feature)

    def forward(self, state, feature):
        _delta_state = self.state_net(state, feature)
        return state + _delta_state


class StateGnnNet(nn.Module):
    """
        GNN-based transition function.
    Args:
        input_dim: input shape.
        hidden_dim: hidden layer dimension.
        feature_dim: Number of users.
        copy_feature: Apply same user feature to all slots.
        act_fn: activate function.(default: 'relu')
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 feature_dim,
                 copy_feature=True,
                 act_fn='relu'):
        super(StateGnnNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.copy_feature = copy_feature
        self.feature_dim = feature_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim))

        node_input_dim = hidden_dim + input_dim + self.feature_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, input_dim))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target):
        out = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = utils.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, device):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            # offset [batch_size, 1]
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            # offset [batch_size, num_objects * (num_objects - 1)]
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1).to(device)

        return self.edge_list

    def forward(self, states, feature):
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.input_dim)

        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, states.device
            )
            # row: in, col: out (col->row)
            row, col = edge_index
            edge_attr = self._edge_model(
                node_attr[row], node_attr[col])
        
        if self.copy_feature:
            feature_vec = utils.to_one_hot(
                feature, self.feature_dim).repeat(1, num_nodes)
        else:
            feature_vec = utils.to_one_hot(
                feature, self.feature_dim)
        feature_vec = feature_vec.view(-1, self.feature_dim)

        # Attach feature to each state
        node_attr = torch.cat([node_attr, feature_vec], dim=-1)

        delta_states = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, input_dim]
        return delta_states.view(batch_size, num_nodes, -1)


class StateLinearNet(nn.Module):
    """ Linear state transition
    Args:
        input_dim: input shape.
        hidden_dim: hidden layer dimension.
        feature_dim: Number of users.
        copy_feature: Apply same user feature to all slots.
        act_fn: activate function.(default: 'relu')
    """
    def __init__(self, input_dim, hidden_dim, feature_dim, copy_feature=True, act_fn='relu'):
        super(StateLinearNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.copy_feature = copy_feature

        if self.ignore_feature:
            self.feature_dim = 0
        else:
            self.feature_dim = feature_dim
        node_dim = self.input_dim + self.feature_dim
        self.linear_trainsition = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, input_dim))

    def forward(self, states: tensor, feature: Optional[tensor]) -> tensor:
        """
        Transfer state to the next state.
        Parameters
        ----------
        states [batch_size, num_objects, input_dim]
            state input.
        feature
            features that combining on the current state.
        Returns
        -------
        delta_states [batch_size, num_objects, hidden_dim]
            delta states output.
        """
        global feature_vec
        batch_size = states.size(0)
        num_nodes = states.size(1)
        node_attr = states.view(-1, self.input_dim)
        if self.copy_feature:
            feature_vec = utils.to_one_hot(
                feature, self.feature_dim).repeat(1, num_nodes)
            feature_vec = feature_vec.view(-1, self.feature_dim)
        delta_states = self.linear_trainsition(torch.cat((node_attr, feature_vec), dim=-1))
        return delta_states.view(batch_size, num_nodes, -1)
