"""Utility functions."""
import torch
from torch.utils import data
from torch import nn


class APairDataset(data.Dataset):

    def __init__(self, pt_file):
        self.pairs_buffer = torch.load(pt_file)
        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.pairs_buffer)):
            num_steps = len(self.pairs_buffer[ep]['u_idx'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = self.pairs_buffer[ep]['obs'][step].toarray()
        action = self.pairs_buffer[ep]['u_idx'][step]
        next_obs = self.pairs_buffer[ep]['next_obs'][step].toarray()

        return obs, action, next_obs


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    elif act_fn == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    if len(indices.size()) > 1:
        zeros = torch.zeros(
            indices.size()[0], indices.size()[1], max_index, dtype=torch.float32,
            device=indices.device)
        return zeros.scatter_(2, indices.unsqueeze(2), 1.)
    else:
        zeros = torch.zeros(
            indices.size()[0], max_index, dtype=torch.float32,
            device=indices.device)
        return zeros.scatter_(1, indices.unsqueeze(1), 1.)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


def truncated_normal_(tensor, mean=0., std=1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
