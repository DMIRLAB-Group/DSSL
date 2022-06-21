import torch
import torch.nn.functional as F
import numpy as np


# Evaluate function: Normalized discounted cumulative gain (NDCG@k) and Recall@k
def NDCG_binary_at_k_season_batch(pred_val, next_obs, k=100):
    batch_users = pred_val.shape[0]
    objects_num = pred_val.shape[1]

    pred_val = pred_val.view(batch_users * objects_num, -1)
    next_obs = next_obs.view(batch_users * objects_num, -1)

    pred_val = F.softmax(pred_val, dim=-1)
    _, idx_topk = torch.topk(pred_val, k=k, dim=1, largest=True, sorted=True)

    if torch.__version__ == '1.0.1.post2':
        next_obs = next_obs.byte()
        next_obs = (torch.ones_like(next_obs) & next_obs).float()
    else:
        next_obs = next_obs.bool().int()
    topk = next_obs[torch.arange(batch_users * objects_num).unsqueeze(1), idx_topk]

    # build the discount template
    tp = torch.reciprocal(torch.log2(torch.arange(2, k + 2)
                                     .float().to(pred_val.device)))
    #
    dcg = (topk * tp).sum(dim=-1)
    idcg = torch.tensor([(tp[:min(n.int(), k)]).sum()
                         for n in next_obs.sum(dim=1)], device=tp.device)
    ndcg = dcg / idcg
    ndcg = ndcg.view(batch_users, objects_num)
    return ndcg


def Recall_at_k_season_batch(pred_val, next_obs, k=100):
    batch_users = pred_val.shape[0]
    objects_num = pred_val.shape[1]

    pred_val = pred_val.view(batch_users * objects_num, -1)
    next_obs = next_obs.view(batch_users * objects_num, -1)

    _, idx_topk = torch.topk(pred_val, k=k, dim=1, largest=True, sorted=True)

    if torch.__version__ == '1.0.1.post2':
        X_pred_binary = torch.zeros_like(pred_val, dtype=torch.uint8)
        X_pred_binary[torch.arange(batch_users * objects_num).unsqueeze(1), idx_topk] = 1
        X_true_binary = next_obs.byte()
    else:
        X_pred_binary = torch.zeros_like(pred_val, dtype=torch.bool)
        X_pred_binary[torch.arange(batch_users * objects_num).unsqueeze(1), idx_topk] = True
        X_true_binary = next_obs.bool()
    tmp = (X_true_binary & X_pred_binary).sum(dim=1).float()
    recall = tmp / torch.min(torch.tensor(k, device=pred_val.device), X_true_binary.sum(dim=1)).float()
    recall = recall.view(batch_users, objects_num)
    return recall


def MAP_at_k_season_batch(pred_val, next_obs, k=5):
    batch_users = pred_val.shape[0]
    objects_num = pred_val.shape[1]

    pred_val = pred_val.view(batch_users * objects_num, -1)
    next_obs = next_obs.view(batch_users * objects_num, -1)

    _, idx_topk = torch.topk(pred_val, k=k, dim=1, largest=True, sorted=True)

    if torch.__version__ == '1.0.1.post2':
        next_obs = next_obs.byte()
        next_obs = (torch.ones_like(next_obs) & next_obs).double()
    else:
        next_obs = next_obs.bool().int()
    aps = torch.zeros(batch_users * objects_num)
    for idx in range(batch_users * objects_num):
        if next_obs[idx].nonzero().size()[0] > 0:
            predicted = idx_topk[idx].tolist()
            actual = next_obs[idx].nonzero().squeeze(dim=1).tolist()
            aps[idx] = apk(actual, predicted, k=k)
        else:
            aps[idx] = np.nan
    return aps.view(batch_users, objects_num)


## steal from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=100):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual: #and p not in predicted[:i]: # not necessary for us since we will not make duplicated recs
            num_hits += 1.0
            score += num_hits / (i+1.0)

    # we handle this part before making the function call
    #if not actual:
    #    return np.nan

    return score / min(len(actual), k)
