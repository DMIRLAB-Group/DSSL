import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from models.DSSL import DSSL
from utils.metrics import Recall_at_k_season_batch, NDCG_binary_at_k_season_batch, MAP_at_k_season_batch
from utils.utils import APairDataset


parser = argparse.ArgumentParser()
parser.add_argument('--testset', type=str,
                    default='data/NYC/nyc_eval_week.pt',
                    help='Path to training dataset')
parser.add_argument('--save-folder', type=str,
                    default='results/DSSL_nyc',
                    help='Path to save experiment log.')
parser.add_argument('--meta-name', type=str,
                    default='model_metadata.pkl',
                    help='Metadata of train.')
parser.add_argument('--model-name', type=str,
                    default='model_weight.pt',
                    help='Model name saved by train.')
# Parameter of training model
parser.add_argument('--batch-size', type=int, default=8,
                    help='Batch size.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--gpu', type=str, default='0',
                    help='index of GPU.')
parser.add_argument('--num-workers', type=int, default=5,
                    help='Number of dataloader workers.')
args_eval = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args_eval.gpu
args_eval.cuda = not args_eval.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args_eval.cuda else 'cpu')

meta_file = os.path.join(args_eval.save_folder, args_eval.meta_name)
model_file = os.path.join(args_eval.save_folder, args_eval.model_name)

args = pickle.load(open(meta_file, 'rb'))['args']
print(args)
print(args_eval)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

testset = APairDataset(pt_file=args_eval.testset)
eval_loader = data.DataLoader(
    testset, batch_size=args_eval.batch_size, shuffle=False, num_workers=args_eval.num_workers)

model = DSSL(
    state_net=args.state_net,
    q_dims=args.q_dims,
    p_dims=args.p_dims,
    keep_prob=args.keep_prob,
    feature_dim=args.feature_dim,
    copy_feature=args.copy_feature).to(device)

model.load_state_dict(torch.load(model_file))

model.eval()

n2_list, n1_list, n0_list = [], [], []
r2_list, r1_list, r0_list = [], [], []
m2_list, m1_list, m0_list = [], [], []
topk0 = 5
topk1 = 10
topk2 = 15
with torch.no_grad():
    pbar = tqdm(eval_loader)
    for batch_idx, data_batch in enumerate(pbar):
        data_batch = [tensor.to(device) for tensor in data_batch]
        obs, u_idx, next_obs = data_batch

        pred_val = model(obs.float(), u_idx)

        r0_list.append(Recall_at_k_season_batch(pred_val, next_obs, k=topk0))
        r1_list.append(Recall_at_k_season_batch(pred_val, next_obs, k=topk1))
        r2_list.append(Recall_at_k_season_batch(pred_val, next_obs, k=topk2))
        n0_list.append(NDCG_binary_at_k_season_batch(pred_val, next_obs, k=topk0))
        n1_list.append(NDCG_binary_at_k_season_batch(pred_val, next_obs, k=topk1))
        n2_list.append(NDCG_binary_at_k_season_batch(pred_val, next_obs, k=topk2))
        m0_list.append(MAP_at_k_season_batch(pred_val, next_obs, k=topk0))
        m1_list.append(MAP_at_k_season_batch(pred_val, next_obs, k=topk1))
        m2_list.append(MAP_at_k_season_batch(pred_val, next_obs, k=topk2))

    r0_list = torch.cat(r0_list).transpose(0, 1).to('cpu')
    r1_list = torch.cat(r1_list).transpose(0, 1).to('cpu')
    r2_list = torch.cat(r2_list).transpose(0, 1).to('cpu')
    n0_list = torch.cat(n0_list).transpose(0, 1).to('cpu')
    n1_list = torch.cat(n1_list).transpose(0, 1).to('cpu')
    n2_list = torch.cat(n2_list).transpose(0, 1).to('cpu')
    m0_list = torch.cat(m0_list).transpose(0, 1).to('cpu')
    m1_list = torch.cat(m1_list).transpose(0, 1).to('cpu')
    m2_list = torch.cat(m2_list).transpose(0, 1).to('cpu')

    r0 = np.nanmean(r0_list, axis=1)
    r1 = np.nanmean(r1_list, axis=1)
    r2 = np.nanmean(r2_list, axis=1)
    n0 = np.nanmean(n0_list, axis=1)
    n1 = np.nanmean(n1_list, axis=1)
    n2 = np.nanmean(n2_list, axis=1)
    m0 = np.nanmean(m0_list, axis=1)
    m1 = np.nanmean(m1_list, axis=1)
    m2 = np.nanmean(m2_list, axis=1)

    print("Test Recall@{}=".format(topk0), np.mean(r0))
    print("Test Recall@{}=".format(topk1), np.mean(r1))
    print("Test Recall@{}=".format(topk2), np.mean(r2))
    print("Test NDCG@{}=".format(topk0), np.mean(n0))
    print("Test NDCG@{}=".format(topk1), np.mean(n1))
    print("Test NDCG@{}=".format(topk2), np.mean(n2))
    print("Test MAP@{}=".format(topk0), np.mean(m0))
    print("Test MAP@{}=".format(topk1), np.mean(m1))
    print("Test MAP@{}=".format(topk2), np.mean(m2), '\n')
