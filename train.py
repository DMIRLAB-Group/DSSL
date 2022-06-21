import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data

from models.DSSL import DSSL
from utils.utils import APairDataset


parser = argparse.ArgumentParser()
# Parameter of data path and save path
parser.add_argument('--dataset', type=str,
                    default='data/NYC/nyc_train_week.pt',
                    help='Path to training dataset')
parser.add_argument('--save-folder', type=str,
                    default='results/DSSL_nyc',
                    help='Path to save experiment.')
parser.add_argument('--meta-name', type=str,
                    default='model_metadata.pkl',
                    help='Metadata of train.')
parser.add_argument('--model-name', type=str,
                    default='model_weight.pt',
                    help='Model name saved by train.')
# Parameter of training model
parser.add_argument('--batch-size', type=int, default=8,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=150,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=1e-4,
                    help='Learning rate.')
parser.add_argument('--weight-decay', type=float, default=0.001,
                    help='Scale of L2 norm.')
parser.add_argument('--seed', type=int, default=2020,
                    help='Random seed (default: 2020).')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--gpu', type=str, default='0',
                    help='index of GPU.')
parser.add_argument('--num-workers', type=int, default=5,
                    help='Number of dataloader workers.')
# Parameter of encoding MLP and decoding MLP
parser.add_argument('--q-dims', type=int, nargs='+', default=[38333, 300, 50],
                    action='store', dest='q_dims',
                    help='Dimensionality of q process.')
parser.add_argument('--p-dims', type=int, nargs='+', default=[50, 300, 38333],
                    action='store', dest='p_dims',
                    help='Dimensionality of p process.')
parser.add_argument('--anneal', type=float, default=1,
                    help='Scale of KL.')
parser.add_argument('--keep-prob', type=float, default=0.1,
                    help='Prob of dropout.')
# Parameter of state dependency network
parser.add_argument('--state-net', type=str, default='gnn',
                    help='default: gnn, linear.')
parser.add_argument('--feature-dim', type=int, default=1083,
                    help='Number of users.')
parser.add_argument('--copy-feature', action='store_true', default=True,
                    help='Apply same user feature to all slots.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')

# Set the random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Mkdir
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
meta_file = os.path.join(args.save_folder, args.meta_name)
model_file = os.path.join(args.save_folder, args.model_name)

pickle.dump({'args': args}, open(meta_file, "wb"))


# Load the training dataset
dataset = APairDataset(pt_file=args.dataset)
train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# Build a model
model = DSSL(
    state_net=args.state_net,
    q_dims=args.q_dims,
    p_dims=args.p_dims,
    keep_prob=args.keep_prob,
    feature_dim=args.feature_dim,
    copy_feature=args.copy_feature).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    weight_decay=args.weight_decay,
    lr=args.learning_rate
)
print(args)

# Train model.
print('Starting model training...')
best_loss = 1e9
count = 0
for _, epoch in enumerate(range(1, args.epochs + 1)):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader)
    for batch_idx, data_batch in enumerate(pbar):
        optimizer.zero_grad()

        data_batch = [tensor.to(device) for tensor in data_batch]
        # obs (batch_size, num_objects, num_items)
        obs, u_idx, next_obs = data_batch

        loss = model.compute_loss(obs.float(), u_idx, next_obs.float(), anneal=args.anneal)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_description('Loss: {:.6f}'.format(loss.item()))

    avg_loss = train_loss / (batch_idx + 1)
    print('Epoch: {} Average loss: {:.6f}\n'.format(epoch, avg_loss))

    if train_loss < best_loss:
        count = 0
        best_loss = train_loss
        torch.save(model.state_dict(), model_file)
    else:
        count += 1
        if count == 5:
            print('early stop!')
            break
