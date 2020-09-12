import os
import json
import argparse
from tqdm import tqdm
from datetime import date

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dynamics import DynamicsEnsemble


DATAPATH = '/home/dylan/projects/offline/datasets/hopper_random_agent.json'


def set_seed(seed):
    torch.manual_seed(seed)

def train(args, X, y, dev):
    date_ = str(date.today())
    exp_code = date_ + '_' + str(np.random.randint(1000000))
    writer = SummaryWriter(args.write_to + '_' + exp_code)

    cp_save_path = os.path.join(args.checkpoints, exp_code)
    os.mkdir(cp_save_path)

    in_dim = X.size(1)
    out_dim = y.size(1)

    dynamics = DynamicsEnsemble(args.ensemble_size, 
                                in_dim, 
                                out_dim, 
                                args.encoder_hidden_dim,
                                args.decoder_hidden_dim,
                                args.latent_dim, 
                                args.n_hidden,
                                training=True).to(dev)

    exit(0)

    # shuffle prior to breaking off validation
    shuffle = torch.randperm(X.size()[0])
    X = X[shuffle]
    y = y[shuffle] 

    # break off validation set
    X, X_val = X[:-args.val_size], X[-args.val_size:]
    y, y_val = y[:-args.val_size], y[-args.val_size:]

    print(
        'X size: {}, y size: {}, X_val size: {}, y_val size: {}'.format(X.size(), y.size(), X_val.size(), y_val.size())
    )

    n_samples = X.size(0)

    val_mse_best = 100000

    for epoch in tqdm(range(args.n_epochs)):
        dynamics.train()

        # create and apply random permutation to shuffle data
        shuffle = torch.randperm(X.size()[0])
        X = X[shuffle]
        y = y[shuffle]

        losses = []
        mse_losses = []
        
        for i in range(0, n_samples, args.batch_size):
            next_obs_hat = dynamics(X[i:i+args.batch_size].to(dev))
            next_obs = y[i:i+args.batch_size].to(dev)
            loss, mse_loss = dynamics.update(next_obs, next_obs_hat)
            losses.append(loss)
            mse_losses.append(mse_loss)

        # set to eval mode before running validation
        dynamics.eval()
        # tell graph not to keep track of gradients for validation
        # to avoid running .backward() on model
        with torch.no_grad():
            next_obs_pred = dynamics(X_val.to(dev))
            next_obs = y_val.to(dev)
            next_obs = torch.cat(5*[next_obs.unsqueeze(0)], dim=0)
            val_mse_loss = F.mse_loss(next_obs_pred[0], next_obs)

        if val_mse_loss < val_mse_best:
            loss = torch.mean(val_mse_loss).item()
            checkpoint_path = os.path.join(args.checkpoints, exp_code, str(loss) + '.pt')

            torch.save({
            'epoch': epoch,
            'model_state_dict': dynamics.state_dict(),
            'optimizer_state_dict': dynamics.opt.state_dict(),
            'loss': loss,
            }, checkpoint_path)

            val_mse_best = val_mse_loss

        losses = torch.stack(losses)
        mse_losses = torch.stack(mse_losses)

        writer.add_scalar("Loss/train/mse_loss", torch.mean(mse_losses), epoch)
        writer.add_scalar("Loss/train/loss0", torch.mean(losses[:,0]), epoch)
        writer.add_scalar("Loss/train/loss1", torch.mean(losses[:,1]), epoch)
        writer.add_scalar("Loss/train/loss2", torch.mean(losses[:,2]), epoch)
        writer.add_scalar("Loss/train/loss3", torch.mean(losses[:,3]), epoch)
        writer.add_scalar("Loss/train/loss4", torch.mean(losses[:,4]), epoch)

        writer.add_scalar("Val_Loss/train/mse_loss", torch.mean(val_mse_loss), epoch)

        print('iteration: {} - mse_loss: {} - val_mse_loss: {}'.format(i, torch.mean(mse_losses), torch.mean(val_mse_loss)))
        


def main(args):
    with open(DATAPATH) as data_file: 
        data = json.load(data_file)

    # extract data from dictionry
    observations = torch.tensor(data['observations'])
    actions = torch.tensor(data['actions'])
    next_observations = torch.tensor(data['next_observations'])
    rewards = torch.tensor(data['rewards'])

    # concatenate obs and actions for input
    # concatenate next_obs and reward for output
    X = [torch.cat([o, a], dim=0) for (o, a) in zip(observations, actions)]
    y = [torch.cat([no, r.unsqueeze(0)], dim=0) for (no, r) in zip(next_observations, rewards)]

    # convert list of tensors to tensor
    X = torch.stack(X)
    y = torch.stack(y)

    print(X.size())
    print(y.size())

    dev = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    set_seed(args.seed)

    train(args, X, y, dev)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-ne', '--n_epochs', type=int, default=100)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-es', '--ensemble_size', type=int, default=5)
    parser.add_argument('-nh', '--n_hidden', type=int, default=3)
    parser.add_argument('-eh', '--encoder_hidden_dim', type=int, default=256)
    parser.add_argument('-dh', '--decoder_hidden_dim', type=int, default=128)
    parser.add_argument('-ld', '--latent_dim', type=int, default=8)
    parser.add_argument('-val', '--val_size', type=int, default=10000)
    parser.add_argument('-wt', '--write_to', default='runs/train_dynamics')
    parser.add_argument('-cp', '--checkpoints', default='checkpoints/dynamics')

    args = parser.parse_args()

    main(args)