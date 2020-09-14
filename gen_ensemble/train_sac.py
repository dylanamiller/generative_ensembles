import os
import json
import argparse
from tqdm import tqdm
from datetime import date

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import SAC
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback


from dynamics import DynamicsEnsemble


DATAPATH = '/home/dylan/projects/offline/datasets/hopper_random_agent.json'


class TensorboardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, exp_code, check_freq: int, log_dir: str, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.writer = None
        self.exp_code = exp_code
        self.log_dir = log_dir
        self.check_freq = check_freq
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')     # x represents timesteps
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-50:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

            writer.add_scalar("agent_reward/train/mean_episode_reward", mean_reward, self.n_calls)

            print('iteration: {} - mse_loss: {} - val_mse_loss: {}'.format(i, torch.mean(mse_losses), torch.mean(val_mse_loss)))

        return True


def set_seed(seed):
    torch.manual_seed(seed)

def train(args, data):
    # extract data from dictionry
    observations = torch.tensor(data['observations'])
    actions = torch.tensor(data['actions'])
    next_observations = torch.tensor(data['next_observations'])
    rewards = torch.tensor(data['rewards'])
    dones = data['dones']

    in_dim = torch.cat([observations[0], actions[0]]).size(0)
    out_dim = torch.cat([next_observations[0], rewards[0]]).size(0)

    dynamics = DynamicsEnsemble(args.ensemble_size, 
                                in_dim, 
                                out_dim, 
                                args.encoder_hidden_dim,
                                args.decoder_hidden_dim,
                                args.latent_dim, 
                                args.n_hidden,
                                )

    # load saved model into dynamics ensemble
    checkpoint = torch.load(args.CHECKPOINTPATH)
    dynamics.load_state_dict(checkpoint['model_state_dict'])
    dynamics.opt.load_state_dict(checkpoint['optimizer_state_dict'])

    # define SAC agent
    agent = SAC('MlpPolicy', dynamics)

    # set up callback
    date_ = str(date.today())
    exp_code = date_ + '_' + str(np.random.randint(1000000))
    log_dir = args.write_to + '_' + exp_code
    callback = TensorboardCallback(exp_code, check_freq=100, log_dir=log_dir)

    timesteps = 1e5
    agent.learn(total_timesteps=int(timesteps), callback=callback)

    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "SAC: Generative Ensembles")
    plt.show()

    # for rollout in tqdm(range(args.n_rollouts)):
    #     # randomly choose observation to start rollout
    #     # if rollout starts at done=True, reselect
    #     done = True
    #     while done:
    #         obs_idx = torch.randint(observations.size()[0])
    #         done = dones[obs_idx]

    #     # choose obs and manually set obs as rollout start
    #     obs = observations[obs_idx]
    #     dynamics.reset(obs)

    #     for i in range(args.rollout_length):
    #         action = agent(obs)     # don't have agent yet
    #         next_obs, reward = dynamics.step(action)
            

    #     losses = torch.stack(losses)
    #     mse_losses = torch.stack(mse_losses)

    #     writer.add_scalar("Loss/train/mse_loss", torch.mean(mse_losses), epoch)
    #     writer.add_scalar("Loss/train/loss0", torch.mean(losses[:,0]), epoch)
    #     writer.add_scalar("Loss/train/loss1", torch.mean(losses[:,1]), epoch)
    #     writer.add_scalar("Loss/train/loss2", torch.mean(losses[:,2]), epoch)
    #     writer.add_scalar("Loss/train/loss3", torch.mean(losses[:,3]), epoch)
    #     writer.add_scalar("Loss/train/loss4", torch.mean(losses[:,4]), epoch)

    #     writer.add_scalar("Val_Loss/train/mse_loss", torch.mean(val_mse_loss), epoch)

    #     print('iteration: {} - mse_loss: {} - val_mse_loss: {}'.format(i, torch.mean(mse_losses), torch.mean(val_mse_loss)))
        


def main(args):
    with open(DATAPATH) as data_file: 
        data = json.load(data_file)

    set_seed(args.seed)

    train(args, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-nr', '--n_rollouts', type=int, default=100)
    parser.add_argument('-rl', '--rollout_length', type=int, default=128)
    parser.add_argument('-es', '--ensemble_size', type=int, default=5)
    parser.add_argument('-nh', '--n_hidden', type=int, default=3)
    parser.add_argument('-eh', '--encoder_hidden_dim', type=int, default=256)
    parser.add_argument('-dh', '--decoder_hidden_dim', type=int, default=128)
    parser.add_argument('-ld', '--latent_dim', type=int, default=8)
    parser.add_argument('-val', '--val_size', type=int, default=10000)
    parser.add_argument('-wt', '--write_to', default='runs/train_sac')
    parser.add_argument('-cp', '--checkpoints', default='checkpoints/dynamics')

    args = parser.parse_args()

    main(args)