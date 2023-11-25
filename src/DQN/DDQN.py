import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from itertools import count
import pickle
from tqdm import tqdm

from ReplayMemory import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class DeepQLearning:

    def __init__(self, env, gamma, eps_fn, n_replay=2e4):

        self.env = env
        self.state_size = len(self.env.observation_space)
        self.n_actions = self.env.action_space.n

        self.n_replay = n_replay
        self.reset_replay()

        self.gamma = gamma
        self.eps_fn = eps_fn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset_replay(self, n_replay=None):
        if n_replay is None: n_replay = self.n_replay
        self.replay = ReplayMemory(int(n_replay))

    def read_nn_models(self, policy_net, target_net):
        self.policy_net = policy_net.to(self.device)
        self.target_net = target_net.to(self.device)

    def pick_action(self, state):

        if self.forbid_actions_fn is None:
            allowed_actions = torch.arange(self.env.action_space.n).to(self.device)
        else:
            # allowed_actions = self.forbid_actions_fn(state.detach().numpy().squeeze(), self.env) #TODO
            allowed_actions = self.forbid_actions_fn(state, self.env).to(self.device)

        sample = random.random()

        if sample > self.eps:
            if self.fit_transforming:
                state = self.fit_transformer.state_transformation(state)
            with torch.no_grad():
                action = self.policy_net(state)[:, allowed_actions].max(1)[1].view(1, 1)
        else:
            p = torch.zeros(self.env.action_space.n, device=self.device)
            p[allowed_actions] = 1 / len(allowed_actions)
            action = p.multinomial(num_samples=1).type(torch.long).unsqueeze(0)
            # action = torch.tensor([[np.random.choice(torch.arange(self.env.action_space.n)[allowed_actions])]], device=self.device, dtype=torch.long)
        return action

    def transfer_to_tensor(self, x, dtype=torch.float):
        return torch.tensor(x, device=self.device, dtype=dtype).unsqueeze(0)

    def init_cnts(self):
        self.episode_durations = []
        self.episode_returns = []

    def incerement_cnts(self, G, t):
        self.episode_durations.append(t + 1)
        self.episode_returns.append(G)

    def update_targetnet(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau +\
                                         target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def get_batches(self):

        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        '''
        Compute a mask of non-final states and concatenate the batch elements
        (a final state would've been the one after which simulation ended)
        '''
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)

        try:
            non_final_next_state_batch = torch.cat([s.to(self.device) for s in batch.next_state if s is not None])
        except:
            non_final_next_state_batch = None

        return (state_batch, action_batch, reward_batch, non_final_next_state_batch, non_final_mask)

    def get_batch_values(self, non_final_mask, non_final_next_state_batch, reward_batch):

        '''
        Compute V(s_{t+1}) for all next states.
        Expected values of actions for non_final_next_states are computed based
        on the "older" target_net; selecting their best reward with max(1)[0].
        This is merged based on the mask, such that we'll have either the expected
        state value or 0 in case the state was final.
        '''
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_state_batch is not None:
            with torch.no_grad():
                if self.forbid_actions_fn is None:
                    arg_max = self.policy_net(non_final_next_state_batch).max(1)[1].unsqueeze(1)
                else:
                    allowed_action = self.forbid_actions_fn(non_final_next_state_batch, self.env).to(self.device)
                    arg_max = torch.masked_fill(self.policy_net(non_final_next_state_batch), ~allowed_action, -9999).max(1)[1].unsqueeze(1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_state_batch).gather(1, arg_max).squeeze()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze()

        return expected_state_action_values.type(torch.float32)

    def update_policynet(self):

        if len(self.replay) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, non_final_next_state_batch, non_final_mask = self.get_batches()

        if self.fit_transforming:
            state_batch = self.fit_transformer.state_transformation(state_batch)
            non_final_next_state_batch = self.fit_transformer.state_transformation(non_final_next_state_batch)
            reward_batch = self.fit_transformer.value_transformation(reward_batch)

        expected_state_action_values = self.get_batch_values(non_final_mask, non_final_next_state_batch, reward_batch)

        '''Optimize the model'''
        for epoch in range(1, self.n_epochs + 1):
            self.optimizer.zero_grad()

            '''
            Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
            These are the actions which would've been taken for each batch state according to policy_net
            '''
            state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

            loss = self.loss_criterion(state_action_values, expected_state_action_values)
            loss.backward()
            if self.clipping:
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.clip)
            self.optimizer.step()

    def set_train_params(self, loss_criterion, optimizer, batch_size, n_epochs=1, fit_transformer=None, forbid_actions_fn=None, clip=100):
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.fit_transforming = fit_transformer is not None
        if self.fit_transforming: self.fit_transformer = fit_transformer
        self.forbid_actions_fn = forbid_actions_fn
        self.batch_size = batch_size
        self.clipping = clip is not None
        if self.clipping: self.clip = clip
        self.n_epochs = n_epochs

    def train(self, n_episodes, tau=0.05, n_steps_policy_update=1, n_steps_target_update=4, include_episode_number_storage=False,
              verbose=True, n_running_mean=100, n_store_training_episodes=0, pickup_training=False, savefile=None):

        step_cnt = 0

        store_training = n_store_training_episodes > 0 if n_store_training_episodes is not None else False

        self.tau = tau

        if not pickup_training:
            self.reset_replay()
            self.init_cnts()
            self.current_step = 0
        else:
            step_cnt = self.load_training_results(savefile, include_episode_number_storage)

        if verbose:
            self.init_fig()

        for i_episode in tqdm(range(self.current_step+1, n_episodes + 1)):

            G = 0

            self.eps = self.eps_fn(i_episode)

            state, _ = self.env.reset()
            state = self.transfer_to_tensor(state)

            for t in count():

                step_cnt += 1

                action = self.pick_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action.item())

                next_state = None if terminated else self.transfer_to_tensor(next_state)

                reward = self.transfer_to_tensor([reward], dtype=torch.float64)

                # if action == 2 and state[:, 2] == 0: #TODO
                #     reward = self.transfer_to_tensor([-50], dtype=torch.float64)
                #     terminated = True

                done = terminated or truncated

                self.replay.push(state, action, next_state, reward)

                state = next_state

                G += self.gamma ** t * reward.item()

                if step_cnt % n_steps_policy_update == 0:
                    self.update_policynet()

                if step_cnt % n_steps_target_update == 0:
                    self.update_targetnet()

                if done:
                    self.incerement_cnts(G, t)
                    if verbose:
                        self.update_plot(n_running_mean=n_running_mean)
                    if store_training:
                        if i_episode % n_store_training_episodes == 0:
                            self.save_training_results(savefile, i_episode, step_cnt, include_episode_number_storage)
                    break

    def get_values(self, state):
        if not torch.is_tensor(state):
            state = self.transfer_to_tensor(state).squeeze()
        if self.fit_transforming:
            state = self.fit_transformer.state_transformation(state)
            values = self.policy_net(state)
            values = self.fit_transformer.value_detransformation(values).reshape(values.shape)
        else:
            values = self.policy_net(state)
        if self.forbid_actions_fn is not None:
            allowed_actions = self.forbid_actions_fn(state, self.env)
            values = torch.masked_fill(values, ~allowed_actions, -9999)
        if self.device.type == 'cuda':
            return values.squeeze().cpu().detach().numpy()
        else:
            return values.squeeze().detach().numpy()

    def get_optimal_value(self, state):
        values = self.get_values(state)
        return values.max(axis=1)

    def get_optimal_action(self, state):
        values = self.get_values(state)
        return np.argmax(values, axis=1)

    def save_training_results(self, filename, i_episode=None, step_cnt=None, include_episode_number_storage=False):
        np.save(filename+'_Counters.npy', np.array([self.episode_durations, self.episode_returns]))
        if i_episode is not None:
            np.save(filename+'_Steps.npy', np.array([[i_episode], [step_cnt]]))
        filehandler = open(filename+'_Replay', 'wb')
        pickle.dump(self.replay, filehandler)
        if include_episode_number_storage:
            state = {
                'epoch': step_cnt,
                'state_dict': self.policy_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, filename+'_'+str(i_episode)+'_PolicyNet')
            state = {
                'epoch': step_cnt,
                'state_dict': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, filename+'_'+str(i_episode)+'_TargetNet')
            # torch.save(self.policy_net.state_dict(), filename+'_'+str(i_episode)+'_PolicyNet')
            # torch.save(self.target_net.state_dict(), filename+'_'+str(i_episode)+'_TargetNet')
        else:
            state = {
                'epoch': step_cnt,
                'state_dict': self.policy_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, filename+'_PolicyNet')
            state = {
                'epoch': step_cnt,
                'state_dict': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, filename+'_TargetNet')
            # torch.save(self.policy_net.state_dict(), filename+'_PolicyNet')
            # torch.save(self.target_net.state_dict(), filename+'_TargetNet')

    def load_training_results(self, filename, include_episode_number_storage=False):
        self.current_step, step_cnt = np.load(filename+'_Steps.npy', allow_pickle=True)
        self.current_step = self.current_step.item()
        step_cnt = step_cnt.item()
        filehandler = open(filename+'_Replay', 'rb')
        self.replay = pickle.load(filehandler)
        if include_episode_number_storage:
            state = torch.load(filename+'_'+str(self.current_step)+'_PolicyNet')
            self.policy_net.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            state = torch.load(filename+'_'+str(self.current_step)+'_TargetNet')
            self.policy_net.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            # self.policy_net.load_state_dict(torch.load(filename+'_'+str(self.current_step)+'_PolicyNet'))
            # self.target_net.load_state_dict(torch.load(filename+'_'+str(self.current_step)+'_TargetNet'))
        else:
            state = torch.load(filename+'_PolicyNet')
            self.policy_net.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            state = torch.load(filename+'_TargetNet')
            self.policy_net.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            # self.policy_net.load_state_dict(torch.load(filename+'_PolicyNet'))
            # self.target_net.load_state_dict(torch.load(filename+'_TargetNet'))
        self.episode_durations, self.episode_returns = np.load(filename + '_Counters.npy')
        self.episode_durations = list(self.episode_durations)
        self.episode_returns = list(self.episode_returns)
        if self.fit_transforming:
            self.fit_transformer.load_transformer(filename)
        return step_cnt

    def init_fig(self):
        self.fig, self.axs = plt.subplots(2)

    def update_plot(self, n_running_mean=100):

        self.axs[0].clear()
        self.axs[1].clear()

        rewards_t = torch.tensor(self.episode_returns, dtype=torch.float)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)

        self.axs[0].plot(durations_t.numpy(), c='b')
        self.axs[1].plot(rewards_t.numpy(), c='b')

        if len(rewards_t) >= n_running_mean:
            means = durations_t.unfold(0, n_running_mean, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(n_running_mean-1), means)).detach().numpy().squeeze()
            self.axs[0].plot(means, c='r')

            means = rewards_t.unfold(0, n_running_mean, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(n_running_mean-1), means)).detach().numpy().squeeze()
            self.axs[1].plot(means, c='r')

        self.axs[1].set_xlabel('Episode')
        self.axs[0].set_ylabel('Duration [steps]')
        self.axs[1].set_ylabel('Return')

        plt.draw()
        plt.pause(0.001)

