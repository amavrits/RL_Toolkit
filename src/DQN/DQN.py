import numpy as np
import pandas as pd
import torch

from itertools import count
import pickle
from tqdm import tqdm

from ReplayMemory import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class DeepQLearning:

    def __init__(self, env, gamma, eps_fn, n_replay=1e5):

        self.env = env
        self.n_states = self.env.state_mesh.shape[0]
        self.state_size = self.env.state_mesh.shape[1]
        self.n_actions = self.env.action_space.n

        self.n_replay = n_replay
        self.init_replay()

        self.gamma = gamma
        self.eps_fn = eps_fn

    def init_replay(self, n_replay=None):
        if n_replay is None: n_replay = self.n_replay
        self.replay = ReplayMemory(int(n_replay))

    def init_nn_models(self, nn_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = nn_model(self.state_size, self.n_actions).to(self.device)
        self.target_net = nn_model(self.state_size, self.n_actions).to(self.device)

    def pick_action(self, state):
        sample = random.random()
        if sample > self.eps:
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        return action

    def transfer_to_tensor(self, x, dtype=torch.float):
        return torch.tensor(x, device=self.device, dtype=dtype).unsqueeze(0)

    def init_cnts(self):
        self.episode_actions = []
        self.episode_durations = []
        self.episode_rewards = []

    def incerement_cnts(self, action, reward, t):
        self.episode_actions.append(action)
        self.episode_durations.append(t + 1)
        self.episode_rewards.append(reward)

    def update_targetnet(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def update_policynet(self):

        if len(self.replay) < self.batch_size:
            return

        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

        if self.fit_transforming:
            state_batch = self.fit_transformer.state_transformation(state_batch)
            non_final_next_state_batch = self.fit_transformer.state_transformation(non_final_next_state_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_state_batch).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.squeeze()

        ''' REMOVE?
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        self.optimizer.zero_grad()
        loss = self.loss_criterion(state_action_values, expected_state_action_values)
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        '''

        if self.fit_transforming:
            expected_state_action_values = self.fit_transformer.value_transformation(expected_state_action_values)

        # Optimize the model
        for epoch in range(1, self.n_epochs + 1):
            self.optimizer.zero_grad()
            state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
            loss = self.loss_criterion(state_action_values, expected_state_action_values)
            loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

    def set_train_params(self, loss_criterion, optimizer, batch_size, tau, n_epochs=1, fit_transformer=None, clip=True):
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.fit_transforming = fit_transformer is not None
        if self.fit_transforming: self.fit_transformer = fit_transformer
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.tau = tau
        self.clip = clip

    def train(
            self,
            loss_criterion,
            optimizer,
            n_episodes,
            batch_size,
            n_epochs=1,
            tau=0.05,
            n_steps_update=100,
            fit_transformer=None,
            verbose=True,
            clip=True
    ):

        self.set_train_params(
            loss_criterion=loss_criterion,
            optimizer=optimizer,
            fit_transformer=fit_transformer,
            batch_size=batch_size,
            n_epochs=n_epochs,
            tau=tau,
            clip=clip
        )

        self.eps = self.eps_fn(0)

        if verbose:
            self.init_fig()

        self.init_cnts()

        for i_episode in tqdm(range(1, n_episodes + 1)):

            self.eps = self.eps_fn(i_episode)

            state, _ = self.env.reset()
            state = self.transfer_to_tensor(state)

            for t in count():

                action = self.pick_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action.item())

                next_state = None if terminated else self.transfer_to_tensor(next_state)

                reward = self.transfer_to_tensor([reward])

                done = terminated or truncated

                self.replay.push(state, action, next_state, reward)

                state = next_state

                ''' REMOVE?
                if (i_episode + t) % n_update == 0:
                    self.update_policynet()
                    self.update_targetnet()
                '''

                if done:
                    self.incerement_cnts(action, reward, t)
                    if verbose:
                        self.update_plot()
                    break

            if i_episode % n_steps_update == 0:
                self.update_policynet()
                self.update_targetnet()

        print('Complete')

    def get_optimal_action(self, state):
        if not torch.is_tensor(state):
            state = self.transfer_to_tensor(state).squeeze()
        if self.fit_transforming:
            state = self.fit_transformer.state_transformation(state)
            values = self.policy_net(state)
            values = self.fit_transformer.value_detransformation(values).reshape(values.shape)
        else:
            values = self.policy_net(state).detach().numpy()
        action = values.argmax(axis=1).squeeze()
        return action

    def get_optimal_value(self, state):
        if not torch.is_tensor(state):
            state = self.transfer_to_tensor(state).squeeze()
        if self.fit_transforming:
            state = self.fit_transformer.state_transformation(state)
            values = self.policy_net(state)
            values = self.fit_transformer.value_detransformation(values).reshape(values.shape)
        else:
            values = self.policy_net(state).detach().numpy()
        return values.squeeze()

    def save_training_results(self, filename):
        filehandler = open(filename+'_Replay', 'wb')
        pickle.dump(self.replay, filehandler)
        torch.save(self.policy_net.state_dict(), filename+'_PolicyNet')
        torch.save(self.target_net.state_dict(), filename+'_TargetNet')
        if self.fit_transforming:
            filehandler = open(filename + '_Transformer', 'wb')
            self.fit_transformer.save_transformer(filehandler)

    def load_training_results(self, filename):
        filehandler = open(filename+'_Replay', 'rb')
        self.replay = pickle.load(filehandler)
        self.policy_net.load_state_dict(torch.load(filename+'_PolicyNet'))
        self.target_net.load_state_dict(torch.load(filename+'_TargetNet'))
        if self.fit_transforming:
            filehandler = open(filename + '_Transformer', 'rb')
            self.fit_transformer.load_transformer(filehandler)

    def init_fig(self):
        self.fig, self.axs = plt.subplots(2)

    def update_plot(self, n_running_mean=100):

        self.axs[0].clear()
        self.axs[1].clear()

        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)

        self.axs[0].plot(durations_t.numpy(), c='b')
        self.axs[1].plot(rewards_t.numpy(), c='b')

        if len(rewards_t) >= n_running_mean:
            means = durations_t.unfold(0, n_running_mean, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(n_running_mean-1), means))
            self.axs[0].plot(means.numpy(), c='r')

            means = rewards_t.unfold(0, n_running_mean, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(n_running_mean-1), means))
            self.axs[1].plot(means.numpy(), c='r')

        self.axs[1].set_xlabel('Episode')
        self.axs[0].set_ylabel('Duration [steps]')
        self.axs[1].set_ylabel('Reward')

        plt.draw()
        plt.pause(0.001)

