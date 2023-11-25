import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import count

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class TabularQLearning:

    def __init__(self, env, gamma, alpha_fn, eps_fn):

        self.env = env
        self.n_states = self.env.state_mesh.shape[0]
        self.state_size = self.env.state_mesh.shape[1]
        self.n_actions = self.env.action_space.n

        self.gamma = gamma
        self.alpha_fn = alpha_fn
        self.eps_fn = eps_fn

    def reset_Qtable(self):
        self.Qtable = np.zeros((self.n_states, self.n_actions))

    def reset_visits(self):
        self.visits = np.zeros((self.n_states, self.n_actions))

    def update_visits(self, state_idx, action):
        self.visits[state_idx, action] += 1

    def pick_action(self, state):

        state_idx = self.env.get_observation_idx(state)

        if self.forbid_actions_fn is None:
            allowed_actions = np.arange(self.env.action_space.n)
        else:
            allowed_actions = self.forbid_actions_fn(state, self.env)

        if np.random.uniform(0, 1) < self.eps:
            action = np.random.choice(allowed_actions)
        else:
            action = np.argmax(self.Qtable[state_idx, allowed_actions])

        return action

    def get_next_state(self, next_state, terminated):
        if terminated:
            next_state = None
            next_value = 0
        else:
            next_state_idx = self.env.get_observation_idx(next_state)
            next_value = self.Qtable[next_state_idx].max()
        return next_state, next_value

    def update_Qtable(self, state_idx, action, reward, next_value):
        old_value = self.Qtable[state_idx, action]
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_value)
        self.Qtable[state_idx, action] = new_value

    def train(self, n_episodes=1_000, forbid_actions_fn=None, verbose=True):

        self.reset_Qtable()

        self.reset_visits()

        self.eps = self.eps_fn(0)

        self.forbid_actions_fn = forbid_actions_fn

        if verbose:
            self.init_fig()

        self.init_cnts()

        for i_episode in tqdm(range(1, n_episodes + 1)):

            self.eps = self.eps_fn(i_episode)

            state, _ = self.env.reset()

            for t in count():

                state_idx = self.env.get_observation_idx(state)

                self.alpha = self.alpha_fn(state, i_episode)

                action = self.pick_action(state)

                if state[0] <= 8 and action == 2:
                    print()

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated

                next_state, next_value = self.get_next_state(next_state, terminated)

                self.update_Qtable(state_idx, action, reward, next_value)

                self.update_visits(state_idx, action)

                state = next_state

                if done:
                    self.incerement_cnts(action, reward, t)
                    if verbose:
                        self.update_plot()
                    break

        self.get_optimal_values()

        self.get_optimal_policy()

    def get_Qtable(self):
        return self.Qtable

    def get_visits(self):
        return self.visits

    def get_optimal_values(self):
        self.optimal_values = np.max(self.Qtable, axis=1)

    def get_optimal_policy(self):
        self.optimal_policy = np.argmax(self.Qtable, axis=1)

    def get_optimal_action(self, state):
        state_idx = self.env.get_observation_idx(state)
        return self.optimal_policy[state_idx]

    def get_policy_df(self, state_columns):
        df = pd.DataFrame(
            data=np.c_[self.env.state_mesh, self.optimal_policy, self.optimal_values],
            columns=state_columns + ['π*', 'V*']
        )
        return df

    def save_training_results(self, filename):
        np.save(filename+'_Qtable.npy', self.Qtable)
        np.save(filename+'_Visits.npy', self.visits)

    def load_training_results(self, filename):
        self.Qtable = np.load(filename+'_Qtable.npy')
        self.visits = np.load(filename+'_Visits.npy')
        self.get_optimal_values()
        self.get_optimal_policy()

    def init_cnts(self):
        self.episode_actions = []
        self.episode_durations = []
        self.episode_rewards = []

    def incerement_cnts(self, action, reward, t):
        self.episode_actions.append(action)
        self.episode_durations.append(t + 1)
        self.episode_rewards.append(reward)

    def init_fig(self):
        self.fig, self.axs = plt.subplots(2)

    def update_plot(self, n_running_mean=100):

        self.axs[0].clear()
        self.axs[1].clear()

        rewards_t = np.array(self.episode_rewards)
        durations_t = np.array(self.episode_durations)

        self.axs[0].plot(durations_t, c='b')
        self.axs[1].plot(rewards_t, c='b')

        if len(rewards_t) >= n_running_mean:
            means = durations_t[-n_running_mean:].mean()
            self.axs[0].plot(means, c='r')

            means = rewards_t[-n_running_mean:].mean()
            self.axs[1].plot(means, c='r')

        self.axs[1].set_xlabel('Episode')
        self.axs[0].set_ylabel('Duration [steps]')
        self.axs[1].set_ylabel('Reward')

        plt.draw()
        plt.pause(0.001)