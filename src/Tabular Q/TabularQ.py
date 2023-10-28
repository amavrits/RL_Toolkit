import numpy as np
import pandas as pd
from tqdm import tqdm

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

    def pick_action(self, state_idx):
        if np.random.uniform(0, 1) < self.eps:
            action = np.random.choice(np.arange(self.n_actions))
        else:
            action = np.argmax(self.Qtable[state_idx])
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

    def train(self, n_episodes=1_000):

        self.reset_Qtable()
        self.reset_visits()
        self.eps = self.eps_fn(0)

        for i_episode in tqdm(range(1, n_episodes + 1)):

            self.eps = self.eps_fn(i_episode)

            state, _ = self.env.reset()

            done = False

            while not done:

                state_idx = self.env.get_observation_idx(state)

                self.alpha = self.alpha_fn(state)

                action = self.pick_action(state_idx)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated

                next_state, next_value = self.get_next_state(next_state, terminated)

                self.update_Qtable(state_idx, action, reward, next_value)

                self.update_visits(state_idx, action)

                state = next_state

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
