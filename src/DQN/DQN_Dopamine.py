import numpy as np
import math

import torch
import torch.optim as optim

from itertools import count
import pickle
from tqdm import tqdm

class DQNAgent(object):

    def __init__(self,
                 num_actions,
                 state_shape,
                 state_dtype,
                 n_network,
                 gamma=0.99,
                 batch_size=64,
                 update_horizon=1,
                 min_replay_history=20_000,
                 update_period=1,
                 target_update_period=1_000,
                 epsilon_fn=None,
                 torch_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 use_staging=False,
                 optimizer=torch.optim.RMSprop(
                     lr=0.00025,
                     weight_decay=0.95,
                     momentum=0.0,
                     eps=0.00001,
                     centered=True),
                 loss=torch.torch.nn.MSELoss()
                 ):

        self.num_actions = num_actions
        self.observation_shape = state_shape
        self.observation_dtype = state_dtype
        self.n_network = n_network
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_horizon = update_horizon
        self.cumulative_gamma = math.pow(gamma, update_horizon)
        self.min_replay_history = min_replay_history
        self.target_update_period = target_update_period
        self.epsilon_fn = epsilon_fn
        self.update_period = update_period
        self.device = torch_device
        self.training_steps = 0
        self.optimizer = optimizer
        self.loss = loss

        def create_network(self, name):
            """Builds the neural network used to compute the agent's Q-values."""
            network = self.n_network(self.num_actions, name=name)
            return network

        def build_networks(self):
            """Builds the Q-value network computations needed for acting and training."""
            self.q_nn = self.create_network(name='Q')
            self.target_nn = self.create_network(name='Target')
            self.q_nn_outputs = self.q_nn(self.state_ph)
            """ TODO(bellemare): Ties should be broken. They are unlikely to happen when 
            using a deep network, but may affect performance with a linear approximation scheme."""
            self.q_argmax = torch.max(self.q_nn_outputs.q_values, 1)[1]
            self.target_nn_outputs = self.target_nn(self._replay.states)
            self.next_target_nn_outputs = self.target_nn(self.replay.next_states)

        def build_target_q_op(self):
            """Build an op used as a target for the Q-value."""

            # Get the maximum Q-value across the actions dimension.
            next_q_max = torch.max(self.next_target_nn_outputs.q_values, 1)[1]

            # Calculate the Bellman target value.
            #   Q_t = R_t + \gamma^N * Q'_t+1
            # where,
            #   Q'_t+1 = \argmax_a Q(S_t+1, a)
            #          (or) 0 if S_t is a terminal state,
            # and
            #   N is the update horizon (by default, N=1).
            return self._replay.rewards +\
                   self.cumulative_gamma * next_q_max * (1. - self._replay.terminals.astype(torch.float32))

        def build_train_op(self):
            """Builds a training op."""

            replay_action_one_hot = torch.nn.functional.one_hot(self._replay.actions, self.num_actions)
            replay_chosen_q = torch.sum(self.q_nn_outputs.q_values * replay_action_one_hot, axis=1)

            target = self._build_target_q_op()
            loss = self.loss(target, replay_chosen_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return self.optimizer.step()



