import numpy as np

import torch

from ReplayMemory import Transition, ReplayMemory

from itertools import count
import pickle
from tqdm import tqdm


class Agent(object):

    def __init__(self,
                 n_actions,
                 state_size,
                 state_dtype,
                 n_network,
                 gamma=0.99,
                 batch_size=64,
                 replay_size=20_000,
                 update_period=1,
                 target_update_period=1_000,
                 target_soft_tau=1,
                 eval_agent_period=1,
                 agent_eval_fn=None,
                 epsilon_fn=None,
                 torch_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 loss=torch.nn.MSELoss(),
                 gradient_clip=1_000
                 ):

        self.n_actions = n_actions
        self.state_size = state_size
        self.state_dtype = state_dtype
        self.n_network = n_network
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.target_update_period = target_update_period
        self.target_soft_tau = target_soft_tau
        self.eval_agent_period = eval_agent_period
        self.agent_eval_fn = agent_eval_fn
        self.run_agent_eval = self.agent_eval_fn is not None
        self.epsilon_fn = epsilon_fn
        self.update_period = update_period
        self.device = torch_device
        self.training_steps = 0
        self.loss = loss
        self.gradient_clip = gradient_clip
        self.replay = ReplayMemory(self.replay_size)

        self.create_networks()

    def create_networks(self):
        self.q_network = self.n_network(self.state_size, self.n_actions).to(self.device)
        self.target_network = self.n_network(self.state_size, self.n_actions).to(self.device)
        for param in self.target_network.parameters(): # TODO: Not really needed. Remove?
            param.requires_grad = False

    def set_optimizer(self, optimizer, **kwargs):
        self.optimizer = optimizer(params=self.q_network.parameters(), **kwargs)

    def update_q_net(self):

        if self.training_steps < self.batch_size:
            return

        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        '''
        Compute a mask of non-final states and concatenate the batch elements
        (a final state would've been the one after which simulation ended)
        '''
        non_final_mask = torch.tensor(torch.all(~torch.isnan(next_state_batch), dim=1), device=self.device, dtype=torch.bool)
        non_final_next_state_batch = next_state_batch[non_final_mask]

        next_state_q_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_q_values[non_final_mask] = self.target_network(non_final_next_state_batch).max(1)[0]
        target_values = reward_batch.squeeze() + self.gamma * next_state_q_values

        '''Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        These are the actions which would've been taken for each batch state according to policy_net'''
        action_batch_one_hot = torch.nn.functional.one_hot(action_batch, self.n_actions)
        q_values = torch.sum(self.q_network(state_batch) * action_batch_one_hot, dim=1)

        '''Optimize the model'''
        self.optimizer.zero_grad()
        loss = self.loss(q_values, target_values)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()

    def update_targetnet(self):
        # q_network_state_dict = self.q_network.state_dict()
        # target_net_state_dict = self.target_network.state_dict()
        # for key in q_network_state_dict:
        #     target_net_state_dict[key] = q_network_state_dict[key] * self.target_soft_tau +\
        #                                  target_net_state_dict[key] * (1 - self.target_soft_tau)
        # self.target_network.load_state_dict(target_net_state_dict)
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_q_values(self, state):
        return self.q_network(state).detach().numpy()

    def eval_agent(self, i_episode):
        self.agent_evals[i_episode] = self.agent_eval_fn(self.q_network)

    def select_action(self, state, i_episode):

        eps = self.epsilon_fn(i_episode)
        rand = torch.rand(1)

        if rand > eps:
            rand_idx = torch.randint(low=0, high=self.n_actions, size=(1,))
            action = torch.arange(self.n_actions)[rand_idx]
        else:
            action = self.q_network(state).max(1)[1]

        return action

    def train(self, env, n_episodes=10_000):

        self.training_steps = 0
        self.agent_evals = {}

        for i_episode in tqdm(range(n_episodes)):

            state, _ = env.reset()
            state = torch.tensor(state, dtype=self.state_dtype, device=self.device).unsqueeze(0)

            for t in count():

                action = self.select_action(state, i_episode)

                next_state, reward, terminated, truncated, _ = env.step(action.item())

                # next_state = torch.tensor([np.nan]).unsqueeze(0) * terminated +\
                #              torch.tensor(next_state, dtype=self.state_dtype, device=self.device).unsqueeze(0) *\
                #              (1 - terminated)
                next_state = torch.tensor([np.nan]*len(next_state), device=self.device).unsqueeze(0) if terminated else torch.tensor(next_state, dtype=self.state_dtype, device=self.device).unsqueeze(0)

                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)

                done = terminated or truncated

                self.replay.push(state, action, next_state, reward)

                state = next_state

                if self.training_steps % self.update_period == 0:
                    self.update_q_net()

                if self.training_steps % self.target_update_period == 0:
                    self.update_targetnet()

                self.training_steps += 1

                if done:
                    break

            if self.run_agent_eval:
                if i_episode % self.eval_agent_period == 0:
                    self.eval_agent(i_episode)

