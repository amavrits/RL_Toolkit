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
                 loss_fn=torch.nn.MSELoss(),
                 gradient_clip_ratio=1_000
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
        self.loss_fn = loss_fn
        self.gradient_clip_ratio = gradient_clip_ratio

        self.replay = ReplayMemory(self.replay_size)

        self.create_networks()

    def create_networks(self):
        self.q_network = self.n_network(self.state_size, self.n_actions).to(self.device)
        self.target_network = self.n_network(self.state_size, self.n_actions).to(self.device)

    def set_optimizer(self, optimizer, **kwargs):
        # self.optimizer = optimizer(params=self.q_network.parameters(), **kwargs)
        self.optimizer = optimizer(filter(lambda p: p.requires_grad, self.q_network.parameters()), **kwargs)

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
        non_final_mask = torch.all(torch.logical_not(torch.isnan(next_state_batch)), dim=1).to(self.device)
        # non_final_mask = torch.tensor(torch.all(torch.logical_not(torch.isnan(next_state_batch), dim=1)), device=self.device, dtype=torch.bool)
        non_final_next_state_batch = next_state_batch[non_final_mask]

        next_state_q_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            action_max = self.q_network(non_final_next_state_batch).max(1)[1].unsqueeze(1)
            next_state_q_values[non_final_mask] = self.target_network(non_final_next_state_batch).gather(1, action_max).squeeze()
        target_values = reward_batch.squeeze() + self.gamma * next_state_q_values

        '''Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        These are the actions which would've been taken for each batch state according to policy_net'''
        action_batch_one_hot = torch.nn.functional.one_hot(action_batch, self.n_actions)
        q_values = torch.sum(self.q_network(state_batch) * action_batch_one_hot, dim=1)

        '''Optimize the model'''
        self.optimizer.zero_grad()
        self.loss = self.loss_fn(q_values, target_values)
        self.loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), self.gradient_clip_ratio)
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
        if self.device.type == 'cuda':
            return self.q_network(state).cpu().detach().numpy()
        else:
            return self.q_network(state).detach().numpy()

    def eval_agent(self):
        with torch.no_grad():
            self.agent_evals[self.i_episode] = self.agent_eval_fn(self.i_episode, self.q_network)

    def select_action(self, state):

        eps = self.epsilon_fn(self.i_episode)
        rand = torch.rand(1)

        if rand > eps:
            rand_idx = torch.randint(low=0, high=self.n_actions, size=(1,))

            # #TODO: Check correctness and speed
            # illegal_actions = self.illegal_actions_fn(state)
            # p = torch.ones(self.n_actions) / self.n_actions
            # p[illegal_actions] = 0
            # p = p / p.sum()
            # rand_idx = p.multinomial(num_samples=1, replacement=False)

            action = torch.arange(self.n_actions, device=self.device)[rand_idx]
        else:
            with torch.no_grad():
                action = self.q_network(state).max(1)[1]

        return action

    def train(self, env, n_episodes=10_000, checkpoint_file=None, checkpoint_period=500_000):

        self.n_episodes = n_episodes

        if not hasattr(self, 'agent_evals'):
            self.agent_evals = {}

        if not hasattr(self, 'i_episode'):
            self.i_episode = 0

        if not hasattr(self, 'training_steps'):
            self.training_steps = 0

        if not hasattr(self, 'episode_returns'):
            self.episode_returns = []

        for i_episode in tqdm(range(self.i_episode+1, self.n_episodes+1)):

            episode_return = 0

            state, _ = env.reset()

            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=self.state_dtype, device=self.device)
            state = state.unsqueeze(0)

            for _ in count():

                action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action.item())

                if terminated:
                    next_state = torch.tensor([np.nan]*len(next_state), device=self.device).unsqueeze(0)
                else:
                    if not torch.is_tensor(next_state):
                        next_state = torch.tensor(next_state, dtype=self.state_dtype, device=self.device)
                    next_state = next_state.unsqueeze(0)

                episode_return += reward

                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)

                done = terminated or truncated

                self.replay.push(state, action, next_state, reward)

                state = next_state

                self.training_steps += 1

                if self.training_steps % self.update_period == 0:
                    self.update_q_net()

                if self.training_steps % self.target_update_period == 0:
                    self.update_targetnet()

                if done:
                    break

            self.episode_returns.append(episode_return)
            self.i_episode += 1

            if self.run_agent_eval:
                if i_episode % self.eval_agent_period == 0:
                    self.eval_agent()

            if checkpoint_file is not None:
                if i_episode % checkpoint_period == 0:
                    self.save_checkpoint(checkpoint_file)

    def save_agent_evaluation(self, savefile):
        with open(savefile, 'wb') as fp:
            pickle.dump(self.agent_evals, fp)

    def save_checkpoint(self, checkpoint_file):

        torch.save({
            'epoch': self.i_episode,
            'step': self.training_steps,
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'episode_returns': self.episode_returns,
        }, checkpoint_file+'_Q_net.pt')

        filehandler = open(checkpoint_file + '_Replay', 'wb')
        pickle.dump(self.replay, filehandler)

        np.save(checkpoint_file+'_EpisodeReturns.npy', np.array(self.episode_returns))

        self.save_agent_evaluation(savefile=checkpoint_file+'_AgentEvaluations.pkl')

    def load_checkpoint(self, checkpoint_file):

        checkpoint = torch.load(checkpoint_file+'_Q_net.pt')
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']
        self.i_episode = checkpoint['epoch']
        self.training_steps = checkpoint['step']
        self.episode_returns = checkpoint['episode_returns']

        filehandler = open(checkpoint_file + '_Replay', 'rb')
        self.replay = pickle.load(filehandler)

        filehandler = open(checkpoint_file + '_AgentEvaluations.pkl', 'rb')
        self.agent_evals = pickle.load(filehandler)

        self.episode_returns = list(np.load(checkpoint_file+'_EpisodeReturns.npy'))

