from make_env import make_env
import networkx as nx
import numpy as np
import random
SAMPLE_NUMS = 10
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out

class ValueNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class Agent:
    def __init__(self):
        super(Agent).__init__()
        self.value_network = ValueNetwork(input_size=STATE_DIM, hidden_size=24, output_size=1).to(device)
        self.actor_network = ActorNetwork(STATE_DIM, 24, ACTION_DIM).to(device)
        self.actor_network_optim = torch.optim.Adam(self.actor_network.parameters(), lr=0.0001)
        self.tao = deepcopy(self.value_network).to(device)
        self.y = 1
        self.reward = None
        self.shape = []
        for name, param in self.value_network.named_parameters():
            self.shape.append(deepcopy(param.cpu().data).numpy().shape)

    def update_tao(self, consensus_tao, qs):
        target_values = qs
        values = self.value_network(states_var.to(device))
        criterion = nn.MSELoss()
        value_network_loss = criterion(values, target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(self.value_network.parameters(), 0.5)
        temp = []
        for name, param in self.value_network.named_parameters():
            temp.append(deepcopy(param.grad))
            param.grad.zero_()
        i = 0
        for name, param in self.tao.named_parameters():
            param.data = torch.from_numpy(consensus_tao[i]).to(device) - (temp[i].to(device) * 0.1)
            i += 1



def consensus(agents, adjacent_matrix):
    # adjacent_matrix = np.asarray(nx.to_numpy_matrix(commu_graph))
    n_agent = len(agents)
    y_list = [[] for _ in range(n_agent)]
    tao_list = [[] for _ in range(n_agent)]
    reward_list = [[] for _ in range(n_agent)]
    for i, agent in enumerate(agents):
        for j, _agent in enumerate(agents):
            if i == j or adjacent_matrix[i, j] > 0.0:
                y_list[i].append(_agent.y)
                reward_list[i].append(np.array(_agent.reward))
                tao = []
                for name, param in _agent.tao.named_parameters():
                    tao.append(deepcopy(param.cpu().data).numpy())
                tao_list[i].append(np.array(tao))

    consensus_y = [np.mean(np.asarray(x), axis=0) for x in y_list]
    consensus_tao = [np.mean(np.asarray(x), axis=0) for x in tao_list]
    consensus_reward = [np.mean(np.asarray(x), axis=0) for x in reward_list]
    return consensus_tao, consensus_y, consensus_reward


def learn(agent_list, states_var, next_states_var, rewards_var, actions_var, rewards, adjacent_matrix):
    for i in range(N_agent):
        agent_list[i].reward = rewards[i]
    consensus_tao, consensus_y, consensus_rewards = consensus(agent_list, adjacent_matrix)
    for i in range(N_agent):
        target = 0.99 * agent_list[i].value_network(next_states_var.to(device)).detach() + rewards_var[i].to(device)
        agent_list[i].update_tao(consensus_tao[i], target)
    for j in range(N_agent):
        temp = []
        for name, param in agent_list[j].tao.named_parameters():
            temp.append(param)
        i = 0
        for name, param in agent_list[j].value_network.named_parameters():
            param.data = deepcopy(temp[i])
            i += 1
        # agent_list[j].y = consensus_y[j]

    for i in range(N_agent):
        agent_list[i].actor_network_optim.zero_grad()
        log_softmax_actions = agent_list[i].actor_network(states_var.to(device))
        consensus_reward = Variable(torch.Tensor(consensus_rewards[i]).view(-1, 1))
        with torch.no_grad():
            sigma = consensus_reward.to(device) - agent_list[i].value_network(states_var.to(device)) + 0.99 * agent_list[i].value_network(
                next_states_var.to(device))
        actor_network_loss = - torch.mean(torch.sum(log_softmax_actions * actions_var[i].to(device), 1) * sigma.squeeze(1))
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(agent_list[i].actor_network.parameters(), 0.5)
        agent_list[i].actor_network_optim.step()

def roll_out(agent_list, task, sample_nums, init_state):
    states = []
    next_states = []
    actions = [[] for i in range(N_agent)]
    rewards = [[] for i in range(N_agent)]
    state = init_state
    total_reward = 0
    for j in range(sample_nums):
        states.append(state)
        acts = []
        for i in range(N_agent):
            log_softmax_action = agent_list[i].actor_network(Variable(torch.Tensor([state])).to(device))
            softmax_action = torch.exp(log_softmax_action)
            action = np.random.choice(ACTION_DIM, p=softmax_action.cpu().data.numpy()[0])
            if np.random.rand(1) >= 0.95:
                action = np.random.choice(ACTION_DIM)
            acts.append(action)
            one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
            actions[i].append(one_hot_action)
        next_state, reward, done, _ = task.step(acts)
        next_states.append(next_state)
        for i in range(N_agent):
            rewards[i].append(reward[i] * coe[i])
        r = np.mean(reward)
        total_reward += r
        next_state = np.squeeze(np.array(next_state).reshape((1, STATE_DIM)))
        state = next_state
    return states, next_states, actions, rewards, state, total_reward

env = make_env('simple_spread_custom',benchmark=True)
STATE_DIM = env.observation_space[0].shape[0]
N_agent = env.n
STATE_DIM *= N_agent
ACTION_DIM = env.action_space[0].n
seed = 270
coe = [random.uniform(0, 2) for i in range(N_agent)]

er = nx.cycle_graph(N_agent)
adjacent_matrix = np.asarray(nx.to_numpy_matrix(er))


for x in range(10):
    agent_list = [Agent() for i in range(N_agent)]
    for i_episode in range(210):
        init_state = env.reset()
        init_state = np.squeeze(np.array(init_state).reshape((1, STATE_DIM)))
        reward = 0
        for step in range(100):

            states, next_states, actions, rewards, current_state, total_reward = roll_out(agent_list, env, SAMPLE_NUMS,
                                                                        init_state)
            reward += total_reward
            init_state = current_state
            states_var = Variable(torch.Tensor(states).view(-1, STATE_DIM))
            next_states_var = Variable(torch.Tensor(next_states).view(-1, STATE_DIM))
            actions_var = []
            rewards_var = []
            for i in range(N_agent):
                actions_var.append(Variable(torch.Tensor(actions[i]).view(-1, ACTION_DIM)))
                rewards_var.append(Variable(torch.Tensor(rewards[i]).view(-1, 1)))
            learn(agent_list, states_var, next_states_var, rewards_var, actions_var, rewards, adjacent_matrix)
        print("Epoch: %s, Reward: %s " % (i_episode, reward / 1000 ))

