import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import keyboard
import numpy as np
from gymnasium import spaces
from pathlib import Path

BATCH_SIZE = 128
GAMMA = 0.99
LR = 3E-4


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"

)
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp, right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)




class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def reset(self):
        self.memory.clear()

class Agent:
    def __init__(self, n_observables, n_actions, device):
        self.memory = ReplayMemory(500000)
        self.policynet = DQN(n_observables, n_actions).to(device)
        self.targetnet = DQN(n_observables, n_actions).to(device)
        self.targetnet.load_state_dict(self.policynet.state_dict())
        self.optimizer = optim.AdamW(self.policynet.parameters(), lr=LR, amsgrad=True)
        self.steps_done = 0

    def target_net(self):
        return self.targetnet
    def policy_net(self):
        return self.policynet
    def soft_update(self, tau):
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.targetnet.state_dict()
            policy_net_state_dict = self.policynet.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau+target_net_state_dict[key]*(1-tau)
            self.targetnet.load_state_dict(target_net_state_dict)
    def greedy_action(self, state):
        with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
            return self.policynet(state).max(1).indices.view(1,1)
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute the mask of a non-final state and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                        if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policynet(state_batch).gather(1, action_batch)

        # Compute V(s{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        #with torch.no_grad():
            #next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        with torch.no_grad():
        # Double DQN: policy_net picks the action, target_net evaluates it
            next_actions = self.policynet(non_final_next_states).max(1).indices.unsqueeze(1)
            next_state_values[non_final_mask] = self.targetnet(non_final_next_states).gather(1, next_actions).squeeze(1)
        # Compute the expected Q values
        expected_state_actions_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_actions_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policynet.parameters(), 100)
        self.optimizer.step()
        return loss.item()
    def store_transition(self, state, action, next_state, reward):
        # save the transition to the agents memory.
        self.memory.push(state, action, next_state, reward)
    def Steps_done(self):
        # counter for action exploration function.
        return self.steps_done
    def save(self, path):
        # save the network to the specified path
        torch.save({
            "policy": self.policynet.state_dict(),
            "target": self.targetnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)

    def load_weights(self, path):
        # attempt to load the weights from file. Return 'False' if it doesn't exist.
        try:
            self.policynet.load_state_dict(torch.load(path)['policy'])
            self.targetnet.load_state_dict(self.policynet.state_dict())
            return True
        except FileNotFoundError:
            return False
    def step(self):
        # increment the steps_done counter for action exploration function.
        self.steps_done += 1