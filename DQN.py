import collections
import random

import torch.nn
import torch.nn.functional as F
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)

    def size(self):
        return len(self.buffer)


class QNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 256)
        self.fc2 = torch.nn.Linear(256, 512)
        self.fc3 = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon, target_update):
        self.action_dim = action_dim
        self.q_net = QNet(state_dim, self.action_dim)
        self.target_q_net = QNet(state_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0

    def take_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state = np.array(state)

        state = torch.tensor(state, dtype=torch.float)
        return self.q_net(state).argmax().item()

    def update(self, batch):
        states = torch.tensor(batch["states"], dtype=torch.float)
        actions = torch.tensor(batch["actions"], dtype=torch.long).view(-1, 1)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values

        loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1
