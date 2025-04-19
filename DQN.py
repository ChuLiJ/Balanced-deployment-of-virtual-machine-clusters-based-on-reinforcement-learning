import random
import collections

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
    def __init__(self, state_dim, action_dim, device):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 32)
        self.fc2 = torch.nn.Linear(32, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, action_dim)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ConvolutionalQnet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(state_dim, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x))
        return self.head(x)


class DQN:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = QNet(state_dim, self.action_dim, device).to(device)
        self.target_q_net = QNet(state_dim, self.action_dim, device).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        return self.q_net(state).argmax().item()

    def update(self, batch):
        states = torch.tensor(batch["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float).to(self.device)

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
