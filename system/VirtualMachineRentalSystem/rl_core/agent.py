import random
import collections
import torch.nn
import torch.nn.functional as F
import numpy as np

from collections import deque

from .utils import SumTree, NoisyLinear, compute_advantage


class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def push(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) < self.n:
            return None

        R = 0
        for i in range(self.n):
            R += self.gamma ** i * self.buffer[i][2]

        state, action, _, _, _ = self.buffer[0]
        _, _, _, next_state, done = self.buffer[-1]

        n_step_transition = (state, action, R, next_state, done)
        self.buffer.popleft()
        return n_step_transition

    def flush(self):
        transitions = []
        while len(self.buffer) > 0:
            state, action, reward, _, _ = self.buffer[0]
            R = 0
            for i, (_, _, r, _, _) in enumerate(self.buffer):
                R += (self.gamma ** i) * r
            next_state, _, _, ns, done = self.buffer[-1]
            transitions.append((state, action, R, ns, done))
            self.buffer.popleft()
        return transitions


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, experience):
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size):
        indices = []
        experiences = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(p)
            experiences.append(data)

        sampling_probs = np.array(priorities) / self.tree.total()
        is_weights = np.power(len(self.tree) * sampling_probs, -self.beta)
        is_weights /= is_weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)
        return indices, experiences, is_weights

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (np.abs(td) + 1e-5) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def size(self):
        return len(self.tree)


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
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, action_dim)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DuelingQNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(DuelingQNet, self).__init__()
        self.fc1 = NoisyLinear(state_dim, 64)
        self.fc2 = NoisyLinear(64, 128)

        self.V_stream = NoisyLinear(128, 64)
        self.V = NoisyLinear(64, 1)

        self.A_stream = NoisyLinear(128, 64)
        self.A = NoisyLinear(64, action_dim)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        V = F.relu(self.V_stream(x))
        V = self.V(V)

        A = F.relu(self.A_stream(x))
        A = self.A(A)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

    def reset_noisy(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.V_stream.reset_noise()
        self.V.reset_noise()
        self.A_stream.reset_noise()
        self.A.reset_noise()


class RainbowQNet(DuelingQNet):
    def __init__(self, state_dim, action_dim, atoms, device):
        super(RainbowQNet, self).__init__(state_dim, action_dim, device)
        self.atoms = atoms
        self.action_dim = action_dim
        self.V = NoisyLinear(64, atoms)
        self.A = NoisyLinear(64, action_dim * atoms)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        V = F.relu(self.V_stream(x))
        V = self.V(V)

        A = F.relu(self.A_stream(x))
        A = self.A(A).view(-1, self.action_dim, self.atoms)

        Q = V.unsqueeze(1) + (A - A.mean(dim=1, keepdim=True))
        return Q


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

        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss.item()


class DoubleDQN(DQN):
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        super(DoubleDQN, self).__init__(state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)

    def update(self, batch):
        states = torch.tensor(batch["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdims=True)  # 与普通DQN不同的地方
            max_next_q_values = self.target_q_net(next_states).gather(1, next_actions)

            q_targets = rewards + self.gamma * max_next_q_values

        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss.item()


class DDDQN(DoubleDQN):
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        super().__init__(state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)
        self.q_net = DuelingQNet(state_dim, action_dim, device=device).to(device)
        self.target_q_net = DuelingQNet(state_dim, action_dim, device=device).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def take_action(self, state, epsilon=None):
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        return self.q_net(state).argmax().item()

    def update(self, batch):
        states = torch.tensor(batch["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float).to(self.device)
        is_weights = batch["is_weights"].to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdims=True)  # 与普通DQN不同的地方
            max_next_q_values = self.target_q_net(next_states).gather(1, next_actions)
            q_targets = rewards + self.gamma * max_next_q_values

        td_errors = (q_targets - q_values).detach().cpu().numpy().flatten()

        loss = (is_weights * F.mse_loss(q_values, q_targets, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.q_net.reset_noisy()
        self.target_q_net.reset_noisy()

        return loss.item(), td_errors


class RainbowDQN(DDDQN):
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                 atoms=51, v_min=-10, v_max=10):
        super(RainbowDQN, self).__init__(state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = atoms
        self.delta_z = (v_max - v_min) / (atoms - 1)
        self.support = torch.linspace(v_min, v_max, atoms).to(device)
        self.q_net = RainbowQNet(state_dim, action_dim, atoms, device)
        self.target_q_net = RainbowQNet(state_dim, action_dim, atoms, device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

    def take_action(self, state, epsilon=None):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.q_net(state)
            probs = F.softmax(logits, dim=-1)
            q_values = torch.sum(probs * self.support, dim=-1)
        return q_values.argmax().item()

    def update(self, batch):
        states = torch.tensor(batch["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long).to(self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float).to(self.device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float).to(self.device)
        is_weights = batch["is_weights"].to(self.device)
        batch_size = states.size(0)

        current_logits = self.q_net(states)
        actions = actions.view(batch_size, 1, 1).expand(-1, 1, self.atoms)
        selected_logits = current_logits.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_logits = self.q_net(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_q = torch.sum(next_probs * self.support, dim=-1)
            next_actions = next_q.argmax(dim=1)

            target_next_logits = self.target_q_net(next_states)
            target_next_probs = F.softmax(target_next_logits, dim=-1)
            next_actions = next_actions.view(batch_size, 1, 1).expand(-1, 1, self.atoms)
            target_next_probs = target_next_probs.gather(1, next_actions).squeeze(1)

            target_support = rewards.unsqueeze(1) + self.gamma * self.support.unsqueeze(0)
            target_support = torch.clamp(target_support, self.v_min, self.v_max)

            # 计算分布投影
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().clamp(0, self.atoms - 1).long()
            u = b.ceil().clamp(0, self.atoms - 1).long()

            # 分配概率
            target_probs = torch.zeros_like(target_next_probs)
            offset = torch.arange(batch_size, device=self.device).unsqueeze(1) * self.atoms
            l_flat = (l + offset).view(-1)
            u_flat = (u + offset).view(-1)
            d_to_l = ((u.float() - b) * target_next_probs).view(-1)
            d_to_u = ((b - l.float()) * target_next_probs).view(-1)

            target_probs.view(-1).scatter_add_(0, l_flat, d_to_l)
            target_probs.view(-1).scatter_add_(0, u_flat, d_to_u)

            # 计算KL散度损失
        log_prob = F.log_softmax(selected_logits, dim=1)
        kl_div = F.kl_div(log_prob, target_probs, log_target=False, reduction='none').sum(dim=1)
        loss = (kl_div * is_weights.squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 计算TD误差（基于期望）
        with torch.no_grad():
            current_probs = F.softmax(selected_logits, dim=1)
            current_q = torch.sum(current_probs * self.support, dim=1)
            target_q = torch.sum(target_probs * self.support, dim=1)
            td_errors = (target_q - current_q).abs().cpu().numpy()

        # 更新目标网络和噪声
        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_net.reset_noisy()
        self.target_q_net.reset_noisy()

        self.scheduler.step(loss)

        return loss.item(), td_errors


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, device):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 3)
        self.log_std = torch.nn.Parameter(torch.zeros(3))
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x))
        x2 = F.relu(self.fc3(x1) + x1)
        x3 = F.relu(self.fc4(x2))
        out = self.fc5(x3)
        mu = F.sigmoid(out)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, device):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 1)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x))
        x2 = F.relu(self.fc3(x1) + x1)
        x3 = F.relu(self.fc4(x2))
        return self.fc5(x3)


class PPO:
    def __init__(self, state_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, device)
        self.critic = ValueNet(state_dim, device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.5,
                                                                          patience=5)
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.critic_optimizer, mode='min', factor=0.5
                                                                           , patience=5)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dict = torch.distributions.Normal(mu, std)
        action = action_dict.rsample()
        action = torch.clamp(action, 0.0, 1.0)
        return action.squeeze().tolist()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        with torch.no_grad():
            mu, std = self.actor(states)
            old_dist = torch.distributions.Normal(mu, std)
            old_log_probs = old_dist.log_prob(actions).sum(dim=-1, keepdim=True)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            dist = torch.distributions.Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

        self.actor_scheduler.step(actor_loss)
        self.critic_scheduler.step(critic_loss)

        return actor_loss, critic_loss
