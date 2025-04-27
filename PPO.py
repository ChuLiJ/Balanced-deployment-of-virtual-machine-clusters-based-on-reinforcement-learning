import torch.nn
import torch.nn.functional as F

from utils import compute_advantage


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

        return actor_loss, critic_loss
