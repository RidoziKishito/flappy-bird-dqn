import torch
from torch import nn, optim
import random

class DQN(nn.Module):
    def __init__(self, state_dim=4, n_actions=2):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, state_dim=4, n_actions=2, lr=1e-3, gamma=0.99, device=None):
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.n_actions = n_actions 
        self.gamma = gamma

        self.policy_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def act(self, state, epsilon=0.1):
        # epsilon-greedy
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_v)
            action = int(q.argmax(dim=1).item())
            return action

    def compute_loss(self, batch):
        # batch: (states, actions, rewards, next_states, dones)
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors and move to device
        states_v = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_v = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_v = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_v = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q(s,a)
        q_values = self.policy_net(states_v).gather(1, actions_v).squeeze(1)

        # target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states_v).max(1)[0]
            expected_q = rewards_v + self.gamma * next_q * (1.0 - dones_v)

        loss = nn.functional.smooth_l1_loss(q_values, expected_q) 
        return loss

    def update(self, replay_buffer, batch_size=64, target_update=1000):
        if len(replay_buffer) < batch_size:
            return None
        batch = replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()


        # Hard target update
        # self.steps_done += 1

        # # Hard target update every N steps
        # if self.steps_done % self.target_update_freq == 0:
        #     self.target_net.load_state_dict(self.policy_net.state_dict())

        # return loss.item()

        # Soft target update
        tau = 0.001  
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + policy_param.data * tau
            )
        return loss.item()

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            # 'optimizer_state': self.optimizer.state_dict() (Optional)
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(data['policy_state_dict'])
        self.target_net.load_state_dict(data['target_state_dict'])

        # self.policy_net.load_state_dict(data['policy'])
        # self.target_net.load_state_dict(data['target'])   


        # # Safe loading optimizer state (optional)
        # try:
        #     self.optimizer.load_state_dict(data['optimizer_state'])
        #     # Move tensors to the correct device
        #     for state in self.optimizer.state.values():
        #         for k, v in state.items():
        #             if isinstance(v, torch.Tensor):
        #                 state[k] = v.to(self.device)
        # except Exception:
        #     # Optimizer load failed
        #     print("Warning: optimizer state could not be fully loaded.")