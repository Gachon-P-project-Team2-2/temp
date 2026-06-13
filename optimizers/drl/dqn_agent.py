from __future__ import annotations

import numpy as np


class IndependentDQNAgent:
    """Shared DQN for independent base-station on/off control."""

    def __init__(self, state_dim: int = 2, action_dim: int = 2, lr: float = 0.01, gamma: float = 0.9):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        class DQNNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(state_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, action_dim),
                )

            def forward(self, x):
                return self.layers(x)

        self._torch = torch
        self.gamma = float(gamma)
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        self.model = DQNNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def get_actions_batch(self, states_matrix: np.ndarray, train: bool = True) -> np.ndarray:
        torch = self._torch
        states_matrix = np.asarray(states_matrix, dtype=float)
        k = states_matrix.shape[0]
        actions = np.zeros(k, dtype=int)

        if train:
            explore_mask = np.random.rand(k) < self.epsilon
            actions[explore_mask] = np.random.randint(0, 2, size=int(np.sum(explore_mask)))
            exploit_mask = ~explore_mask
        else:
            exploit_mask = np.ones(k, dtype=bool)

        if np.any(exploit_mask):
            states_t = torch.FloatTensor(states_matrix[exploit_mask])
            with torch.no_grad():
                q_values = self.model(states_t)
            actions[exploit_mask] = torch.argmax(q_values, dim=1).numpy()

        return actions

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        torch = self._torch
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)

        current_q = self.model(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.model(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
