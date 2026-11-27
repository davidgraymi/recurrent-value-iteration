"""Learned recurrent value updater (VIN-style) and a one-shot MLP baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentValueUpdater(nn.Module):
    """Learned VIN-style value updater.

    One shared MLP f_theta maps [reward(s,a), V_t(next_state(s,a))] -> Q_t(s,a);
    V_{t+1}(s) = max_a Q_t(s,a). The same MLP (same parameters) is applied to
    every state, action, and recurrent iteration. The successor lookup and the
    max are fixed structure; only f_theta is learned.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def step(self, V, next_state, reward):
        """One recurrent update. V: [B, S]; next_state, reward: [B, S, A]."""
        B, S, A = next_state.shape
        succ = torch.gather(V, 1, next_state.reshape(B, S * A)).reshape(B, S, A)
        x = torch.stack([reward, succ], dim=-1)  # [B, S, A, 2]
        Q = self.f(x).squeeze(-1)  # [B, S, A]
        return Q.max(dim=-1).values  # [B, S]

    def forward(self, next_state, reward, n_iters):
        """Run `n_iters` updates from V_0 = 0. Returns [B, n_iters, S] = V_1..V_n."""
        B, S, _ = next_state.shape
        V = reward.new_zeros(B, S)
        values = []
        for _ in range(n_iters):
            V = self.step(V, next_state, reward)
            values.append(V)
        return torch.stack(values, dim=1)


class OneShotMLP(nn.Module):
    """Predict V* directly from the flattened MDP tables.

    Input: one-hot encoding of next_state ([S, A, S] -> S*A*S values) concatenated
    with the flattened reward table (S*A values). For S=8, A=3 this is 216 inputs.
    """

    def __init__(self, n_states=8, n_actions=3, hidden=256):
        super().__init__()
        self.n_states = n_states
        in_dim = n_states * n_actions * n_states + n_states * n_actions
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_states),
        )

    def forward(self, next_state, reward):
        B = next_state.shape[0]
        onehot = F.one_hot(next_state, self.n_states).to(reward.dtype).reshape(B, -1)
        x = torch.cat([onehot, reward.reshape(B, -1)], dim=-1)
        return self.net(x)
