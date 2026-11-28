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

    def q(self, x):
        """Action values from features x = [..., (reward, successor value)]."""
        return self.f(x).squeeze(-1)

    def step(self, V, next_state, reward):
        """One recurrent update. V: [B, S]; next_state, reward: [B, S, A]."""
        B, S, A = next_state.shape
        succ = torch.gather(V, 1, next_state.reshape(B, S * A)).reshape(B, S, A)
        x = torch.stack([reward, succ], dim=-1)  # [B, S, A, 2]
        Q = self.q(x)  # [B, S, A]
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


class ResidualValueUpdater(RecurrentValueUpdater):
    """Bellman structure plus a learned correction.

    Q_t(s,a) = r + gamma * v + eps_theta(r, v), where eps_theta is the same
    2-64-64-1 MLP. Its last layer starts at zero, so the untrained model is exact
    value iteration (zero_init=True). While training, the mean squared correction on every input
    the model sees is stored in `eps_penalty`, so the loss can push eps toward 0.
    With zero_init=False the last layer keeps PyTorch's default random init, so
    training has to remove a random initial correction.
    """

    def __init__(self, hidden=64, gamma=0.9, zero_init=True):
        super().__init__(hidden)
        self.gamma = gamma
        if zero_init:
            nn.init.zeros_(self.f[-1].weight)
            nn.init.zeros_(self.f[-1].bias)
        self.eps_penalty = []

    def q(self, x):
        eps = self.f(x).squeeze(-1)
        if self.training:
            self.eps_penalty.append((eps ** 2).mean())
        return x[..., 0] + self.gamma * x[..., 1] + eps


class ConditionedValueUpdater(nn.Module):
    """Recurrent updater that takes the discount as an input to the update.

    Q_t(s,a) = [base] + f_theta(r, v[, gamma]), where
      - gamma_input: whether f_theta also sees gamma (3 inputs instead of 2);
      - residual:    whether to add the Bellman base r + g * v;
      - base_gamma:  None -> g is the supplied gamma (correct base);
                     float -> g is this fixed value (a misspecified base).
    f_theta is the same 2(or 3)-64-64-1 GELU MLP with PyTorch's default init.
    With residual=True, set `use_correction = False` to run the base operator
    alone, and the mean squared correction is stored in `eps_penalty` while
    training.
    """

    def __init__(self, hidden=64, gamma_input=False, residual=False, base_gamma=None):
        super().__init__()
        self.gamma_input, self.residual, self.base_gamma = gamma_input, residual, base_gamma
        self.f = nn.Sequential(
            nn.Linear(3 if gamma_input else 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.use_correction = True
        self.eps_penalty = []

    def correction(self, r, v, gamma):
        """f_theta(r, v[, gamma]) with r, v: [..., A] and gamma broadcastable to r."""
        feats = [r, v]
        if self.gamma_input:
            feats.append(torch.broadcast_to(gamma, r.shape).to(r.dtype))
        return self.f(torch.stack(feats, dim=-1)).squeeze(-1)

    def q(self, r, v, gamma):
        if not self.residual:
            return self.correction(r, v, gamma)
        g = gamma if self.base_gamma is None else self.base_gamma
        base = r + g * v
        if not self.use_correction:
            return base
        eps = self.correction(r, v, gamma)
        if self.training:
            self.eps_penalty.append((eps ** 2).mean())
        return base + eps

    def step(self, V, next_state, reward, gamma):
        """V: [B, S]; next_state, reward: [B, S, A]; gamma: [B]."""
        B, S, A = next_state.shape
        succ = torch.gather(V, 1, next_state.reshape(B, S * A)).reshape(B, S, A)
        return self.q(reward, succ, gamma.reshape(B, 1, 1).to(reward.dtype)).max(dim=-1).values

    def forward(self, next_state, reward, n_iters, gamma):
        B, S, _ = next_state.shape
        V = reward.new_zeros(B, S)
        values = []
        for _ in range(n_iters):
            V = self.step(V, next_state, reward, gamma)
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
