"""Random deterministic MDPs and exact (NumPy) value iteration.

Everything here is independent of the learned models: it uses NumPy in float64
and never imports torch.
"""

import numpy as np

N_STATES = 8
N_ACTIONS = 3
GAMMA = 0.9

# Fixed, distinct seeds for each split so the three datasets are independent.
DATA_SEEDS = {"train": 1000, "val": 2000, "test": 3000}


def generate_mdps(n, seed, n_states=N_STATES, n_actions=N_ACTIONS):
    """Generate `n` random deterministic MDPs.

    Returns
    -------
    next_state : int64 array [n, n_states, n_actions], successor state of (s, a)
    reward     : float64 array [n, n_states, n_actions], uniform in [-1, 1]
    """
    rng = np.random.default_rng(seed)
    next_state = rng.integers(0, n_states, size=(n, n_states, n_actions), dtype=np.int64)
    reward = rng.uniform(-1.0, 1.0, size=(n, n_states, n_actions))
    return next_state, reward


def q_values(V, next_state, reward, gamma=GAMMA):
    """Q(s, a) = reward(s, a) + gamma * V(next_state(s, a)), batched.

    V: [n, S]; next_state, reward: [n, S, A]. Returns [n, S, A].
    """
    n = V.shape[0]
    successor_values = V[np.arange(n)[:, None, None], next_state]
    return reward + gamma * successor_values


def bellman_optimality(V, next_state, reward, gamma=GAMMA):
    """Bellman optimality operator: (T V)(s) = max_a Q(s, a). Returns [n, S]."""
    return q_values(V, next_state, reward, gamma).max(axis=-1)


def value_iteration(next_state, reward, gamma=GAMMA, tol=1e-8, max_iter=500):
    """Exact value iteration from V_0 = 0, batched over MDPs.

    Iterates until max |V_{t+1} - V_t| < tol for every MDP in the batch, or
    until `max_iter` iterations. Once an MDP has converged its later iterates
    are numerically unchanged, so running the batch jointly is equivalent to
    running each MDP on its own.

    Returns
    -------
    trajectory : [n, T+1, S] array with trajectory[:, t] = V_t (V_0 = 0).
    v_star     : [n, S] converged values (= trajectory[:, -1]).
    n_iters    : [n] int array, iteration at which each MDP first met `tol`.
    """
    n, S, _ = next_state.shape
    V = np.zeros((n, S))
    traj = [V]
    n_iters = np.full(n, max_iter, dtype=np.int64)
    for t in range(1, max_iter + 1):
        V_new = bellman_optimality(V, next_state, reward, gamma)
        diff = np.abs(V_new - V).max(axis=-1)
        newly = (diff < tol) & (n_iters == max_iter)
        n_iters[newly] = t
        traj.append(V_new)
        V = V_new
        if (diff < tol).all():
            break
    trajectory = np.stack(traj, axis=1)
    return trajectory, V, n_iters


def exact_trajectory(next_state, reward, n_steps, gamma=GAMMA):
    """Exactly `n_steps` Bellman updates from V_0 = 0. Returns [n, n_steps+1, S]."""
    n, S, _ = next_state.shape
    V = np.zeros((n, S))
    traj = [V]
    for _ in range(n_steps):
        V = bellman_optimality(V, next_state, reward, gamma)
        traj.append(V)
    return np.stack(traj, axis=1)


def make_dataset(split, n):
    """Generate a split and its exact solution. Returns a dict of NumPy arrays."""
    next_state, reward = generate_mdps(n, DATA_SEEDS[split])
    trajectory, v_star, n_iters = value_iteration(next_state, reward)
    return {
        "next_state": next_state,
        "reward": reward,
        "trajectory": trajectory,
        "v_star": v_star,
        "n_iters": n_iters,
    }
