# Experiment: Anytime Refinement of Recurrent Value Estimates
## Objective

Test whether a small learned recurrent model can improve its estimate of optimal state values as inference computation increases, including at iteration counts beyond its training horizon.

The experiment is inspired by Value Iteration Networks (VIN) and recurrent value iteration. It is not intended to establish novelty over VIN.

The primary research question is:

**Does a learned recurrent value updater continue to reduce value error and action regret when allowed to run for more iterations than it was trained on, or does it plateau, oscillate, or degrade?**

Implement this as a small, reproducible PyTorch experiment that trains and evaluates in under one hour on an Apple Silicon MacBook.

## 1. Environment

Generate a dataset of small deterministic discounted Markov decision processes (MDPs).

Each MDP has:

- 8 states
- 3 actions per state
- Deterministic transitions
- Random transition destinations
- Random rewards in the range [-1, 1]
- Discount factor γ = 0.9

Represent each MDP using:

- `next_state`: integer tensor of shape `[8, 3]`
- `reward`: float tensor of shape `[8, 3]`

For each state-action pair, `next_state[s, a]` identifies the next state and `reward[s, a]` is the immediate reward.

Generate independent random MDPs for training, validation, and testing. Do not use terminal states. The discount factor makes the infinite-horizon optimal value well-defined.

Use a fixed seed for all dataset generation.

## 2. Exact reference solution

Implement exact value iteration as the ground-truth reference.

Initialize:

`V_0(s) = 0`

Apply the Bellman optimality operator:

`V_{t+1}(s) = max_a [ reward(s,a) + γ V_t(next_state(s,a)) ]`

Run value iteration until the maximum absolute difference between consecutive value vectors is below `1e-8`, or until 500 iterations have been reached.

Store the complete sequence of value vectors during iteration.

The converged value vector is `V*`. The exact value at iteration `t` is `V_t`.

This reference must be implemented independently of the learned model.

## 3. Learned recurrent updater

Implement a shared recurrent value-update model.

The recurrent state is the value vector:

`V_t ∈ R^8`

At each iteration, for every state-action pair, gather the value of its successor state:

`x_t(s,a) = [ reward(s,a), V_t(next_state(s,a)) ]`

Pass this two-dimensional feature vector through a shared MLP:

`Q_t(s,a) = f_θ(x_t(s,a))`

Use this architecture:

- Linear: 2 → 64
- GELU
- Linear: 64 → 64
- GELU
- Linear: 64 → 1

The MLP must share weights across all states, actions, and recurrent iterations.

Aggregate action values using an explicit maximum:

`V_{t+1}(s) = max_a Q_t(s,a)`

Initialize `V_0 = 0`.

The model must use the same parameters at every iteration. Do not create separate layers for different iteration depths.

The transition lookup and max operation are known structural components. The MLP learns the value-update function. This is a deliberately small learned Bellman-style updater, not a claim that the entire planning process is learned from scratch.

## 4. Training

Train the recurrent updater to approximate exact value-iteration trajectories.

For each training MDP, calculate the exact reference values:

`V_1, V_2, ..., V_16`

Run the learned model for 16 recurrent iterations, producing:

`V̂_1, V̂_2, ..., V̂_16`

Use mean squared error at every iteration:

`L = mean_{t=1..16} mean_s (V̂_t(s) - V_t(s))²`

Average the loss across all iterations and all MDPs in the batch.

Training configuration:

- Training MDPs: 10,000
- Validation MDPs: 1,000
- Batch size: 128
- Optimizer: Adam
- Learning rate: 1e-3
- Maximum epochs: 100
- Early stopping: validation loss does not improve for 10 epochs
- Gradient clipping: global norm 1.0
- Random seed: 42

Shuffle training MDPs every epoch.

Evaluate validation loss after each epoch. Save the checkpoint with the lowest validation loss.

If training exceeds the one-hour target, reduce the training set to 5,000 MDPs and the maximum epochs to 50. Do not change the test set.

## 5. Baselines

Implement the following baselines.

### A. Exact value iteration

Run the exact Bellman update from `V_0 = 0`.

This establishes the best possible value error at every iteration and confirms the environment behaves as expected.

### B. One-shot neural network

Implement a feed-forward MLP that receives the flattened transition and reward tables for an MDP and predicts `V*` directly.

Architecture:

- Input: flattened transition and reward representation
- Linear: input → 256
- GELU
- Linear: 256 → 256
- GELU
- Linear: 256 → 8

Train it using supervised MSE against `V*`.

This baseline tests whether iterative computation provides a benefit over direct prediction.

### C. Learned recurrent updater

Use the shared recurrent model described above.

This is the primary model under investigation.

Do not label it as a novel architecture. Refer to it as the learned recurrent value updater or learned VIN-style updater.

## 6. Evaluation

Evaluate on 1,000 previously unseen test MDPs.

For the recurrent model, run inference for:

`1, 2, 4, 8, 16, 24, 32, 48, 64, 96`

The model was trained for 16 iterations. Iterations 24 through 96 are out-of-training-horizon evaluations.

Record the value vector after every iteration from 1 through 96, not only the selected checkpoints.

For each test MDP and each iteration, calculate:

### Value error

`value_error(t) = mean_s |V̂_t(s) - V*(s)|`

### Maximum value error

`max_value_error(t) = max_s |V̂_t(s) - V*(s)|`

### Action regret

Choose the greedy action using the learned value estimate:

`â(s) = argmax_a [ reward(s,a) + γ V̂_t(next_state(s,a)) ]`

Calculate its true one-step optimal action value using `V*`:

`Q*(s,a) = reward(s,a) + γ V*(next_state(s,a))`

Then calculate:

`regret(t) = mean_s [ max_a Q*(s,a) - Q*(s,â(s)) ]`

Also report the percentage of states where the learned greedy action is optimal.

For the one-shot model, report value error and action regret. It has no recurrent iteration count.

## 7. Primary hypothesis and success criteria

The primary hypothesis is that the learned recurrent updater will continue to improve its value estimate after iteration 16, even though it was trained only through iteration 16.

Report:

- Mean and median value error at every iteration
- Mean and median action regret at every iteration
- Percentage of test MDPs whose value error improves from iteration 16 to iteration 32
- Percentage of test MDPs whose value error improves from iteration 32 to iteration 64
- Percentage of test MDPs whose value error improves from iteration 64 to iteration 96
- Whether the model plateaus, oscillates, or diverges

Define continued improvement as a reduction in median test value error over the specified interval.

Do not require strict improvement at every individual iteration. Recurrent updates may fluctuate.

The experiment is considered a positive result for anytime refinement if:

1. Median value error decreases from iteration 16 to iteration 64.
2. Median action regret also decreases over that interval.
3. The improvement occurs on unseen MDPs.
4. The model does not simply collapse to a constant value or diverge at later iterations.

These criteria are experimental indicators, not proof of a general capability.

## 8. Required plots

Generate the following plots:

1. Median value error versus recurrent iteration, with a logarithmic y-axis.
2. Median action regret versus recurrent iteration.
3. Percentage of test MDPs improving over each iteration interval.
4. Exact value iteration error versus iteration.
5. Learned recurrent error versus exact value iteration error.

Mark iteration 16 clearly as the training horizon.

Include the one-shot model's value error and regret as horizontal reference lines on the relevant plots.

Save all plots as PNG files.

## 9. Required implementation

Create a clean Python project with the following structure:

`src/mdp.py`
- Random MDP generation
- Exact value iteration
- Bellman optimality calculation

`src/models.py`
- Learned recurrent updater
- One-shot baseline

`src/train.py`
- Training loop
- Validation
- Checkpoint saving
- Early stopping

`src/evaluate.py`
- Test evaluation
- Per-iteration metrics
- Baseline comparisons

`src/plots.py`
- Plot generation

`tests/`
- Unit tests for MDP generation
- Unit tests for exact value iteration
- Tests for recurrent tensor shapes
- Tests that recurrent parameters are shared across iterations
- Tests that the exact Bellman operator converges
- Tests that evaluation metrics are correct

`README.md`
- Experiment overview
- Setup instructions
- Commands to train and evaluate
- Expected outputs
- Explanation of the hypothesis and limitations

Include a `requirements.txt` or `pyproject.toml`.

## 10. Reproducibility and runtime

Use PyTorch and NumPy.

Support Apple Silicon MPS when available, with CPU fallback.

Set and document random seeds for Python, NumPy, and PyTorch.

The entire experiment should be runnable with a small number of commands:

1. Install dependencies.
2. Train models.
3. Evaluate.
4. Generate plots.

The default configuration must complete in under one hour on a MacBook with 16 GB RAM.

Prefer simple, readable code over abstraction. Avoid unnecessary frameworks, large datasets, external APIs, and complicated environment dependencies.

## 11. Interpretation

The experiment should distinguish three possible outcomes:

**Continued refinement:** The learned updater keeps reducing value error and action regret beyond iteration 16.

**Fixed-point plateau:** The learned updater converges quickly to a stable estimate, but additional iterations do not improve it.

**Instability:** The learned updater's error oscillates or grows at larger iteration counts.

Compare these outcomes with exact value iteration. Exact value iteration should converge because the discounted Bellman operator is a contraction. The learned operator is not guaranteed to be a contraction, so its long-horizon behavior is the main object of study.

Do not claim that this experiment proves recurrent models can replace MCTS. It is a controlled first test of whether a learned recurrent value update can support useful inference-time computation.

The final report should state clearly whether the model improved beyond its training horizon, how much it improved, and whether the improvement was accompanied by lower action regret.