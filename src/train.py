"""Train the learned recurrent value updater and the one-shot baseline.

Usage:
    python -m src.train                 # default config (10k train MDPs, 100 epochs)
    python -m src.train --small         # fallback config (5k train MDPs, 50 epochs)
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from src.mdp import make_dataset
from src.models import OneShotMLP, RecurrentValueUpdater

SEED = 42
TRAIN_ITERS = 16
CKPT_DIR = "checkpoints"
RESULTS_DIR = "results"


def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch (CPU and MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(name="auto"):
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def to_tensors(data, device):
    """Convert a NumPy dataset dict into float32 / int64 torch tensors on `device`."""
    return {
        "next_state": torch.as_tensor(data["next_state"], dtype=torch.long, device=device),
        "reward": torch.as_tensor(data["reward"], dtype=torch.float32, device=device),
        # Targets V_1..V_16 for the recurrent model.
        "targets": torch.as_tensor(
            data["trajectory"][:, 1 : TRAIN_ITERS + 1], dtype=torch.float32, device=device
        ),
        "v_star": torch.as_tensor(data["v_star"], dtype=torch.float32, device=device),
    }


def recurrent_loss(model, batch):
    pred = model(batch["next_state"], batch["reward"], TRAIN_ITERS)  # [B, 16, S]
    # mean over iterations, states and MDPs
    return ((pred - batch["targets"]) ** 2).mean()


def oneshot_loss(model, batch):
    pred = model(batch["next_state"], batch["reward"])
    return ((pred - batch["v_star"]) ** 2).mean()


def batches(data, idx, batch_size):
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        yield {k: v[b] for k, v in data.items()}


@torch.no_grad()
def evaluate_loss(model, loss_fn, data, batch_size):
    model.eval()
    n = data["reward"].shape[0]
    idx = torch.arange(n, device=data["reward"].device)
    total = 0.0
    for batch in batches(data, idx, batch_size):
        total += loss_fn(model, batch).item() * batch["reward"].shape[0]
    return total / n


def train_model(name, model, loss_fn, train, val, args, device,
                seed=SEED, ckpt_dir=CKPT_DIR, val_loss_fn=None, verbose=True):
    """Adam + grad clipping + early stopping; saves the best-validation checkpoint.

    `val_loss_fn` defaults to `loss_fn`; pass a different one when the training
    loss has extra terms that should not decide early stopping.
    """
    set_seed(seed)
    val_loss_fn = val_loss_fn or loss_fn
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(seed)
    n_train = train["reward"].shape[0]
    best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    history = []
    ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=gen).to(device)  # reshuffle each epoch
        running = 0.0
        for batch in batches(train, perm, args.batch_size):
            loss = loss_fn(model, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item() * batch["reward"].shape[0]
        train_loss = running / n_train
        val_loss = evaluate_loss(model, val_loss_fn, val, args.batch_size)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                ckpt_path,
            )
        else:
            bad_epochs += 1
        if verbose:
            print(
                f"[{name}] epoch {epoch:3d}  train {train_loss:.3e}  val {val_loss:.3e}"
                f"{'  *' if improved else ''}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
        if bad_epochs >= args.patience:
            print(f"[{name}] early stopping at epoch {epoch}", flush=True)
            break

    elapsed = time.time() - t0
    print(f"[{name}] best val {best_val:.3e} at epoch {best_epoch}; {elapsed:.0f}s", flush=True)
    return {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "seconds": elapsed,
        "history": history,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=10_000)
    p.add_argument("--n-val", type=int, default=1_000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--small", action="store_true", help="5k train MDPs, 50 epochs")
    p.add_argument("--only", choices=["recurrent", "oneshot"], default=None)
    args = p.parse_args()
    if args.small:
        args.n_train, args.epochs = 5_000, 50

    set_seed(SEED)
    device = get_device(args.device)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"device: {device}", flush=True)

    t0 = time.time()
    train = to_tensors(make_dataset("train", args.n_train), device)
    val = to_tensors(make_dataset("val", args.n_val), device)
    print(f"datasets ready in {time.time() - t0:.1f}s", flush=True)

    summary_path = os.path.join(RESULTS_DIR, "train_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    summary["config"] = {k: v for k, v in vars(args).items()} | {
        "seed": SEED,
        "train_iters": TRAIN_ITERS,
        "device": str(device),
    }

    if args.only in (None, "recurrent"):
        summary["recurrent"] = train_model(
            "recurrent", RecurrentValueUpdater(), recurrent_loss, train, val, args, device
        )
    if args.only in (None, "oneshot"):
        summary["oneshot"] = train_model(
            "oneshot", OneShotMLP(), oneshot_loss, train, val, args, device
        )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"total training time {time.time() - t0:.0f}s; wrote {summary_path}")


if __name__ == "__main__":
    main()
