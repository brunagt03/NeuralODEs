import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generators import (
    TaskType, DataConfig,
    generate_dataset, generate_nested_datasets, generate_grid,
    SAMPLE_SIZES,
)
from flow import ActivationType, make_activation

# ── Global experiment settings ─────────────────────────────────────────────────
DEPTH      = 6
WIDTH      = 6          # flow-activation width; baselines are matched to this
N_FREQ     = 6          # spatial frequency n for all tasks
TEST_N     = 10_000     # fixed held-out test set size
N_SEEDS    = 5          # seeds per (task, activation, N) cell
MAX_STEPS  = 60_000     # gradient steps — same for all experiments
EVAL_EVERY = 200        # steps between test-accuracy checkpoints
LR         = 0.005
CURVE_N    = 2_000      # training set size for learning curve experiment (fixed for all activations)
PATIENCE   = 2_000

FLOW_ACTS  = {ActivationType.CELLULAR, ActivationType.VORTEX, ActivationType.SHEAR}
BASE_ACTS  = {ActivationType.RELU, ActivationType.SIREN}
ALL_ACTS   = [
    ActivationType.CELLULAR,
    ActivationType.VORTEX,
    ActivationType.SHEAR,
    ActivationType.RELU,
    ActivationType.SIREN,
]

COLORS = {
    ActivationType.CELLULAR: "#2196F3",
    ActivationType.VORTEX:   "#FF9800",
    ActivationType.SHEAR:    "#4CAF50",
    ActivationType.RELU:     "#F44336",
    ActivationType.SIREN:    "#9C27B0",
}


# ── Architecture ───────────────────────────────────────────────────────────────

class SimpleResNetBlock(nn.Module):
    def __init__(self, width=3, dt=0.1, activation_type=ActivationType.CELLULAR):
        super().__init__()
        self.dt = dt
        act = make_activation(act_type=activation_type, width=width)
        if activation_type in BASE_ACTS:
            self.feature_extractor = act
        else:
            self.feature_extractor = nn.Sequential(nn.Linear(2, 2), act)
        self.projection = nn.Linear(act.out_features, 2)

    def forward(self, x):
        v = self.projection(self.feature_extractor(x))
        return x + self.dt * v


class FlowResNet(nn.Module):
    def __init__(self, num_blocks=6, width=3, activation=ActivationType.CELLULAR):
        super().__init__()
        self.blocks = nn.ModuleList([
            SimpleResNetBlock(width=width, activation_type=activation)
            for _ in range(num_blocks)
        ])
        self.final_classifier = nn.Linear(2, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_classifier(x)


# ── Parameter matching ─────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_width(flow_act: ActivationType, flow_width: int,
                  depth: int, baseline_act: ActivationType) -> int:
    reference = FlowResNet(num_blocks=depth, width=flow_width, activation=flow_act)
    target_params = count_params(reference)
    for w in range(1, 256):
        candidate = FlowResNet(num_blocks=depth, width=w, activation=baseline_act)
        if count_params(candidate) >= target_params:
            return w
    raise RuntimeError("Could not find a matching width up to 256")


def make_matched_model(activation: ActivationType, depth: int,
                       flow_width: int) -> nn.Module:
    if activation in FLOW_ACTS:
        return FlowResNet(num_blocks=depth, width=flow_width, activation=activation)
    else:
        w = matched_width(ActivationType.CELLULAR, flow_width, depth, activation)
        return FlowResNet(num_blocks=depth, width=w, activation=activation)


# ── Single unified training function ──────────────────────────────────────────

def train_fixed_steps_with_curve(
        model: nn.Module,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_test:  torch.Tensor,
        Y_test:  torch.Tensor,
) -> tuple[nn.Module, list[float]]:
    """
    THE single training function used by all experiments.

    Trains for exactly MAX_STEPS steps with:
      - Adam(lr=LR)
      - gradient clipping (max_norm=1.0)
      - no early stopping, no weight rewinding

    Records test accuracy every EVAL_EVERY steps and returns it as a list.
    The final element of the returned curve equals final_accuracy(model, X_test, Y_test).

    Returns
    -------
    model      : the model after MAX_STEPS (in-place, also returned for convenience)
    curve      : list of length MAX_STEPS // EVAL_EVERY, test accuracy at each checkpoint
    """
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    curve = []

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                preds = (torch.sigmoid(model(X_test)) > 0.5).float()
                acc   = (preds == Y_test).float().mean().item()
            curve.append(acc)

    return model, curve


def final_accuracy(model: nn.Module,
                   X_test: torch.Tensor, Y_test: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_test)) > 0.5).float()
        return (preds == Y_test).float().mean().item()


# ── Training with early stopping (sample-efficiency sweep only) ───────────────

def train_to_convergence(model: nn.Module,
                         X_train: torch.Tensor, Y_train: torch.Tensor,
                         X_val:   torch.Tensor, Y_val:   torch.Tensor,
                         ) -> tuple[nn.Module, list]:
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss    = float("inf")
    steps_no_improve = 0
    best_state       = copy.deepcopy(model.state_dict())
    val_curve        = []

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), Y_val).item()
                preds    = (torch.sigmoid(model(X_val)) > 0.5).float()
                val_acc  = (preds == Y_val).float().mean().item()
            val_curve.append((step, val_acc))

            if val_loss < best_val_loss - 1e-4:
                best_val_loss    = val_loss
                steps_no_improve = 0
                best_state       = copy.deepcopy(model.state_dict())
            else:
                steps_no_improve += EVAL_EVERY
                if steps_no_improve >= PATIENCE:
                    break

    model.load_state_dict(best_state)
    return model, val_curve


# ── Experiment 1: Learning curve ───────────────────────────────────────────────

def run_learning_curve(task: TaskType, out_dir: str) -> tuple[dict, dict]:
    """
    Fix N=CURVE_N. Train every activation for MAX_STEPS steps using
    train_fixed_steps_with_curve. Average test-accuracy curves over N_SEEDS seeds.

    Returns
    -------
    all_curves : dict[ActivationType -> list of per-seed curves]
    all_models : dict[ActivationType -> list of trained models]
                 Returned so run_boundary_comparison can reuse them
                 without retraining. Pass both to that function to
                 skip the redundant training pass.
    """
    print(f"\n{'='*60}")
    print(f"Learning Curve  |  Task: {task.value.upper()}  |  N={CURVE_N}")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    X_test, Y_test, _ = generate_dataset(
        DataConfig(task=task, n=N_FREQ, N=TEST_N, seed=9999)
    )

    all_curves = {act: [] for act in ALL_ACTS}
    all_models = {act: [] for act in ALL_ACTS}

    for act in ALL_ACTS:
        print(f"\n  Activation: {act.value}")
        for seed in range(N_SEEDS):
            datasets = generate_nested_datasets(
                task=task, n=N_FREQ, max_N=CURVE_N, seed=seed
            )
            X_train, Y_train, _ = datasets[CURVE_N]

            model = make_matched_model(act, DEPTH, WIDTH)
            torch.manual_seed(seed)

            model, curve = train_fixed_steps_with_curve(
                model, X_train, Y_train, X_test, Y_test
            )

            all_curves[act].append(curve)
            all_models[act].append(model)
            print(f"    seed={seed}  final_acc={curve[-1]*100:.1f}%")

    # ── Plot ───────────────────────────────────────────────────────────────
    steps = np.arange(EVAL_EVERY, MAX_STEPS + 1, EVAL_EVERY)
    fig, ax = plt.subplots(figsize=(9, 5))

    for act in ALL_ACTS:
        mat   = np.array(all_curves[act])
        means = mat.mean(axis=0)
        stds  = mat.std(axis=0)
        color = COLORS[act]
        ax.plot(steps[:len(means)], means, label=act.value, color=color, lw=2)
        ax.fill_between(steps[:len(means)],
                        means - stds, means + stds,
                        alpha=0.15, color=color)

    ax.set_xlabel("Gradient Steps")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Learning Curve — {task.value.capitalize()} "
                 f"(n={N_FREQ}, N={CURVE_N})\n"
                 f"depth={DEPTH}, matched params, {N_SEEDS} seeds ± 1 std")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{task.value}_learning_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path}")

    return all_curves, all_models


# ── Experiment 2: Decision boundary ───────────────────────────────────────────

def run_boundary_comparison(task: TaskType, out_dir: str,
                            all_curves: dict | None = None,
                            all_models: dict | None = None) -> None:
    """
    Displays decision boundaries for each activation type.

    If all_curves and all_models are provided (returned by run_learning_curve),
    the already-trained models are reused and no retraining occurs.

    If either is None, training is performed from scratch using the identical
    setup as run_learning_curve (same seeds, same function, same steps), so
    results are guaranteed to be consistent either way.

    Displays the MEDIAN seed (the one whose final accuracy is closest to
    the mean across seeds). Reports mean±std in the title.
    """
    print(f"\n{'='*60}")
    print(f"Decision Boundary  |  Task: {task.value.upper()}")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    X_test, Y_test, _ = generate_dataset(
        DataConfig(task=task, n=N_FREQ, N=TEST_N, seed=9999)
    )
    xs, ys, Z_true = generate_grid(resolution=256, task=task, n=N_FREQ)
    grid_pts    = np.c_[xs.ravel(), ys.ravel()]
    grid_tensor = torch.tensor(grid_pts, dtype=torch.float32)

    n_acts = len(ALL_ACTS)
    fig, axes = plt.subplots(2, n_acts + 1, figsize=(3 * (n_acts + 1), 6))

    # Column 0: ground truth
    for row in range(2):
        axes[row, 0].contourf(xs, ys, Z_true, levels=[0, 0.5, 1],
                              cmap="bwr", alpha=0.85)
        axes[row, 0].set_title("Ground Truth")
        axes[row, 0].axis("off")
    axes[0, 0].set_ylabel("Boundary", fontsize=9)
    axes[1, 0].set_ylabel("Confidence", fontsize=9)

    last_im = None

    for col, act in enumerate(ALL_ACTS, start=1):
        seed_probs = []
        seed_accs  = []

        reusing = (all_curves is not None) and (all_models is not None)

        for seed in range(N_SEEDS):
            if reusing:
                # Reuse model trained during run_learning_curve — no retraining
                model = all_models[act][seed]
                acc   = all_curves[act][seed][-1]
            else:
                # Standalone mode: retrain from scratch with identical setup
                datasets = generate_nested_datasets(
                    task=task, n=N_FREQ, max_N=CURVE_N, seed=seed
                )
                X_train, Y_train, _ = datasets[CURVE_N]
                model = make_matched_model(act, DEPTH, WIDTH)
                torch.manual_seed(seed)
                model, curve = train_fixed_steps_with_curve(
                    model, X_train, Y_train, X_test, Y_test
                )
                acc = curve[-1]

            seed_accs.append(acc)
            model.eval()
            with torch.no_grad():
                Z_prob = torch.sigmoid(model(grid_tensor)).numpy().reshape(256, 256)
            seed_probs.append(Z_prob)

            print(f"    {act.value}  seed={seed}  acc={acc*100:.1f}%"
                  + ("  (reused)" if reusing else ""))

        mean_acc   = np.mean(seed_accs)
        std_acc    = np.std(seed_accs)
        median_idx = int(np.argmin(np.abs(np.array(seed_accs) - mean_acc)))
        display_prob = seed_probs[median_idx]
        Z_boundary   = (display_prob > 0.5).astype(float)

        print(f"    -> displaying seed={median_idx} "
              f"(acc={seed_accs[median_idx]*100:.1f}%, "
              f"mean={mean_acc*100:.1f}% ± {std_acc*100:.1f}%)")

        axes[0, col].contourf(xs, ys, Z_boundary, levels=[0, 0.5, 1],
                              cmap="bwr", alpha=0.85)
        axes[0, col].set_title(f"{act.value}\n"
                               f"{mean_acc*100:.1f}%±{std_acc*100:.1f}%",
                               fontsize=9)
        axes[0, col].axis("off")

        last_im = axes[1, col].imshow(
            display_prob, origin="lower", extent=[0, 1, 0, 1],
            cmap="RdBu_r", vmin=0, vmax=1, aspect="auto"
        )
        axes[1, col].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes[1, :], orientation="horizontal",
                     fraction=0.02, pad=0.04, label="P(class=1)")

    fig.suptitle(
        f"Decision Boundaries — {task.value.capitalize()} "
        f"(n={N_FREQ}, N={CURVE_N})\n"
        f"Median seed shown — mean±std over {N_SEEDS} seeds "
        f"(matches learning curve endpoint)",
        fontsize=11, y=1.02
    )
    fig.tight_layout()

    path = os.path.join(out_dir, f"{task.value}_boundaries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Experiment 3: Sample-efficiency sweep ─────────────────────────────────────

def run_sample_efficiency_sweep(task: TaskType, out_dir: str) -> dict:
    """
    Vary N (training set size). For each N, train to convergence (early stopping
    on a fixed validation set) and report test accuracy. Averaged over N_SEEDS.

    Uses train_to_convergence (with early stopping) because for very small N
    there is no point running MAX_STEPS — the model converges in far fewer steps.
    This experiment is about data efficiency, not optimisation speed.
    """
    print(f"\n{'='*60}")
    print(f"Sample Efficiency  |  Task: {task.value.upper()}")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    X_test, Y_test, _ = generate_dataset(
        DataConfig(task=task, n=N_FREQ, N=TEST_N, seed=9999)
    )
    X_val, Y_val, _ = generate_dataset(
        DataConfig(task=task, n=N_FREQ, N=1000, seed=8888)
    )

    results = {act: {"means": [], "stds": []} for act in ALL_ACTS}

    for act in ALL_ACTS:
        print(f"\n  Activation: {act.value}")
        for N in SAMPLE_SIZES:
            seed_accs = []
            for seed in range(N_SEEDS):
                datasets = generate_nested_datasets(
                    task=task, n=N_FREQ, max_N=N, seed=seed
                )
                X_train, Y_train, _ = datasets[N]
                model = make_matched_model(act, DEPTH, WIDTH)
                torch.manual_seed(seed)
                model, _ = train_to_convergence(
                    model, X_train, Y_train, X_val, Y_val
                )
                acc = final_accuracy(model, X_test, Y_test)
                seed_accs.append(acc)

            mean_acc = np.mean(seed_accs)
            std_acc  = np.std(seed_accs)
            results[act]["means"].append(mean_acc)
            results[act]["stds"].append(std_acc)
            print(f"    N={N:<5}  acc={mean_acc*100:.1f}% ± {std_acc*100:.1f}%")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(SAMPLE_SIZES[:len(results[ALL_ACTS[0]]["means"])])

    for act in ALL_ACTS:
        means = np.array(results[act]["means"])
        stds  = np.array(results[act]["stds"])
        color = COLORS[act]
        ax.plot(x, means, marker="o", label=act.value, color=color, lw=2)
        ax.fill_between(x, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Training Samples (N)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Sample Efficiency — {task.value.capitalize()} (n={N_FREQ})\n"
                 f"depth={DEPTH}, matched params, {N_SEEDS} seeds ± 1 std")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{task.value}_efficiency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path}")
    return results


# ── Experiment 4: Spatial frequency sweep ─────────────────────────────────────

def run_spatial_frequency_sweep(task: TaskType, out_dir: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Spatial Frequency Sweep  |  Task: {task.value.upper()}")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)
    freqs   = [2, 4, 6, 8, 10, 12]
    results = {act: {"means": [], "stds": []} for act in ALL_ACTS}

    for act in ALL_ACTS:
        print(f"\n  Activation: {act.value}")
        for n in freqs:
            X_test, Y_test, _ = generate_dataset(
                DataConfig(task=task, n=n, N=TEST_N, seed=9999)
            )
            X_val, Y_val, _ = generate_dataset(
                DataConfig(task=task, n=n, N=1000, seed=8888)
            )
            seed_accs = []
            for seed in range(N_SEEDS):
                cfg = DataConfig(task=task, n=n, N=2000, seed=seed)
                X_train, Y_train, _ = generate_dataset(cfg)
                model = make_matched_model(act, DEPTH, WIDTH)
                torch.manual_seed(seed)
                model, _ = train_to_convergence(
                    model, X_train, Y_train, X_val, Y_val
                )
                acc = final_accuracy(model, X_test, Y_test)
                seed_accs.append(acc)

            mean_acc = np.mean(seed_accs)
            std_acc  = np.std(seed_accs)
            results[act]["means"].append(mean_acc)
            results[act]["stds"].append(std_acc)
            print(f"    n={n:<5}  acc={mean_acc*100:.1f}% ± {std_acc*100:.1f}%")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(freqs)

    for act in ALL_ACTS:
        means = np.array(results[act]["means"])
        stds  = np.array(results[act]["stds"])
        color = COLORS[act]
        ax.plot(x, means, marker="o", label=act.value, color=color, lw=2)
        ax.fill_between(x, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Spatial Frequency (n)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Spatial Frequency Sweep — {task.value.capitalize()}\n"
                 f"depth={DEPTH}, matched params, {N_SEEDS} seeds ± 1 std")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(out_dir, f"{task.value}_frequency_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path}")
    return results


# ── Parameter count table ──────────────────────────────────────────────────────

def print_param_table():
    print(f"\n{'Activation':<12} {'Width':>6} {'Params':>10}")
    print("-" * 32)
    for act in ALL_ACTS:
        model  = make_matched_model(act, DEPTH, WIDTH)
        params = count_params(model)
        w = WIDTH if act in FLOW_ACTS else matched_width(
            ActivationType.CELLULAR, WIDTH, DEPTH, act
        )
        print(f"{act.value:<12} {w:>6} {params:>10,}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    ROOT = f"results_last_version_clip_depth={DEPTH}_width={WIDTH}"

    print_param_table()

    for task in TaskType:
        curves, models = run_learning_curve(
            task    = task,
            out_dir = f"{ROOT}/learning_curves",
        )
        run_boundary_comparison(
            task       = task,
            out_dir    = f"{ROOT}/boundaries",
            all_curves = curves,
            all_models = models,
        )
        # run_sample_efficiency_sweep(
        #     task    = task,
        #     out_dir = f"{ROOT}/sample_efficiency",
        # )
