"""Run the attractor-state detection pipeline.

Two stages: (1) a multi-seed comparison of GMM, HMM, and a shuffled
null model, and (2) a single full run that saves networks and plots.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import (
    agreement,
    bootstrap_edge_count,
    detect_states_gmm,
    detect_states_hmm,
    fit_state_networks,
    network_density_summary,
    state_persistence_summary,
)
from simulate import simulate_attractor_series
from visualize import (
    plot_pca_state_space,
    plot_state_network,
    plot_state_timeline,
)


OUT_DIR = "output"
N_TIMEPOINTS = 1000
N_SYMPTOMS = 5
N_STATES = 2
GMM_WINDOW = 5
N_SEEDS = 50
N_BOOTSTRAP = 200
CI_LEVEL = 95.0


def _symptom_columns(data):
    """Names of the symptom columns (everything except time)."""
    return [c for c in data.columns if c != "time"]


def _load_data(seed):
    """Simulate one dataset. Returns (data, X, true_states)."""
    data, true_states = simulate_attractor_series(
        n_timepoints=N_TIMEPOINTS,
        n_symptoms=N_SYMPTOMS,
        n_states=N_STATES,
        seed=seed,
    )
    X = data[_symptom_columns(data)].values
    return data, X, true_states


def run_multi_seed_comparison():
    """Compare GMM, HMM, and a shuffled null across multiple seeds."""
    print(f"Running {N_SEEDS} seeds")

    gmm_accs, hmm_accs, null_accs = [], [], []

    for seed in range(N_SEEDS):
        _, X, true_states = _load_data(seed)

        states_gmm = detect_states_gmm(
            X, n_states=N_STATES, window=GMM_WINDOW, seed=seed
        )
        gmm_accs.append(agreement(states_gmm, true_states, N_STATES))

        states_hmm = detect_states_hmm(X, n_states=N_STATES, seed=seed)
        hmm_accs.append(agreement(states_hmm, true_states, N_STATES))

        # Null: shuffle time points so any temporal structure is destroyed.
        # Its floor is ~50% with 2 states, because agreement takes the best
        # label permutation.
        rng = np.random.default_rng(seed)
        X_shuffled = X[rng.permutation(len(X))]
        states_null = detect_states_gmm(
            X_shuffled, n_states=N_STATES, window=GMM_WINDOW, seed=seed
        )
        null_accs.append(agreement(states_null, true_states, N_STATES))

    df = pd.DataFrame({
        "method": ["GMM", "HMM", "Null (shuffled)"],
        "mean_agreement": [
            np.mean(gmm_accs),
            np.mean(hmm_accs),
            np.mean(null_accs),
        ],
        "std_agreement": [
            np.std(gmm_accs),
            np.std(hmm_accs),
            np.std(null_accs),
        ],
    })
    print(df.to_string(index=False))
    return df


def run_full_pipeline(seed=42):
    """Run the pipeline end to end on a single seed and save outputs."""
    print(f"\nFull pipeline on seed={seed}")
    os.makedirs(OUT_DIR, exist_ok=True)

    data, X, true_states = _load_data(seed)
    symptom_cols = _symptom_columns(data)
    data.to_csv(os.path.join(OUT_DIR, "simulated_data.csv"), index=False)

    states = detect_states_hmm(X, n_states=N_STATES, seed=seed)
    acc = agreement(states, true_states, N_STATES)
    print(f"Agreement with ground truth: {acc:.1%}")

    persistence = state_persistence_summary(states, n_states=N_STATES)
    persistence.to_csv(os.path.join(OUT_DIR, "state_summary.csv"), index=False)
    print("\nState persistence:")
    print(persistence.to_string(index=False))

    networks, symptom_names = fit_state_networks(
        X, states, symptom_names=symptom_cols
    )

    density = network_density_summary(networks, symptom_names)
    print("\nNetwork density:")
    print(density.to_string(index=False))

    # Only states that got a network (enough observations) are bootstrapped.
    print("\nBootstrap edge count:")
    boot_rows = []
    for s in networks:
        mean_e, ci_low, ci_high = bootstrap_edge_count(
            X, states, state=s, n_boot=N_BOOTSTRAP, ci=CI_LEVEL
        )
        boot_rows.append({
            "state": s,
            "mean_edges": round(mean_e, 2),
            "ci_low": round(ci_low, 2),
            "ci_high": round(ci_high, 2),
        })
        print(
            f"  state {s}: {mean_e:.2f} "
            f"({CI_LEVEL:g}% CI: {ci_low:.2f}-{ci_high:.2f})"
        )

    pd.DataFrame(boot_rows).to_csv(
        os.path.join(OUT_DIR, "bootstrap_edge_count.csv"), index=False
    )

    print("\nSaving plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    plot_state_timeline(states, ax=axes[0])
    plot_pca_state_space(X, states, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "state_overview.png"), dpi=150)
    plt.close(fig)

    # One panel per state. plt.subplots returns a bare Axes when n == 1.
    n_found = len(networks)
    fig, axes = plt.subplots(1, n_found, figsize=(6 * n_found, 6))
    if n_found == 1:
        axes = [axes]
    for ax, s in zip(axes, networks.keys()):
        plot_state_network(networks[s], symptom_names, state_label=s, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "networks.png"), dpi=150)
    plt.close(fig)

    print(f"\nResults saved to ./{OUT_DIR}/")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    multi_seed_df = run_multi_seed_comparison()
    multi_seed_df.to_csv(
        os.path.join(OUT_DIR, "multi_seed_results.csv"), index=False
    )

    run_full_pipeline(seed=42)


if __name__ == "__main__":
    main()