"""State detection and per-state network estimation."""

from itertools import permutations
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def _windowed_features(X, window=5):
    """Rolling mean and SD over the last `window` points (trailing, no look-ahead)."""
    df = pd.DataFrame(X)
    roll_mean = df.rolling(window, min_periods=window).mean()
    roll_std = df.rolling(window, min_periods=window).std()
    features = np.hstack([roll_mean.values, roll_std.values])
    # first window-1 rows are NaN, drop them
    return features[window - 1:]


def _fit_var1(X_prev, X_curr, alphas):
    """VAR(1) coefficients, B[i, j] = effect of symptom j at t-1 on symptom i at t."""
    p = X_prev.shape[1]
    B = np.zeros((p, p))
    # one ridge per symptom so each gets its own alpha
    for j in range(p):
        ridge = RidgeCV(alphas=alphas).fit(X_prev, X_curr[:, j])
        B[j, :] = ridge.coef_
    return B


def detect_states_gmm(
    X: np.ndarray,
    n_states: int = 2,
    window: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Cluster windowed mean/SD features with a GMM.

    The first window-1 points have no full window, so they just get the
    label of the first clustered point.
    """
    X = np.asarray(X)
    features = _windowed_features(X, window=window)
    features = StandardScaler().fit_transform(features)

    # n_init > 1 because GMM is sensitive to where it starts
    gmm = GaussianMixture(
        n_components=n_states,
        random_state=seed,
        n_init=5,
    )
    states_trimmed = gmm.fit_predict(features)

    states = np.concatenate([
        np.full(window - 1, states_trimmed[0], dtype=int),
        states_trimmed,
    ])
    return states


def detect_states_hmm(
    X: np.ndarray,
    n_states: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """Gaussian HMM (full covariance) on the raw series."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError("hmmlearn not installed") from e

    X = np.asarray(X)
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        random_state=seed,
        n_iter=100,
    )
    hmm.fit(X)
    return hmm.predict(X)


def agreement(a: np.ndarray, b: np.ndarray, n_states: int) -> float:
    """Fraction of matching labels, maximised over relabelings of b."""
    a = np.asarray(a)
    b = np.asarray(b)
    # state labels are arbitrary, so try every permutation (fine for small n_states)
    best = 0.0
    for perm in permutations(range(n_states)):
        remapped = np.array([perm[x] for x in b])
        best = max(best, (a == remapped).mean())
    return best


def state_persistence_summary(
    states: np.ndarray,
    n_states: Optional[int] = None,
) -> pd.DataFrame:
    """Time in each state, mean dwell time, and P(stay) per state.

    A run still going at the end of the series is counted at its cut-off
    length, so dwell times are a bit underestimated.
    """
    states = np.asarray(states)
    if n_states is None:
        n_states = int(states.max()) + 1

    rows = []
    for s in range(n_states):
        in_state = states == s
        n_visits = int(in_state.sum())

        # lengths of consecutive runs in state s
        runs = []
        run_len = 0
        for v in in_state:
            if v:
                run_len += 1
            elif run_len > 0:
                runs.append(run_len)
                run_len = 0
        if run_len > 0:
            runs.append(run_len)

        # how often we're still in s at the next step
        stayed = total = 0
        for t in range(len(states) - 1):
            if states[t] == s:
                total += 1
                if states[t + 1] == s:
                    stayed += 1

        mean_dwell = float(np.mean(runs)) if runs else float("nan")

        rows.append({
            "state": s,
            "n_timepoints": n_visits,
            "pct_of_series": round(100 * n_visits / len(states), 1),
            "mean_dwell_time": round(mean_dwell, 2),
            "self_transition_prob": round(stayed / total, 3) if total else float("nan"),
        })

    return pd.DataFrame(rows)


def fit_state_networks(
    X: np.ndarray,
    states: np.ndarray,
    symptom_names: Optional[List[str]] = None,
    alphas: Optional[List[float]] = None,
    min_obs: int = 15,
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    """Fit a ridge VAR(1) separately within each state.

    X is (T, p), states is (T,). States with fewer than `min_obs`
    usable transitions are skipped. Returns {state: B} with B of shape
    (p, p), plus the symptom names.
    """
    X = np.asarray(X, dtype=float)
    states = np.asarray(states)
    n_symptoms = X.shape[1]

    if symptom_names is None:
        symptom_names = [f"symptom_{i + 1}" for i in range(n_symptoms)]
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    networks = {}
    for s in np.unique(states):
        # a transition is (t-1 -> t) with t in state s, so t = 0 has no predecessor
        idx_t = np.where(states == s)[0]
        idx_t = idx_t[idx_t > 0]

        if len(idx_t) < min_obs:
            continue

        networks[int(s)] = _fit_var1(X[idx_t - 1], X[idx_t], alphas)

    return networks, symptom_names


def network_to_graph(
    B: np.ndarray,
    symptom_names: List[str],
    edge_threshold: float = 0.05,
    include_self_loops: bool = False,
) -> nx.DiGraph:
    """Turn a coefficient matrix into a directed graph.

    Edge j -> i means symptom j at t-1 predicts symptom i at t. Note the
    threshold acts on ridge-shrunken coefficients, so the edge count
    depends on which alphas were selected.
    """
    G = nx.DiGraph()
    G.add_nodes_from(symptom_names)

    n = len(symptom_names)
    for i in range(n):
        for j in range(n):
            if (include_self_loops or i != j) and abs(B[i, j]) >= edge_threshold:
                G.add_edge(symptom_names[j], symptom_names[i], weight=B[i, j])

    return G


def network_density_summary(
    networks: Dict[int, np.ndarray],
    symptom_names: List[str],
    edge_threshold: float = 0.05,
) -> pd.DataFrame:
    """Edge count and coupling strength per state."""
    rows = []
    for s, B in networks.items():
        G = network_to_graph(B, symptom_names, edge_threshold)
        weights = [abs(d["weight"]) for _, _, d in G.edges(data=True)]

        rows.append({
            "state": s,
            "n_edges": G.number_of_edges(),
            "mean_abs_coupling": round(np.mean(weights), 3) if weights else 0.0,
            "max_abs_coupling": round(np.max(weights), 3) if weights else 0.0,
        })

    return pd.DataFrame(rows).sort_values("state").reset_index(drop=True)


def bootstrap_edge_count(
    X: np.ndarray,
    states: np.ndarray,
    state: int,
    n_boot: int = 200,
    alphas: Optional[List[float]] = None,
    edge_threshold: float = 0.05,
    min_obs: int = 15,
    ci: float = 95.0,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap the off-diagonal edge count for one state.

    Returns (mean, ci_low, ci_high). Divide by p * (p - 1) if you want a
    density. Caveat: this resamples (t-1, t) pairs as if they were
    independent, which ignores autocorrelation, so the interval is
    probably too narrow.
    """
    X = np.asarray(X, dtype=float)
    states = np.asarray(states)
    n_symptoms = X.shape[1]

    if alphas is None:
        alphas = DEFAULT_ALPHAS

    idx = np.where(states == state)[0]
    idx = idx[idx > 0]
    if len(idx) < min_obs:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    off_diag = ~np.eye(n_symptoms, dtype=bool)  # leave the self-loops out

    edge_counts = []
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        B = _fit_var1(X[sample - 1], X[sample], alphas)
        edge_counts.append(int(np.sum(np.abs(B[off_diag]) >= edge_threshold)))

    tail = (100 - ci) / 2
    low, high = np.percentile(edge_counts, [tail, 100 - tail])
    return float(np.mean(edge_counts)), float(low), float(high)