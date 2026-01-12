from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd

def apply_consecutive_rule(binary_preds: np.ndarray, k: int) -> np.ndarray:
    """
    Convert predictions to stable predictions that require k consecutive 1s.
    Marks the whole streak as 1 once streak length >= k.
    """
    out = np.zeros_like(binary_preds)
    streak = 0
    for i, p in enumerate(binary_preds):
        if int(p) == 1:
            streak += 1
        else:
            streak = 0
        if streak >= k:
            out[i - k + 1 : i + 1] = 1
    return out


def infer_base_features_from_predictors(predictors: List[str]) -> List[str]:
    """
    Predictors look like: XMEAS(1)_lag1, XMEAS(1)_ewma, etc.
    Infers the base feature set by stripping suffixes.
    """
    base: List[str] = []
    for p in predictors:
        if p.endswith("_ewma"):
            base.append(p[: -len("_ewma")])
        elif "_lag" in p:
            idx = p.rfind("_lag")
            base.append(p[:idx])
        else:
            base.append(p)
    return sorted(list(set(base)))

# Adds lagged features
def add_lags(df: pd.DataFrame, base_features: List[str], lags: List[int], group_col: str) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        lagged = df.groupby(group_col)[base_features].shift(lag)
        lagged.columns = [f"{c}_lag{lag}" for c in base_features]
        df = pd.concat([df, lagged], axis=1)
    return df

# Adds EWMA features
def add_ewma(df: pd.DataFrame, base_features: List[str], span: int, group_col: str) -> pd.DataFrame:
    df = df.copy()
    for c in base_features:
        df[f"{c}_ewma"] = df.groupby(group_col)[c].transform(lambda x: x.ewm(span=span, adjust=False).mean())
    return df


def make_ae_windows_for_run(
    df_run: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    min_sample: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build flattened rolling windows for a single run.
    Returns:
      X_win: (n_windows, window_size * n_features) Each row is a flattened window
      win_end_samples: (n_windows,) For each window, the sample index corresponding to the last row in the window
    """
    df_run = df_run.sort_values("sample").copy()
    df_run = df_run[df_run["sample"] >= min_sample]
    X = df_run[feature_cols].values
    
    samples = df_run["sample"].values.astype(int)
    if len(X) < window_size:
        return np.empty((0, window_size * len(feature_cols))), np.empty((0,), dtype=int)

    seqs, end_samples = [], []
    for i in range(window_size - 1, len(X)):
        seqs.append(X[i - window_size + 1 : i + 1].reshape(-1))
        end_samples.append(samples[i])

    return np.asarray(seqs), np.asarray(end_samples, dtype=int)
