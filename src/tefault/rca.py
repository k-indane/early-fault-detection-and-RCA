from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Helpers
def _base_feature_name(feat: str) -> str:
    """
    Map engineered feature name back to base name.

    Examples:
      - XMEAS(1)_lag1 -> XMEAS(1)
      - XMEAS(1)_ewma -> XMEAS(1)
      - XMV(3) -> XMV(3)
    """
    if feat.endswith("_ewma"):
        return feat[: -len("_ewma")]
    if "_lag" in feat:
        idx = feat.rfind("_lag")
        return feat[:idx]
    return feat


def _aggregate_contributions(
    feature_names: List[str],
    contribs: np.ndarray,
    aggregate_mode: str = "abs_sum",
) -> Dict[str, float]:
    """
    Aggregate engineered-feature contributions back to base features.

    aggregate_mode:
      - "abs_sum": sum(abs(contrib)) per base feature (default)
    """
    agg: Dict[str, float] = {}
    for f, c in zip(feature_names, contribs):
        base = _base_feature_name(str(f))
        c = float(c)

        if aggregate_mode == "abs_sum":
            agg[base] = agg.get(base, 0.0) + abs(c)
        else:
            raise ValueError(f"Unknown aggregate_mode={aggregate_mode}. Use abs_sum")

    return agg


# XGB RCA (SHAP)
def xgb_rca_shap(
    model: Any,
    X_row: pd.DataFrame,
    feature_names: List[str],
    top_k: int = 10,
    background: Optional[pd.DataFrame] = None,
    aggregate_mode: str = "abs_sum",
) -> List[Dict[str, float]]:
    """
    Compute SHAP feature contributions for a single sample and return top drivers
    aggregated to base variables.

    Returns:
      [{"feature": "<BASE_FEATURE>", "contribution": <float>}, ...]

    """
    try:
        import shap
    except Exception as e:
        raise ImportError(
            "shap is required for xgb_rca_shap but is not installed. "
            "Install with: pip install shap"
        ) from e

    if not isinstance(X_row, pd.DataFrame) or len(X_row) != 1:
        raise ValueError("X_row must be a pandas DataFrame with exactly one row.")

    # Ensure correct column order (same as training/inference)
    X_row = X_row.copy()
    X_row = X_row[feature_names]

    # Choose predicted fault class index for multiclass
    class_idx = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_row)
        if isinstance(proba, list):
            proba = np.asarray(proba)
        if np.ndim(proba) == 2 and proba.shape[0] == 1 and proba.shape[1] > 1:
            class_idx = int(np.argmax(proba[0]))

    # Build explainer
    # Prefer TreeExplainer; fallback to Explainer
    explainer = None
    shap_values = None

    # Background helps stability
    bg = background
    if bg is not None:
        bg = bg.copy()
        bg = bg[feature_names]

    try:
        # TreeExplainer
        explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional") if bg is not None else shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
    except Exception:
        # Generic explainer
        if hasattr(model, "predict_proba") and bg is not None:
            if class_idx is None:
                # Default to max class
                proba = model.predict_proba(X_row)
                class_idx = int(np.argmax(proba[0])) if np.ndim(proba) == 2 else None

            # Returns predicted probability for explained class
            def f(X):
                p = model.predict_proba(X)
                if class_idx is None:
                    # Binary: use positive class prob if available
                    return p[:, -1]
                return p[:, class_idx]

            explainer = shap.Explainer(f, bg)
            shap_values = explainer(X_row).values
        else:
            raise RuntimeError(
                "Failed to create SHAP explainer. Provide a background dataframe and ensure model supports predict_proba."
            )

    # Normalize SHAP output to 1D contributions aligned with feature names
    contrib = None
    if isinstance(shap_values, list):
        # Selects the right class list entry
        if class_idx is None:
            class_idx = 0
        contrib = np.asarray(shap_values[class_idx])[0]
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (n_samples, n_classes, n_features)
            if class_idx is None:
                class_idx = 0
            contrib = arr[0, class_idx, :]
        elif arr.ndim == 2:
            # (n_samples, n_features)
            contrib = arr[0, :]
        elif arr.ndim == 1:
            contrib = arr
        else:
            raise RuntimeError(f"Unexpected SHAP values shape: {arr.shape}")

    contrib = np.asarray(contrib, dtype=float).reshape(-1)

    # Aggregate engineered features to base feature names
    agg = _aggregate_contributions(feature_names, contrib, aggregate_mode=aggregate_mode)

    # Sort and return top_k
    ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[: int(top_k)]
    return [{"feature": k, "contribution": float(v)} for k, v in ranked]


# AE RCA (partial residual-based)
def ae_rca_residuals(
    ae_model: Any,
    scaler: Any,
    df_run: pd.DataFrame, # For one run
    feature_cols: List[str],
    window_size: int,
    alert_sample: int,
    top_k: int = 10,
    residual_mode: str = "mean_abs",
) -> List[Dict[str, float]]:
    """
    Explain AE anomaly by ranking features with largest reconstruction residuals.

    residual_mode:
      - "mean_abs": mean(abs(residual)) across the window (default)

    Returns:
      [{"feature": "<FEATURE>", "contribution": <float>}, ...]
    """
    if df_run.empty:
        return [{"feature": "EMPTY_RUN", "contribution": float("nan")}]

    # Restrict to samples up to alert
    df_run = df_run.sort_values("sample").copy()
    df_w = df_run[df_run["sample"] <= int(alert_sample)].copy()

    # Ensure sufficient window (10)
    if len(df_w) < int(window_size):
        return [{"feature": "INSUFFICIENT_WINDOW", "contribution": float("nan")}]

    # Check all feature columns present
    missing = [c for c in feature_cols if c not in df_w.columns]
    if missing:
        return [{"feature": f"MISSING_COL_{missing[0]}", "contribution": float("nan")}]

    # Extract window that caused ALERT
    df_win = df_w.tail(int(window_size)).copy()

    # Scale features
    X = df_win[feature_cols].astype(float).values  # (window_size, n_features)
    Xs = scaler.transform(X)                       # Scaled (window_size, n_features)

    # Flatten window for AE input + run reconstriction
    x_flat = Xs.reshape(1, -1)
    recon_flat = ae_model.predict(x_flat, verbose=0)
    recon = np.asarray(recon_flat).reshape(int(window_size), len(feature_cols))

    # Compute residuals
    residual = Xs - recon

    # Reduce to one score per feature
    if residual_mode == "mean_abs":
        scores = np.mean(np.abs(residual), axis=0)
    else:
        raise ValueError("residual_mode must be mean_abs")

    # Rank features by residual score
    scores = np.asarray(scores, dtype=float)
    idx = np.argsort(scores)[::-1][: int(top_k)]

    return [{"feature": str(feature_cols[i]), "contribution": float(scores[i])} for i in idx]
