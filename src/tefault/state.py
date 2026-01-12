from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tefault.config import MonitorConfig
from tefault.features import apply_consecutive_rule

# Turns monitoring logic into decision system

@dataclass(frozen=True)
class AlertInfo:
    first_alert_sample: float  # keep float to allow np.nan
    detection_delay_samples: float
    alert_type_first: str

def build_state_timeline(merged: pd.DataFrame, cfg: MonitorConfig) -> pd.DataFrame:
    """
    Given a merged timeline (XGB + AE merged on sample), compute:
      - per-model raw flags
      - per-model stable flags (k consecutive)
      - arbitration (KNOWN takes priority)
      - system flags
      - state (NORMAL/SUSPECT/ALERT)

    Expects columns if present:
      - xgb_pred_fault, xgb_pred_conf
      - ae_flag_raw, ae_recon_error

    Returns a copy of merged with state columns added.
    """
    df = merged.copy()

    # Ensure columns exist + fill NaNs for comparisons
    if "xgb_pred_fault" not in df.columns:
        df["xgb_pred_fault"] = 0
    if "xgb_pred_conf" not in df.columns:
        df["xgb_pred_conf"] = 0.0
    if "ae_flag_raw" not in df.columns:
        df["ae_flag_raw"] = 0
    if "ae_recon_error" not in df.columns:
        df["ae_recon_error"] = np.nan

    df["xgb_pred_fault"] = df["xgb_pred_fault"].fillna(0).astype(int)
    df["xgb_pred_conf"] = df["xgb_pred_conf"].fillna(0.0).astype(float)
    df["ae_flag_raw"] = df["ae_flag_raw"].fillna(0).astype(int)

    # Per-model raw flags
    df["xgb_raw_flag"] = (
        (df["xgb_pred_fault"] != 0) & (df["xgb_pred_conf"] >= float(cfg.xgb_conf_threshold))
    ).astype(int)

    df["ae_raw_flag"] = df["ae_flag_raw"].astype(int)

    # Per-model stable flags (different k)
    df["xgb_stable_flag"] = apply_consecutive_rule(
        df["xgb_raw_flag"].values.astype(int),
        int(cfg.xgb_consecutive_required),
    )

    df["ae_stable_flag"] = apply_consecutive_rule(
        df["ae_raw_flag"].values.astype(int),
        int(cfg.ae_consecutive_required),
    )

    # Arbitration (KNOWN takes priority)
    known_fault = df["xgb_stable_flag"].astype(int) == 1
    anomaly_unclassified = (df["ae_stable_flag"].astype(int) == 1) & (~known_fault)

    df["system_stable_flag"] = (known_fault | anomaly_unclassified).astype(int)
    df["alert_type"] = np.where(
        known_fault,
        "KNOWN_FAULT",
        np.where(anomaly_unclassified, "ANOMALY_UNCLASSIFIED", "NONE"),
    )

    # SUSPECT if any raw fired but stable didn't
    df["system_raw_flag"] = ((df["xgb_raw_flag"] == 1) | (df["ae_raw_flag"] == 1)).astype(int)

    df["state"] = np.where(
        df["system_stable_flag"] == 1,
        "ALERT",
        np.where(df["system_raw_flag"] == 1, "SUSPECT", "NORMAL"),
    )

    return df


def first_alert_info(timeline: pd.DataFrame, cfg: MonitorConfig) -> AlertInfo:
    """
    Compute first alert sample + detection delay + alert type from a timeline
    that already includes system_stable_flag and alert_type.

    Returns AlertInfo with np.nan values if no alert.
    """
    if "system_stable_flag" not in timeline.columns or "alert_type" not in timeline.columns:
        raise ValueError("Timeline must include 'system_stable_flag' and 'alert_type' columns.")
    
    # Find first ALERT
    alert_rows = timeline[timeline["system_stable_flag"] == 1]
    if not alert_rows.empty:
        first_alert_sample = float(alert_rows["sample"].iloc[0])
        detection_delay = float(first_alert_sample - int(cfg.fault_intro_sample))
        alert_type_first = str(alert_rows["alert_type"].iloc[0])
    
    # No ALERTs
    else:
        first_alert_sample = float(np.nan)
        detection_delay = float(np.nan)
        alert_type_first = "NONE"
        
    # Return results
    return AlertInfo(
        first_alert_sample=first_alert_sample,
        detection_delay_samples=detection_delay,
        alert_type_first=alert_type_first,
    )
