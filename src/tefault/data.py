from __future__ import annotations
from typing import Set
import pandas as pd
from tefault.config import MonitorConfig

def select_runs(ff: pd.DataFrame, f: pd.DataFrame, cfg: MonitorConfig) -> pd.DataFrame:
    """
    Note:
      - faulty data uses faultNumber in {1..20}
      - simulationRun in {1..500} is a *seed index* reused across all faults
      - therefore a unique run must be identified by (faultNumber, simulationRun)

    We add a unique run_uid:
      - fault-free:  FF_R<simulationRun>
      - faulty:      F<faultNumber>_R<simulationRun>

    Returns a stacked dataframe with:
      - true_fault (0 for fault-free, faultNumber for faulty)
      - is_faulty_run (0/1)
      - run_uid (unique identifier)
    """

    ff = ff.copy()
    f = f.copy()

    # Ensure required cols exist
    if "faultNumber" not in ff.columns:
        ff["faultNumber"] = 0
    if "faultNumber" not in f.columns:
        raise ValueError("Faulty dataframe must contain faultNumber.")
    if "simulationRun" not in ff.columns or "simulationRun" not in f.columns:
        raise ValueError("Both dataframes must contain simulationRun.")
    if "sample" not in ff.columns or "sample" not in f.columns:
        raise ValueError("Both dataframes must contain sample.")

    # Normalize types
    ff["faultNumber"] = pd.to_numeric(ff["faultNumber"], errors="coerce").fillna(0).astype(int)
    f["faultNumber"] = pd.to_numeric(f["faultNumber"], errors="coerce").astype(int)
    ff["simulationRun"] = pd.to_numeric(ff["simulationRun"], errors="coerce").astype(int)
    f["simulationRun"] = pd.to_numeric(f["simulationRun"], errors="coerce").astype(int)
    ff["sample"] = pd.to_numeric(ff["sample"], errors="coerce").astype(int)
    f["sample"] = pd.to_numeric(f["sample"], errors="coerce").astype(int)

    # Apply run_mode filter
    if cfg.run_mode == "unknown_faults_only":
        f = f[f["faultNumber"].isin(tuple(cfg.unknown_faults))].copy()

    # Excluded faults from this study (3, 9, 15)
    excl: Set[int] = set(int(x) for x in cfg.exclude_faults) if cfg.exclude_faults else set()
    if excl:
        f = f[~f["faultNumber"].isin(excl)].copy()

    # Add labels
    ff["true_fault"] = 0
    f["true_fault"] = f["faultNumber"].astype(int)

    ff["is_faulty_run"] = 0
    f["is_faulty_run"] = 1

    # Add unique run identifier
    ff["run_uid"] = ff["simulationRun"].apply(lambda r: f"FF_R{int(r)}")
    f["run_uid"] = f.apply(lambda r: f"F{int(r['faultNumber'])}_R{int(r['simulationRun'])}", axis=1)

    return pd.concat([ff, f], ignore_index=True)
