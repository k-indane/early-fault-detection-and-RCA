from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
from tefault.config import MonitorConfig
from tefault.monitor import FaultMonitor
from tefault.data import select_runs
from tefault.online_engine import OnlineMonitorEngine

@dataclass
class Session:
    session_id: str
    run_uid: str
    tick_interval_s: float = 0.0

class SessionManager:
    """
    Holds cached data/artifacts and creates OnlineMonitorEngine streams.
    """

    def __init__(self, cfg: MonitorConfig, project_root: str = "."):
        self.cfg = cfg
        self.project_root = project_root

        # Load artifacts + data once
        from pathlib import Path
        self.monitor = FaultMonitor(cfg, project_root=Path(project_root))
        self.monitor.load_artifacts()
        ff, f = self.monitor.load_data()
        self.data = select_runs(ff, f, cfg)

        if self.data.empty:
            raise RuntimeError("No runs available after select_runs(). Check config filters and data paths.")

        # Build SHAP background once
        self.xgb_bg = None
        if cfg.xgb_enable and self.monitor.xgb_scorer is not None:
            self.xgb_bg = self.monitor.xgb_scorer.build_background(self.data, cfg)

        self.sessions: Dict[str, Session] = {}

    # Returns run table
    def list_runs(self) -> pd.DataFrame:
        return (
            self.data[["run_uid", "true_fault", "simulationRun"]]
            .drop_duplicates()
            .sort_values(["true_fault", "simulationRun"])
            .reset_index(drop=True)
        )
    
    # Validates run_uid and creates session
    def create_session(self, run_uid: str, tick_interval_s: float = 0.0) -> Session:
        if run_uid not in set(self.data["run_uid"].unique()):
            raise ValueError(f"Unknown run_uid={run_uid!r}")

        sid = uuid.uuid4().hex
        sess = Session(session_id=sid, run_uid=run_uid, tick_interval_s=float(tick_interval_s))
        self.sessions[sid] = sess
        return sess

    # Returns session by ID
    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    # Builds OnlineMonitorEngine for session
    def build_engine_for_session(self, session_id: str) -> OnlineMonitorEngine:
        sess = self.get_session(session_id)
        if sess is None:
            raise ValueError(f"Unknown session_id={session_id!r}")

        engine = OnlineMonitorEngine(
            cfg=self.cfg,
            xgb_scorer=self.monitor.xgb_scorer,
            ae_scorer=self.monitor.ae_scorer,
            xgb_background=self.xgb_bg,
            enable_rca=True,
        )
        return engine
    
    # Returns run dataframe for session
    def get_run_df(self, run_uid: str) -> pd.DataFrame:
        df_run = self.data[self.data["run_uid"] == run_uid].sort_values("sample").copy()
        if df_run.empty:
            raise ValueError(f"Run dataframe empty for run_uid={run_uid!r}")
        return df_run
