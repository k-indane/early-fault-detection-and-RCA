from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterator, List, Optional
import numpy as np
import pandas as pd
from collections import deque
from tefault.config import MonitorConfig
from tefault.scorers import XGBScorer, AEScorer
from tefault.rca import xgb_rca_shap, ae_rca_residuals

# Reuse monitoring logic in live engine, where data arrives one sample at a time.

# Event types
Event = Dict[str, Any]

@dataclass
class OnlineEngineState:
    run_uid: str
    simulationRun: int
    true_fault: int
    is_faulty_run: int

    # Clock
    current_sample: int = -1

    # XGB running values
    xgb_last_pred_fault: int = 0
    xgb_last_conf: float = 0.0
    xgb_last_raw: int = 0
    xgb_streak: int = 0
    xgb_stable: int = 0

    # AE running values
    ae_last_recon_error: float = float("nan")
    ae_last_raw: int = 0
    ae_streak: int = 0
    ae_stable: int = 0

    # System
    alert_type: str = "NONE"  # NONE | KNOWN_FAULT | ANOMALY_UNCLASSIFIED
    state: str = "NORMAL"     # NORMAL | SUSPECT | ALERT

    # Latch
    alert_fired: bool = False
    rca_emitted: bool = False


class OnlineMonitorEngine:
    """
    Live monitoring engine for a single run.

    - Consumes samples/rows sequentially from a run dataframe.
    - Performs inference every cfg.eval_every_n_samples.
    - Maintains streak counters for stable flags.
    - Emits events (tick/alert/rca/done) suitable for WebSocket streaming.
    """

    def __init__(
        self,
        cfg: MonitorConfig,
        xgb_scorer: Optional[XGBScorer] = None,
        ae_scorer: Optional[AEScorer] = None,
        xgb_background: Optional[pd.DataFrame] = None,
        enable_rca: bool = True,
        rca_top_k: int = 10,
    ):
        self.cfg = cfg
        self.xgb = xgb_scorer
        self.ae = ae_scorer
        self.xgb_background = xgb_background
        self.enable_rca = enable_rca
        self.rca_top_k = int(rca_top_k)

        # Buffers for online feature building
        self._xgb_lag_buffer: Optional[Deque[pd.Series]] = None
        self._xgb_ewma_state: Optional[Dict[str, float]] = None

        self._ae_window_buffer: Optional[Deque[np.ndarray]] = None

        # Keep raw run rows for RCA so we can rebuild XGB row or AE window at alert time
        self._df_run_full: Optional[pd.DataFrame] = None

        self.state: Optional[OnlineEngineState] = None

    # Initialization
    def start(self, df_run: pd.DataFrame) -> Event:
        """
        Initialize engine for a run and return a 'start' event.
        """
        if df_run.empty:
            raise ValueError("df_run is empty.")

        df_run = df_run.sort_values("sample").reset_index(drop=True)

        # Required metadata
        run_uid = str(df_run["run_uid"].iloc[0]) if "run_uid" in df_run.columns else "UNKNOWN_RUN"
        sim_seed = int(df_run["simulationRun"].iloc[0]) if "simulationRun" in df_run.columns else -1
        true_fault = int(df_run["true_fault"].iloc[0]) if "true_fault" in df_run.columns else 0
        is_faulty = int(df_run["is_faulty_run"].iloc[0]) if "is_faulty_run" in df_run.columns else int(true_fault != 0)

        self._df_run_full = df_run  # Store for RCA
        self.state = OnlineEngineState(
            run_uid=run_uid,
            simulationRun=sim_seed,
            true_fault=true_fault,
            is_faulty_run=is_faulty,
        )

        # XGB online buffers
        if self.cfg.xgb_enable and self.xgb is not None:
            max_lag = max(self.xgb.lags) if self.xgb.lags else 0
            self._xgb_lag_buffer = deque(maxlen=max_lag)  # store last raw rows
            self._xgb_ewma_state = {feat: np.nan for feat in self.xgb.base_features}

        # AE online buffers
        if self.cfg.ae_enable and self.ae is not None:
            self._ae_window_buffer = deque(maxlen=self.ae.window_size)

        return {
            "type": "start",
            "run_uid": run_uid,
            "simulationRun": sim_seed,
            "true_fault": true_fault,
            "is_faulty_run": is_faulty,
            "eval_every_n_samples": int(self.cfg.eval_every_n_samples),
        }

    # Streaming loop
    def stream(self, df_run: pd.DataFrame, speed_hz: Optional[float] = None) -> Iterator[Event]:
        """
        Yield events as we iterate through df_run.
        """
        yield self.start(df_run)

        assert self.state is not None
        assert self._df_run_full is not None

        for _, row in self._df_run_full.iterrows():
            sample = int(row["sample"])
            events = self.step(row, sample=sample)
            for e in events:
                yield e

        yield {
            "type": "done",
            "run_uid": self.state.run_uid,
            "simulationRun": self.state.simulationRun,
            "true_fault": self.state.true_fault,
        }

    # Single step
    def step(self, row: pd.Series, sample: int) -> List[Event]:
        """
        Consume one sample row and return a list of emitted events.
        """
        if self.state is None:
            raise RuntimeError("Engine not started. Call start(df_run) first.")

        s = self.state
        s.current_sample = int(sample)

        # Determine if this sample is an evaluation tick
        evaluated = (int(sample) % int(self.cfg.eval_every_n_samples) == 0)

        # Update XGB buffers always (raw values + EWMA state)
        if self.cfg.xgb_enable and self.xgb is not None:
            self._update_xgb_buffers(row)

        # Update AE window buffer always (scaled values)
        if self.cfg.ae_enable and self.ae is not None:
            self._update_ae_buffer(row, true_fault=s.true_fault)

        # Only run inference on evaluation ticks
        if evaluated:
            if self.cfg.xgb_enable and self.xgb is not None:
                self._eval_xgb_at_tick(row)

            if self.cfg.ae_enable and self.ae is not None:
                self._eval_ae_at_tick(sample)

            # Update system state machine based on latest raw flags
            self._update_state_machine()

        # Emit tick every sample
        events: List[Event] = [self._make_tick_event(evaluated=evaluated)]

        # If an alert just fired, emit alert + RCA
        if (s.state == "ALERT") and (not s.alert_fired):
            s.alert_fired = True
            events.append(self._make_alert_event())

            if self.enable_rca and (not s.rca_emitted):
                rca_event = self._compute_rca_if_possible()
                if rca_event is not None:
                    s.rca_emitted = True
                    events.append(rca_event)

        return events

    # XGB online feature updates
    def _update_xgb_buffers(self, row: pd.Series) -> None:
        assert self.xgb is not None
        assert self._xgb_ewma_state is not None

        # Update EWMA state incrementally for base features
        span = float(self.xgb.ewma_span)
        alpha = 2.0 / (span + 1.0)

        for feat in self.xgb.base_features:
            x = float(row[feat])
            prev = self._xgb_ewma_state.get(feat, np.nan)
            if np.isnan(prev):
                self._xgb_ewma_state[feat] = x
            else:
                self._xgb_ewma_state[feat] = alpha * x + (1.0 - alpha) * prev

        # Store raw row for lag access
        if self._xgb_lag_buffer is not None:
            self._xgb_lag_buffer.append(row)

    def _eval_xgb_at_tick(self, row: pd.Series) -> None:
        """
        Build one-row predictor frame and run XGB inference.
        """
        assert self.xgb is not None
        assert self.state is not None
        assert self._xgb_ewma_state is not None

        # Build predictor dict
        pred: Dict[str, float] = {}

        # Base features
        for feat in self.xgb.base_features:
            pred[feat] = float(row[feat])

        # Lag features
        if self._xgb_lag_buffer is not None:
            buf = list(self._xgb_lag_buffer)
        else:
            buf = []

        for lag in self.xgb.lags:
            # Need value at t-lag so buffer holds previous rows and newest at end
            idx = -lag
            if len(buf) >= lag:
                lag_row = buf[idx]
                for feat in self.xgb.base_features:
                    pred[f"{feat}_lag{lag}"] = float(lag_row[feat])
            else:
                for feat in self.xgb.base_features:
                    pred[f"{feat}_lag{lag}"] = np.nan

        # EWMA features
        for feat in self.xgb.base_features:
            pred[f"{feat}_ewma"] = float(self._xgb_ewma_state[feat])

        # Construct X in predictor order
        X_row = pd.DataFrame([{k: pred.get(k, np.nan) for k in self.xgb.predictors}])

        # If any predictor is NaN, can't evaluate yet
        if X_row.isna().any(axis=1).iloc[0]:
            self.state.xgb_last_pred_fault = 0
            self.state.xgb_last_conf = 0.0
            self.state.xgb_last_raw = 0
            return
        
        # Run inference
        y_pred_enc = self.xgb.model.predict(X_row)
        y_pred_fault = int(self.xgb.label_encoder.inverse_transform(y_pred_enc)[0])

        # Confidence
        if hasattr(self.xgb.model, "predict_proba"):
            proba = self.xgb.model.predict_proba(X_row)
            conf = float(np.max(proba, axis=1)[0])
        else:
            conf = float("nan")

        # Update state
        self.state.xgb_last_pred_fault = y_pred_fault
        self.state.xgb_last_conf = conf
        self.state.xgb_last_raw = int((y_pred_fault != 0) and (conf >= float(self.cfg.xgb_conf_threshold)))

    # AE online feature updates
    def _update_ae_buffer(self, row: pd.Series, true_fault: int) -> None:
        """
        Update AE window buffer with the scaled feature vector.
        For faulty runs, we start filling only at fault_intro_sample.
        """
        assert self.ae is not None
        assert self._ae_window_buffer is not None
        assert self.state is not None

        # Min_sample = 0 if true_fault==0 else fault_intro_sample
        if true_fault != 0 and int(row["sample"]) < int(self.cfg.fault_intro_sample):
            return

        # Scale one row features
        vec_df = pd.DataFrame(
        [[float(row[c]) for c in self.ae.feature_cols]],
        columns=self.ae.feature_cols,
        )
        vec_scaled = self.ae.scaler.transform(vec_df).reshape(-1)

        self._ae_window_buffer.append(vec_scaled)

    def _eval_ae_at_tick(self, sample: int) -> None:
        """
        Run AE inference at tick if buffer is full.
        """
        assert self.ae is not None
        assert self.state is not None
        assert self._ae_window_buffer is not None

        if len(self._ae_window_buffer) < int(self.ae.window_size):
            self.state.ae_last_recon_error = float("nan")
            self.state.ae_last_raw = 0
            return
        
        # Build window array
        X_win = np.asarray(list(self._ae_window_buffer), dtype=float).reshape(1, -1)
        recon = self.ae.model.predict(X_win, verbose=0)
        recon_error = float(np.mean(np.square(X_win - recon), axis=1)[0])

        # Update state
        self.state.ae_last_recon_error = recon_error
        self.state.ae_last_raw = int(recon_error > float(self.ae.threshold))

    # State machine update
    def _update_state_machine(self) -> None:
        assert self.state is not None
        s = self.state

        # Update streaks
        if int(s.xgb_last_raw) == 1:
            s.xgb_streak += 1
        else:
            s.xgb_streak = 0

        if int(s.ae_last_raw) == 1:
            s.ae_streak += 1
        else:
            s.ae_streak = 0

        # Update stable flags
        s.xgb_stable = int(s.xgb_streak >= int(self.cfg.xgb_consecutive_required))
        s.ae_stable = int(s.ae_streak >= int(self.cfg.ae_consecutive_required))

        # Update arbitration
        known_fault = (s.xgb_stable == 1)
        anomaly_unclassified = (s.ae_stable == 1) and (not known_fault)

        if known_fault:
            s.alert_type = "KNOWN_FAULT"
        elif anomaly_unclassified:
            s.alert_type = "ANOMALY_UNCLASSIFIED"
        else:
            s.alert_type = "NONE"

        # Determine system state
        system_stable = int(known_fault or anomaly_unclassified)
        system_raw = int((s.xgb_last_raw == 1) or (s.ae_last_raw == 1))

        if system_stable == 1:
            s.state = "ALERT"
        elif system_raw == 1:
            s.state = "SUSPECT"
        else:
            s.state = "NORMAL"

    # Event builders
    def _make_tick_event(self, evaluated: bool) -> Event:
        assert self.state is not None
        s = self.state
        return {
            "type": "tick",
            "run_uid": s.run_uid,
            "simulationRun": s.simulationRun,
            "true_fault": s.true_fault,
            "sample": int(s.current_sample),
            "state": s.state,
            "alert_type": s.alert_type,
            "evaluated": bool(evaluated),
            "xgb": {
                "pred_fault": int(s.xgb_last_pred_fault),
                "conf": float(s.xgb_last_conf) if s.xgb_last_conf is not None else None,
                "raw": int(s.xgb_last_raw),
                "stable": int(s.xgb_stable),
            },
            "ae": {
                "recon_error": float(s.ae_last_recon_error) if not np.isnan(s.ae_last_recon_error) else None,
                "raw": int(s.ae_last_raw),
                "stable": int(s.ae_stable),
            },
        }

    def _make_alert_event(self) -> Event:
        assert self.state is not None
        s = self.state
        return {
            "type": "alert",
            "run_uid": s.run_uid,
            "simulationRun": s.simulationRun,
            "true_fault": s.true_fault,
            "sample": int(s.current_sample),
            "state": s.state,
            "alert_type": s.alert_type,
            "pred_fault": (int(s.xgb_last_pred_fault) if s.alert_type == "KNOWN_FAULT" else None),
        }

    # RCA
    def _compute_rca_if_possible(self) -> Optional[Event]:
        """
        Compute RCA once at alert time. Uses:
          - XGB SHAP aggregated to base variables
          - AE residuals
        """
        if not self.enable_rca:
            return None
        if self._df_run_full is None or self.state is None:
            return None

        s = self.state
        alert_sample = int(s.current_sample)

        if s.alert_type == "KNOWN_FAULT" and self.xgb is not None:
            # Build exact X_row using scorer's method
            X_row = self.xgb.rebuild_feature_row_at_sample(self._df_run_full, alert_sample)
            if X_row is None:
                top = [{"feature": "XGB_FEATURE_ROW_MISSING", "contribution": float("nan")}]
            else:
                top = xgb_rca_shap(
                    model=self.xgb.model,
                    X_row=X_row,
                    feature_names=self.xgb.predictors,
                    top_k=int(self.rca_top_k),
                    background=self.xgb_background,
                    aggregate_mode="abs_sum",
                )
            return {
                "type": "rca",
                "run_uid": s.run_uid,
                "simulationRun": s.simulationRun,
                "sample": alert_sample,
                "model": "xgb",
                "top_drivers": top,
            }

        if s.alert_type == "ANOMALY_UNCLASSIFIED" and self.ae is not None:
            top = ae_rca_residuals(
                ae_model=self.ae.model,
                scaler=self.ae.scaler,
                df_run=self._df_run_full,
                feature_cols=self.ae.feature_cols,
                window_size=self.ae.window_size,
                alert_sample=alert_sample,
                top_k=int(self.rca_top_k),
            )
            return {
                "type": "rca",
                "run_uid": s.run_uid,
                "simulationRun": s.simulationRun,
                "sample": alert_sample,
                "model": "ae",
                "top_drivers": top,
            }

        return None
