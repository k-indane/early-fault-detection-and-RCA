from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tefault.config import MonitorConfig
from tefault.data import select_runs
from tefault.io import append_jsonl, clear_jsonl, prepare_output_paths, write_csv
from tefault.rca import ae_rca_residuals, xgb_rca_shap
from tefault.scorers import AEScorer, XGBScorer
from tefault.state import build_state_timeline, first_alert_info

# Class for running batch monitor, and allowing shared states across runs
class FaultMonitor:
    """
    Batch monitor runner:
      - loads artifacts + data
      - scores XGB + AE per run_uid
      - builds state timeline (NORMAL/SUSPECT/ALERT)
      - triggers RCA once at first alert
      - writes detections + summary CSVs
    """

    def __init__(self, cfg: MonitorConfig, project_root: Optional[Path] = None):
        # Store config + establish project root
        self.cfg = cfg
        self.project_root = project_root or Path.cwd()

        # Load scorer states
        self.xgb_scorer: Optional[XGBScorer] = None
        self.ae_scorer: Optional[AEScorer] = None

        # Background data for permutation-SHAP fallback, build once
        self.xgb_background: Optional[pd.DataFrame] = None

    # Artifact loading
    def load_artifacts(self) -> None:
        if self.cfg.xgb_enable:
            self.xgb_scorer = XGBScorer.from_artifact(self.cfg.xgb_artifact_path, project_root=self.project_root)

        if self.cfg.ae_enable:
            self.ae_scorer = AEScorer.from_artifact(self.cfg.ae_artifact_path, project_root=self.project_root)

    # Data loading (pkl)
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ff = pd.read_pickle(self.project_root / self.cfg.fault_free_path)
        f = pd.read_pickle(self.project_root / self.cfg.faulty_path)
        return ff, f

    # Batch run
    def run(self, write_outputs: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run monitoring over selected runs.
        If write_outputs=False, does not save results to disk.
        """
        # Load artifacts + data
        self.load_artifacts()
        ff, f = self.load_data()
        data = select_runs(ff, f, self.cfg)

        # Build background data once for permutation-SHAP
        if self.cfg.xgb_enable and self.xgb_scorer is not None:
            self.xgb_background = self.xgb_scorer.build_background(data, self.cfg)

        # Prepare output paths
        if write_outputs:
            det_path, sum_path, rca_path = prepare_output_paths(
                project_root=self.project_root,
                detections_csv=self.cfg.detections_csv,
                summary_csv=self.cfg.summary_csv,
                rca_jsonl=self.cfg.rca_jsonl,
            )
            clear_jsonl(rca_path)
        else:
            det_path = sum_path = rca_path = None # Ignore

        # Storage/Collectors
        detections_all: List[pd.DataFrame] = []
        summary_rows: List[Dict[str, Any]] = []

        # Group by run_uid (true_fault + simulationRun)
        for run_uid, df_run in data.groupby("run_uid"):
            df_run = df_run.sort_values("sample").copy()

            true_fault = int(df_run["true_fault"].iloc[0])
            is_faulty_run = int(df_run["is_faulty_run"].iloc[0])
            sim_seed = int(df_run["simulationRun"].iloc[0])

            # Score models
            # if disabled, return minimal dataframe with sample
            # XGB
            if self.cfg.xgb_enable and self.xgb_scorer is not None:
                xgb_df = self.xgb_scorer.score_run(df_run, self.cfg)
            else:
                xgb_df = pd.DataFrame({"sample": df_run["sample"].astype(int)})

            # AE
            if self.cfg.ae_enable and self.ae_scorer is not None:
                ae_df = self.ae_scorer.score_run(df_run, self.cfg)
            else:
                ae_df = pd.DataFrame({"sample": df_run["sample"].astype(int)})

            # Merge predictions, outer merge ensures all samples are kept
            merged = (
                pd.merge(xgb_df, ae_df, on="sample", how="outer")
                .sort_values("sample")
                .reset_index(drop=True)
            )

            # Add identifiers/labels for downstream use
            merged["run_uid"] = str(run_uid)
            merged["simulationRun"] = sim_seed
            merged["true_fault"] = true_fault
            merged["is_faulty_run"] = is_faulty_run

            # Raw signals from scorers, applies decision logic and alert states
            merged = build_state_timeline(merged, self.cfg)
            alert = first_alert_info(merged, self.cfg)
            first_alert_sample = alert.first_alert_sample
            detection_delay = alert.detection_delay_samples
            alert_type_first = alert.alert_type_first
            false_alarm = int(is_faulty_run == 0 and not np.isnan(first_alert_sample))

            # Record XGB fault if first alert is KNOWN_FAULT
            xgb_fault_detected = np.nan
            if alert_type_first == "KNOWN_FAULT" and not np.isnan(first_alert_sample):
                row0 = merged[merged["sample"] == int(first_alert_sample)]
                if not row0.empty and "xgb_pred_fault" in row0.columns:
                    xgb_fault_detected = int(row0["xgb_pred_fault"].iloc[0])

            # Build summary row for this run (one row per run)
            summary_rows.append(
                {
                    "run_uid": str(run_uid),
                    "simulationRun": sim_seed,
                    "true_fault": true_fault,
                    "is_faulty_run": is_faulty_run,
                    "first_alert_sample": first_alert_sample,
                    "detection_delay_samples": detection_delay,
                    "false_alarm": false_alarm,
                    "alert_type_first": alert_type_first,
                    "xgb_fault_detected": xgb_fault_detected,
                }
            )

            # RCA - one event at first alert when otuputs are enabled, at least one alert, and valid write file
            if write_outputs and (not np.isnan(first_alert_sample)) and (rca_path is not None):
                rca_event: Dict[str, Any] = {
                    "run_uid": str(run_uid),
                    "simulationRun": sim_seed,
                    "sample": int(first_alert_sample),
                    "true_fault": true_fault,
                    "alert_type": alert_type_first,
                    "pred_fault": (int(xgb_fault_detected) if not np.isnan(xgb_fault_detected) else None),
                    "model": "none",
                    "top_drivers": [],
                }
                
                # Rebuilds feature vector at first alert for XGB SHAP
                if alert_type_first == "KNOWN_FAULT" and self.cfg.xgb_enable and self.xgb_scorer is not None:
                    rca_event["model"] = "xgb"
                    X_row = self.xgb_scorer.rebuild_feature_row_at_sample(df_run, int(first_alert_sample))
                    if X_row is not None:
                        rca_event["top_drivers"] = xgb_rca_shap(
                            model=self.xgb_scorer.model,
                            X_row=X_row,
                            feature_names=self.xgb_scorer.predictors,
                            top_k=10,
                            background=self.xgb_background,
                            aggregate_mode="abs_sum",
                        )
                    else:
                        rca_event["top_drivers"] = [{"feature": "XGB_FEATURE_ROW_MISSING", "contribution": np.nan}]
                
                # Rebuilds feature vector at first alert for AE residuals
                elif (
                    alert_type_first == "ANOMALY_UNCLASSIFIED"
                    and self.cfg.ae_enable
                    and self.ae_scorer is not None
                ):
                    rca_event["model"] = "ae"
                    rca_event["top_drivers"] = ae_rca_residuals(
                        ae_model=self.ae_scorer.model,
                        scaler=self.ae_scorer.scaler,
                        df_run=df_run,
                        feature_cols=self.ae_scorer.feature_cols,
                        window_size=self.ae_scorer.window_size,
                        alert_sample=int(first_alert_sample),
                        top_k=10,
                    )
                else:
                    rca_event["top_drivers"] = [{"feature": "NO_RCA", "contribution": np.nan}]

                append_jsonl(rca_path, rca_event)

            detections_all.append(merged)

        # Consolidate + write outputs
        detections_df = pd.concat(detections_all, ignore_index=True) if detections_all else pd.DataFrame()
        summary_df = pd.DataFrame(summary_rows)
        if write_outputs and det_path is not None and sum_path is not None:
            write_csv(detections_df, det_path)
            write_csv(summary_df, sum_path)
        return detections_df, summary_df
