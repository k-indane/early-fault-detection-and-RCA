from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from tefault.config import MonitorConfig
from tefault.features import make_ae_windows_for_run

@dataclass
class AEScorer:
    """
    Captures Autoencoder artifact loading + per run scoring.
    """
    model: keras.Model
    scaler: object
    feature_cols: List[str]
    window_size: int
    threshold: float

    @classmethod
    def from_artifact(cls, artifact_path: str | Path, project_root: Path) -> "AEScorer":
        artifact_path = project_root / Path(artifact_path)
        art = joblib.load(artifact_path)

        model = keras.models.load_model(art["model_path"], compile=False)
        scaler = art["scaler"]
        feature_cols = list(art["feature_cols"])
        window_size = int(art["window_size"])
        threshold = float(art["threshold_value"])

        return cls(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            window_size=window_size,
            threshold=threshold,
        )

    def score_run(self, df_run: pd.DataFrame, cfg: MonitorConfig) -> pd.DataFrame:
        """
        Returns dataframe with:
          - sample (window end sample)
          - ae_recon_error
          - ae_flag_raw (1 if recon error > threshold else 0)
        """
        tmp = df_run.copy().sort_values("sample")

        # Check feature columns
        missing = [c for c in self.feature_cols if c not in tmp.columns]
        if missing:
            raise ValueError(f"Missing AE feature columns: {missing[:10]} ...")
        
        # Scale features
        tmp.loc[:, self.feature_cols] = self.scaler.transform(tmp[self.feature_cols])
        
        # Decide where to start evaluating windows (0 for faultfree, 20 for faulty)
        true_fault = int(df_run["true_fault"].iloc[0]) if "true_fault" in df_run.columns else 0
        min_sample = 0 if true_fault == 0 else int(cfg.fault_intro_sample)

        # Make windows
        X_win, end_samples = make_ae_windows_for_run(
            tmp,
            self.feature_cols,
            window_size=self.window_size,
            min_sample=min_sample,
        )

        # Handle no windows case
        if len(X_win) == 0:
            return pd.DataFrame({"sample": tmp["sample"].astype(int)})
        
        # Optional downsampling
        if cfg.eval_every_n_samples > 1:
            keep = end_samples % cfg.eval_every_n_samples == 0
            X_win = X_win[keep]
            end_samples = end_samples[keep]

        # Run
        recon = self.model.predict(X_win, verbose=0)
        recon_error = np.mean(np.square(X_win - recon), axis=1)

        out = pd.DataFrame(
            {
                "sample": end_samples.astype(int),
                "ae_recon_error": recon_error,
            }
        )
        out["ae_flag_raw"] = (out["ae_recon_error"] > self.threshold).astype(int)
        return out
