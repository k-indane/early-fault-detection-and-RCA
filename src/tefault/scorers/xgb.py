from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from tefault.config import MonitorConfig
from tefault.features import add_ewma, add_lags, infer_base_features_from_predictors

@dataclass
class XGBScorer:
    """
    Captures XGB artifact loading + per run scoring.
    """
    model: object
    label_encoder: object
    predictors: List[str]
    base_features: List[str]
    lags: List[int]
    ewma_span: int

    @classmethod
    def from_artifact(cls, artifact_path: str | Path, project_root: Path) -> "XGBScorer":
        artifact_path = project_root / Path(artifact_path)
        art = joblib.load(artifact_path)

        predictors = list(art["predictors"])
        lags = list(art.get("lags", [1, 2]))
        ewma_span = int(art.get("ewma_span", 3))
        base_features = infer_base_features_from_predictors(predictors)

        return cls(
            model=art["model"],
            label_encoder=art["label_encoder"],
            predictors=predictors,
            base_features=base_features,
            lags=lags,
            ewma_span=ewma_span,
        )

    def score_run(self, df_run: pd.DataFrame, cfg: MonitorConfig, group_col: str = "run_uid") -> pd.DataFrame:
        """
        Returns dataframe with:
          - sample
          - xgb_pred_fault
          - xgb_pred_conf
        """
        tmp = df_run.copy()

        # Check feature columns
        missing = [c for c in self.base_features if c not in tmp.columns]
        if missing:
            raise ValueError(f"Missing base feature columns needed for XGB: {missing[:10]} ...")
        
        # Feature engineering
        tmp = add_lags(tmp, self.base_features, self.lags, group_col)
        tmp = add_ewma(tmp, self.base_features, self.ewma_span, group_col)

        tmp = tmp.dropna(subset=self.predictors).copy()

        # Optional downsampling
        if cfg.eval_every_n_samples > 1:
            tmp = tmp[tmp["sample"] % cfg.eval_every_n_samples == 0].copy()

        X = tmp[self.predictors]

        # Run + decode fault predictions
        y_pred_enc = self.model.predict(X)
        y_pred_fault = self.label_encoder.inverse_transform(y_pred_enc).astype(int)

        # Prediction confidence
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            conf = np.max(proba, axis=1)
        else:
            conf = np.full(len(tmp), np.nan)

        return pd.DataFrame(
            {
                "sample": tmp["sample"].astype(int).values,
                "xgb_pred_fault": y_pred_fault,
                "xgb_pred_conf": conf,
            }
        )

    def build_background(self, data_all_runs: pd.DataFrame, cfg: MonitorConfig) -> Optional[pd.DataFrame]:
        """
        Build background dataset for permutation SHAP fallback.
        Uses fault-free data with engineered features to match.
        """

        # Requires true_fault
        if "true_fault" not in data_all_runs.columns:
            return None
        
        # Select fault free data
        bg = data_all_runs[data_all_runs["true_fault"] == 0].copy()
        if bg.empty:
            return None
        
        # Feature engineering
        bg = add_lags(bg, self.base_features, self.lags, group_col="run_uid")
        bg = add_ewma(bg, self.base_features, self.ewma_span, group_col="run_uid")
        bg = bg.dropna(subset=self.predictors).copy()

        # Optional downsampling
        if cfg.eval_every_n_samples > 1:
            bg = bg[bg["sample"] % cfg.eval_every_n_samples == 0].copy()

        # Limit size
        if len(bg) > 200:
            bg = bg.sample(200, random_state=2)

        return bg[self.predictors].copy()

    def rebuild_feature_row_at_sample(self, df_run: pd.DataFrame, sample: int, group_col: str = "run_uid") -> Optional[pd.DataFrame]:
        """
        Rebuild the XGB feature row at a specific sample (needed for SHAP).
        Returns one row DataFrame with predictors or None if missing.
        """
        tmp = df_run.copy()
        tmp = add_lags(tmp, self.base_features, self.lags, group_col)
        tmp = add_ewma(tmp, self.base_features, self.ewma_span, group_col)
        tmp = tmp.dropna(subset=self.predictors).copy()
        
        # Locate sample row
        x_row = tmp[tmp["sample"] == int(sample)]
        if x_row.empty:
            return None
        return x_row[self.predictors].iloc[[0]]
