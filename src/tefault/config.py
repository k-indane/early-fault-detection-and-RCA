from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import yaml

@dataclass
class MonitorConfig:
    fault_free_path: str
    faulty_path: str

    # Monitor behavior
    fault_intro_sample: int = 20
    eval_every_n_samples: int = 1

    # XGB
    xgb_enable: bool = True
    xgb_artifact_path: str = "models/xgb_fault_detector.joblib"
    xgb_conf_threshold: float = 0.80
    xgb_consecutive_required: int = 3

    # AE
    ae_enable: bool = True
    ae_artifact_path: str = "models/ae_unknown_fault_detector.joblib"
    ae_consecutive_required: int = 2

    # Which runs to evaluate
    run_mode: str = "all"  # "all" or "unknown_faults_only"
    unknown_faults: Tuple[int, ...] = (16, 17, 18, 19, 20)

    # Excluded faults from this study
    exclude_faults: Tuple[int, ...] = (3, 9, 15)

    # Outputs
    detections_csv: str = "logs/detections.csv"
    summary_csv: str = "logs/summary_by_run.csv"
    rca_jsonl: str = "logs/rca.jsonl"

    @staticmethod
    def _to_tuple(x: Any) -> Tuple[Any, ...]:
        if x is None:
            return tuple()
        if isinstance(x, tuple):
            return x
        if isinstance(x, list):
            return tuple(x)
        return (x,)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MonitorConfig":
        """
        Load MonitorConfig from monitor.yaml.
        """
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            d: Dict[str, Any] = yaml.safe_load(f) or {}

        data = d.get("data", {}) or {}
        monitor = d.get("monitor", {}) or {}
        xgb = d.get("xgb", {}) or {}
        ae = d.get("ae", {}) or {}
        outputs = d.get("outputs", {}) or {}

        ff_path = data.get("fault_free_path", None)
        faulty_path = data.get("faulty_path", None)
        if not ff_path or not faulty_path:
            raise ValueError(
                "monitor.yaml missing required paths. Expected:\n"
                "data:\n"
                "  fault_free_path: ...\n"
                "  faulty_path: ...\n"
            )

        return cls(
            # Data
            fault_free_path=str(ff_path),
            faulty_path=str(faulty_path),
            # Monitor behavior
            fault_intro_sample=int(monitor.get("fault_intro_sample", cls.fault_intro_sample)),
            eval_every_n_samples=int(monitor.get("eval_every_n_samples", cls.eval_every_n_samples)),
            # Run scope
            run_mode=str(d.get("run_mode", cls.run_mode)),
            unknown_faults=tuple(int(x) for x in cls._to_tuple(d.get("unknown_faults", list(cls.unknown_faults)))),
            exclude_faults=tuple(int(x) for x in cls._to_tuple(d.get("exclude_faults", list(cls.exclude_faults)))),
            # XGB
            xgb_enable=bool(xgb.get("enable", cls.xgb_enable)),
            xgb_artifact_path=str(xgb.get("artifact_path", cls.xgb_artifact_path)),
            xgb_conf_threshold=float(xgb.get("confidence_threshold", cls.xgb_conf_threshold)),
            xgb_consecutive_required=int(xgb.get("consecutive_required", cls.xgb_consecutive_required)),
            # AE
            ae_enable=bool(ae.get("enable", cls.ae_enable)),
            ae_artifact_path=str(ae.get("artifact_path", cls.ae_artifact_path)),
            ae_consecutive_required=int(ae.get("consecutive_required", cls.ae_consecutive_required)),
            # Outputs
            detections_csv=str(outputs.get("detections_csv", cls.detections_csv)),
            summary_csv=str(outputs.get("summary_csv", cls.summary_csv)),
            rca_jsonl=str(outputs.get("rca_jsonl", cls.rca_jsonl)),
        )
