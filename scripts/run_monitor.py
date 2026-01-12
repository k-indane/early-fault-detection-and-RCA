from __future__ import annotations
import argparse
from pathlib import Path
from tefault.config import MonitorConfig
from tefault.monitor import FaultMonitor
import os
import warnings
# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")

# Builds command-line argument parser and returns parsed args
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run batch fault monitoring + RCA logging.")
    p.add_argument(
        "--config",
        type=str,
        default="monitor.yaml",
        help="Path to monitor.yaml (default: monitor.yaml)",
    )
    p.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory used to resolve relative paths (default: .)",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Run monitor but do not write outputs (CSVs/JSONL).",
    )
    return p.parse_args()

 # Run fault monitor
def main(config_path: str, project_root: str = ".", write_outputs: bool = True) -> None:
    cfg = MonitorConfig.from_yaml(config_path)
    monitor = FaultMonitor(cfg=cfg, project_root=Path(project_root))

    detections_df, summary_df = monitor.run(write_outputs=write_outputs)

    # Basic console summary
    n_runs = summary_df.shape[0]
    n_alerts = int((summary_df["alert_type_first"] != "NONE").sum()) if "alert_type_first" in summary_df.columns else 0
    n_false_alarms = int(summary_df.get("false_alarm", 0).sum()) if "false_alarm" in summary_df.columns else 0

    print("\nBatch Monitor Complete")
    print(f"Runs processed: {n_runs}")
    print(f"Runs with alert: {n_alerts}")
    print(f"False alarms: {n_false_alarms}")

    if write_outputs:
        print("\nOutputs written to:")
        print(f"  - {cfg.detections_csv}")
        print(f"  - {cfg.summary_csv}")
        print(f"  - {cfg.rca_jsonl}")
    else:
        print("\n(write_outputs=False) No files were written.")

# Only run if executed directly
if __name__ == "__main__":
    args = parse_args()
    main(
        config_path=args.config,
        project_root=args.project_root,
        write_outputs=(not args.no_write),
    )
