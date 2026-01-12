from __future__ import annotations
import argparse
import os
import time
import warnings
from pathlib import Path
# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a live (online) simulation for a single TEP run_uid.")
    p.add_argument("--config", type=str, default="monitor.yaml", help="Path to monitor.yaml")
    p.add_argument("--project-root", type=str, default=".", help="Project root to resolve relative paths")

    # Run selection
    p.add_argument(
        "--run-uid",
        type=str,
        default=None,
        help="Which run_uid to simulate (e.g., F4_R12). If omitted, picks first faulty run if available.",
    )
    p.add_argument(
        "--list-runs",
        action="store_true",
        help="List available run_uids (and exit).",
    )

    # Output control
    p.add_argument(
        "--print",
        dest="print_mode",
        choices=["evaluated", "alert", "rca", "all", "none"],
        default="evaluated",
        help="What events to print. Default: evaluated",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after this many samples (useful for quick tests).",
    )
    p.add_argument(
        "--sleep-ms",
        type=int,
        default=0,
        help="Sleep this many ms between samples to mimic real-time playback (default: 0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve repo root relative to script
    repo_root = Path(__file__).resolve().parents[1]
    project_root = Path(args.project_root).resolve()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        # Allow config relative to repo root if user runs from elsewhere
        cfg_path = (repo_root / cfg_path).resolve()

    from tefault.config import MonitorConfig
    from tefault.monitor import FaultMonitor
    from tefault.data import select_runs
    from tefault.online_engine import OnlineMonitorEngine

    cfg = MonitorConfig.from_yaml(str(cfg_path))

    m = FaultMonitor(cfg, project_root=project_root)
    m.load_artifacts()
    ff, f = m.load_data()
    data = select_runs(ff, f, cfg)

    if data.empty:
        raise RuntimeError("No runs found after select_runs().")

    # List runs mode
    if args.list_runs:
        # Show a quick list with fault labels
        runs = (
            data[["run_uid", "true_fault", "simulationRun"]]
            .drop_duplicates()
            .sort_values(["true_fault", "simulationRun"])
        )
        print(runs.to_string(index=False))
        return

    # Choose run_uid
    run_uid = args.run_uid
    if run_uid is None:
        # Prefer a faulty run, otherwise first available
        faulty_runs = data[data["true_fault"] != 0]["run_uid"].unique().tolist()
        run_uid = faulty_runs[0] if faulty_runs else str(data["run_uid"].iloc[0])

    df_run = data[data["run_uid"] == run_uid].sort_values("sample").copy()
    if df_run.empty:
        raise ValueError(f"run_uid={run_uid!r} not found. Use --list-runs to see available values.")

    # Build background for SHAP
    xgb_bg = None
    if cfg.xgb_enable and m.xgb_scorer is not None:
        xgb_bg = m.xgb_scorer.build_background(data, cfg)

    engine = OnlineMonitorEngine(
        cfg=cfg,
        xgb_scorer=m.xgb_scorer,
        ae_scorer=m.ae_scorer,
        xgb_background=xgb_bg,
        enable_rca=True,
    )

    print(f"\nSimulating run_uid={run_uid} (n_rows={len(df_run)})")
    if args.print_mode != "none":
        print(f"print_mode={args.print_mode}, sleep_ms={args.sleep_ms}, max_samples={args.max_samples}\n")

    n_seen = 0
    for event in engine.stream(df_run):
        if event["type"] == "tick":
            n_seen += 1
            if args.max_samples is not None and n_seen > int(args.max_samples):
                break

        # Decide what to print
        do_print = False
        if args.print_mode == "all":
            do_print = event["type"] in ("start", "tick", "alert", "rca", "done")
        elif args.print_mode == "none":
            do_print = False
        elif args.print_mode == "evaluated":
            do_print = (event["type"] == "tick" and event.get("evaluated") is True) or event["type"] in ("alert", "rca")
        elif args.print_mode == "alert":
            do_print = event["type"] == "alert"
        elif args.print_mode == "rca":
            do_print = event["type"] == "rca"

        if do_print:
            print(event)

        # Optional playback pacing
        if args.sleep_ms and event["type"] == "tick":
            time.sleep(args.sleep_ms / 1000.0)

        # If we printed RCA, we can stop
        if event["type"] == "rca":
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
