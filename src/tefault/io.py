from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd

# Takes root directory and relative paths to resolved paths
def prepare_output_paths(
    project_root: Path, detections_csv: str, summary_csv: str, rca_jsonl: str
) -> Tuple[Path, Path, Path]:
    det_path = project_root / detections_csv
    sum_path = project_root / summary_csv
    rca_path = project_root / rca_jsonl

    det_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    rca_path.parent.mkdir(parents=True, exist_ok=True)

    return det_path, sum_path, rca_path

# Creates or clears file
def clear_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as _:
        pass

# Appends allow streaming data
def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

# Writes df to CSV
def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
