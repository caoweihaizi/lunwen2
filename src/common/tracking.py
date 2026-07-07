"""实验追踪：每次运行落盘一个 JSON 记录。

ExperimentRecord 记录 config_hash / seed / stage / code_version /
python_version / 起止时间 / status / metrics / outputs，落盘到
results/metrics/<config_hash>_<seed>_<stage>.json。
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .code_version import get_code_version
from .config import PROJECT_ROOT, config_hash, resolve_paths


def _now_iso() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ExperimentRecord:
    """单次实验运行的记录器。结束须调用 finish()。"""

    def __init__(self, cfg, seed: int, stage: str) -> None:
        self.cfg = cfg
        self.seed = seed
        self.stage = stage
        self.chash = config_hash(cfg)
        self.data = {
            "config_hash": self.chash,
            "seed": seed,
            "stage": stage,
            "code_version": get_code_version(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "start_time": _now_iso(),
            "end_time": None,
            "status": "running",
            "metrics": {},
            "outputs": [],
        }
        self._paths = resolve_paths(cfg)

    def log_metric(self, name: str, value: Any, step: int | None = None) -> None:
        entry = {"name": name, "value": value}
        if step is not None:
            entry["step"] = step
        self.data["metrics"][name] = entry

    def log_output(self, path: str) -> None:
        self.data["outputs"].append(str(path))

    def finish(self, status: str = "success") -> Path:
        self.data["end_time"] = _now_iso()
        self.data["status"] = status
        out_dir = self._paths["results_metrics"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.chash}_{self.seed}_{self.stage}.json"
        out_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out_path
