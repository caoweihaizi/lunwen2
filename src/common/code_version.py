"""代码版本指纹。

git 仓库时返回 commit SHA；否则对 src/ 全部 .py 取 SHA256 作为指纹。
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _src_hash() -> str:
    h = hashlib.sha256()
    for py in sorted(SRC_DIR.rglob("*.py")):
        rel = py.relative_to(PROJECT_ROOT).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(py.read_bytes())
        h.update(b"\0")
    return "src:" + h.hexdigest()[:16]


def get_code_version() -> str:
    """返回代码版本指纹。优先 git commit，否则 src 哈希。"""
    commit = _git_commit()
    if commit:
        return commit
    return _src_hash()
