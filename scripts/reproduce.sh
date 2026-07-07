#!/usr/bin/env bash
# 复现脚本：固定种子从空结果目录跑通一个最小流程。
# P0 阶段只验证工程链路：跑 P0 自检，确认 (config_hash, seed, code_version) 可记录。
# 用法: ./scripts/reproduce.sh [SEED]   默认 SEED=0
set -euo pipefail

SEED=${1:-0}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "=== reproduce: SEED=$SEED ==="
python -m src.p0_main seed.data="$SEED"

echo "=== 最近 P0 记录 ==="
ls -1t results/metrics/*_${SEED}_P0.json 2>/dev/null | head -1 | xargs -I{} sh -c 'echo "记录文件: {}"; cat {}'
