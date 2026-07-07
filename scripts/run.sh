#!/usr/bin/env bash
# 统一入口：激活环境 + 加载配置 + 调用阶段 main
# 用法: ./scripts/run.sh <stage> [seed] [extra args]
# 例:   ./scripts/run.sh P0 0
#       ./scripts/run.sh P1 0
set -euo pipefail

STAGE=${1:?usage: run.sh <stage> [seed] [extra args]}
SEED=${2:-0}
shift $(( $# >= 2 ? 2 : $# ))

# 项目根（脚本在 scripts/ 下）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 激活虚拟环境（若存在）
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# stage -> 模块名（小写）
MODULE=$(echo "$STAGE" | tr '[:upper:]' '[:lower:]')_main

python -m "src.${MODULE}" "$@" seed.data="$SEED"
