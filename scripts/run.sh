#!/usr/bin/env bash
# 统一入口：激活环境 + 加载配置 + 调用阶段 main
# 用法: ./scripts/run.sh <stage> [seed] [extra args]
# 例:   ./scripts/run.sh P0 0
#       ./scripts/run.sh P1 0
#
# 种子语义（见 完整工作流.md §4 与 configs/config.yaml）：
#   一个 eval seed s 同时设 seed.data=seed.model=seed.routing=s，
#   即一次"5 种子实验"中，第 s 次运行的所有随机源都用 s。
#   run.sh 只接受单个 seed 参数并同步覆盖三条种子链。
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

python -m "src.${MODULE}" "$@" \
  seed.data="$SEED" seed.model="$SEED" seed.routing="$SEED"
