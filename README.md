# MUCAR — 面向不完整遥测的置信度感知 LEO 卫星网络流量预测与韧性路由

硕士论文项目。研究低轨卫星网络中不完整遥测下的链路流量预测、不确定性校准与置信度感知路由。

## 核心文档

| 文档 | 作用 |
|---|---|
| `技术大纲.md` | 研究方法定义（只读基准） |
| `完整工作流.md` | P0–P15 阶段路线图与门控规则 |
| `doc/第N阶段/` | 每阶段的可执行工作步骤、产出文档、交付审计 |

三者关系：技术大纲定义方法，工作流定义路线，`doc/` 记录执行。冲突时以技术大纲方法为准。

## 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
./scripts/run.sh <stage> [seed]      # 跑某阶段，例: ./scripts/run.sh P0 0
./scripts/reproduce.sh [seed]        # 复现工程链路
```

阶段入口为 `src/pN_main.py`，配置在 `configs/config.yaml`，结果落 `results/metrics/`、日志落 `logs/`。

## 目录

```
data/        原始/中间/处理数据
configs/     YAML 配置
src/         代码（data/topo/sim/missing/predict/calib/route/common）
models/      checkpoint
results/     指标
logs/        日志
scripts/     入口脚本
tests/       测试
doc/         阶段文档
```

## 可复现性

每次运行由 `(config_hash, seed, code_version)` 唯一标识，记录于 `results/metrics/<hash>_<seed>_<stage>.json`。依赖精确版本见 `requirements-lock.txt`。
