"""P6 阶段入口：数据可信度验证（实验一）★ 硬门控 ★。

审计 P1–P5 全部产出，生成审计报告 + 实验一报告。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    ExperimentRecord, get_logger, load_config, resolve_paths, seed_everything,
)
from src.audit import (  # noqa: E402
    audit_od_conservation, audit_link_conservation, audit_commodity_load_restore,
    audit_burst_before_routing, audit_failure_before_label, audit_missing_unchanges_truth,
    compare_policies, spatial_correlation, load_drop_relation, periodicity_check,
    audit_missing_stats, audit_reproducibility,
)


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.seed.data)
    log = get_logger("p6_main", "P6")
    rec = ExperimentRecord(cfg=cfg, seed=cfg.seed.data, stage="P6")
    paths = resolve_paths(cfg)
    out_dir = paths["data_processed"] / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== P6 数据可信度验证（硬门控）===")

    report = {}

    # A. 守恒
    log.info("A. 守恒审计...")
    report["od_conservation"] = audit_od_conservation(cfg)
    report["link_conservation"] = audit_link_conservation(cfg)
    report["commodity_load_restore"] = audit_commodity_load_restore(cfg)
    log.info(f"  OD: {report['od_conservation']['pass']} | "
             f"link: {report['link_conservation']['pass']} | "
             f"commodity_load: {report['commodity_load_restore']['pass']}")

    # B. 因果顺序
    log.info("B. 因果顺序审计...")
    report["burst_before_routing"] = audit_burst_before_routing(cfg)
    report["failure_before_label"] = audit_failure_before_label(cfg)
    report["missing_unchanges_truth"] = audit_missing_unchanges_truth(cfg)
    log.info(f"  burst: {report['burst_before_routing']['pass']} | "
             f"failure: {report['failure_before_label']['pass']} | "
             f"missing: {report['missing_unchanges_truth']['pass']}")

    # C. 跨策略与统计
    log.info("C. 跨策略与统计审计...")
    report["compare_policies"] = compare_policies(cfg)
    report["spatial_correlation"] = spatial_correlation(cfg)
    report["load_drop_relation"] = load_drop_relation(cfg)
    report["periodicity"] = periodicity_check(cfg)
    log.info(f"  policies: {report['compare_policies']['pass']} | "
             f"spatial: {report['spatial_correlation']['pass']} | "
             f"load_drop: {report['load_drop_relation']['pass']} | "
             f"periodicity: {report['periodicity']['pass']}")

    # D. 缺失与可复现
    log.info("D. 缺失与可复现审计...")
    report["missing_stats"] = audit_missing_stats(cfg)
    report["reproducibility"] = audit_reproducibility(cfg)
    log.info(f"  missing: {report['missing_stats']['pass']} | "
             f"repro: {report['reproducibility']['pass']}")

    # 汇总 PASS/FAIL
    all_checks = ["od_conservation", "link_conservation", "commodity_load_restore",
                  "burst_before_routing", "failure_before_label", "missing_unchanges_truth",
                  "compare_policies", "spatial_correlation", "load_drop_relation",
                  "periodicity", "missing_stats", "reproducibility"]
    passed = sum(1 for k in all_checks if report[k].get("pass"))
    total = len(all_checks)
    report["summary"] = {"passed": passed, "total": total,
                         "all_pass": passed == total}

    with open(out_dir / "p6_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # 指标
    for k in all_checks:
        rec.log_metric(f"audit_{k}", int(report[k].get("pass", False)))
    rec.log_metric("n_passed", passed)
    rec.log_metric("n_total", total)

    rec.log_output(str(out_dir))
    out = rec.finish(status="success")
    log.info(f"P6 完成: {passed}/{total} PASS | 记录: {out}")
    if passed == total:
        log.info("P6 硬门控通过，可进入 P7")
    else:
        log.warning(f"P6 硬门控未通过：{total - passed} 项 FAIL，需修复后重审")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
