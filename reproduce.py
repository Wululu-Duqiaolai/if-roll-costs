"""重建论文账户、生成实证图表并逐项对照定稿数字。"""
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def verify():
    """对照公开的定稿舍入值，所有计算值都取自本次生成的结果。"""
    thresholds = pd.read_csv(RESULTS / "signal_thresholds.csv").query("frequency == 'daily'").set_index("signal")
    rolling = thresholds.loc["rolling30"]
    effects = pd.read_csv(RESULTS / "factor_effects.csv").query("scenario == 'base'").set_index("metric")
    wealth = pd.read_csv(RESULTS / "market_account_bridge.csv").query("frequency == 'daily' and year == 'all'").set_index("cycle")
    amounts = wealth[["net_change_yuan", "basis_component_yuan", "fee_yuan"]]
    delta = amounts.loc["quarter"] - amounts.loc["month"]
    loss = pd.read_csv(RESULTS / "signal_loss_intervals.csv").set_index("reference")
    values = {
        "intervals": wealth.at["month", "intervals"],
        "month_turnover": rolling.month_turnover,
        "quarter_turnover": rolling.quarter_turnover,
        "weekly_saving": -effects.at["fee_annual_bp", "month_week_minus_day"],
        "quarter_saving": rolling.base_fee_saving_annual_bp,
        "rolling_threshold": rolling.total_far_ticks_breakeven,
        "regression_threshold": thresholds.at["regularized", "total_far_ticks_breakeven"],
        "wealth_difference": delta.net_change_yuan / 10000,
        "basis_difference": delta.basis_component_yuan / 10000,
        "fee_difference": -delta.fee_yuan / 10000,
        "qlike_difference": loss.at["rolling30", "delta_ml_minus_reference"],
    }
    paper = pd.read_csv(ROOT / "paper_results.csv", dtype={"paper_value": str})
    paper["recomputed"] = paper.metric.map(values)
    paper["matches"] = [format(actual, f".{len(expected.partition('.')[2])}f") == expected
                        for actual, expected in zip(paper.recomputed, paper.paper_value)]
    paper.to_csv(RESULTS / "paper_comparison.csv", index=False)
    print(paper[["metric", "paper_value", "recomputed", "matches"]].to_string(index=False))
    if not paper.matches.all():
        raise ValueError("结果与论文展示精度不符，详见 results/paper_comparison.csv。")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    for stage in ("prepare_data", "if_accounts", "market_structure", "paired_quarters",
                  "signal_comparison", "monthly_budget", "figures"):
        print(f"[{stage}]", flush=True)
        subprocess.run([sys.executable, str(ROOT / (stage + ".py"))], cwd=ROOT, check=True)
    verify()
