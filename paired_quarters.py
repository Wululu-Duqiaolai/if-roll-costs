"""冻结账户上按持有期开始日汇总双路线，并成对抽取20个整季度。"""
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
SEED = 20261114
DRAWS = 2000


def align(daily):
    """在同一持有区间按当日交易前权益度量费用。"""
    base = daily.query("scenario=='base'").copy()
    base["cost_ratio"] = base.fee / base.nav_start
    wide = base.pivot(index=["date", "next_date"], columns=["cycle", "frequency"],
                      values=["net_return", "cost_ratio"])
    assert len(wide) == 1211 and wide.notna().all().all()
    diffs = {}
    for f in ("daily", "weekly"):
        diffs[(f,"net_return_diff_pp")] = 25200 * (wide["net_return","quarter",f] - wide["net_return","month",f])
        diffs[(f,"cost_saving_bp")] = 2520000 * (wide["cost_ratio","month",f] - wide["cost_ratio","quarter",f])
    frame = pd.DataFrame(diffs).droplevel("next_date")
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["frequency","metric"])
    return frame


def main():
    """同一组季度索引适用于两种频率及两项指标。"""
    source = pd.read_csv(RESULTS/"accounts_daily.csv.gz", parse_dates=["date","next_date"])
    diffs = align(source)
    annual = diffs.groupby(diffs.index.year)
    years = annual.mean().stack(["frequency","metric"]).rename("estimate").reset_index()
    years["trading_intervals"] = years["date"].map(annual.size())
    years.rename(columns={"date":"year"}).to_csv(RESULTS/"quarterly_years.csv",index=False)
    quarters = diffs.groupby(diffs.index.to_period("Q"))
    counts, totals = quarters.size(), quarters.sum()
    assert len(counts) == 20 and counts.index[0] == pd.Period("2021Q1") and counts.index[-1] == pd.Period("2025Q4")
    blocks = quarters.mean().stack(["frequency","metric"]).rename("estimate").reset_index()
    blocks["trading_intervals"] = blocks["date"].map(counts)
    blocks.rename(columns={"date":"quarter"}).to_csv(RESULTS/"quarterly_blocks.csv",index=False)
    picks = np.random.default_rng(SEED).integers(0, len(counts), size=(DRAWS,len(counts)))
    denominator = counts.to_numpy()[picks].sum(axis=1)
    draws = totals.to_numpy()[picks].sum(axis=1) / denominator[:,None]
    intervals = []
    for j,(frequency,metric) in enumerate(totals.columns):
        sample = diffs[frequency,metric]
        annual_values = sample.groupby(sample.index.year).mean()
        ci = np.quantile(draws[:,j],[.025,.975])
        intervals.append(dict(frequency=frequency,metric=metric,estimate=sample.mean(),
                              q025=ci[0],q975=ci[1],positive_years=int((annual_values>0).sum()),
                              positive_quarters=int((totals[frequency,metric]>0).sum()),
                              start="2021-01-04",last_mark="2025-12-31",days=len(sample),
                              blocks=len(counts),draws=DRAWS,seed=SEED))
    pd.DataFrame(intervals).to_csv(RESULTS/"quarterly_intervals.csv",index=False)


if __name__ == "__main__":
    main()
