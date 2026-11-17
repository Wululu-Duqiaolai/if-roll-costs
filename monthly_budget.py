"""衡量逐月实现风险偏离账户预算的幅度。"""
from pathlib import Path

import numpy as np
import pandas as pd


def monthly_risk(daily):
    """逐月按交易日起点归属区间，以免跨假期收益重复计入。"""
    months = pd.to_datetime(daily.date).dt.to_period('M')
    groups = daily.groupby(['signal', 'cycle', 'frequency', months.rename('month')])
    values = groups.net_return.agg(['count', 'std']).reset_index()
    values['vol_pct'] = 100 * np.sqrt(252) * values.pop('std')
    values['abs_gap_pp'] = abs(values.vol_pct - 15)
    assert values['count'].min() >= 10 and len(values) == 3*4*60
    return values


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent/'results')
    results = parser.parse_args().results
    monthly = monthly_risk(pd.read_csv(results/'signal_daily.csv.gz'))
    monthly.to_csv(results/'signal_monthly_risk.csv', index=False)
    print(monthly.groupby(['signal','cycle','frequency']).abs_gap_pp.mean().round(3))
