"""对照沪深300价格指数与实际持有的IF合约，分解盈亏并检查委托流动性。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
MULTIPLIER = 300


def select_contract(day, cycle):
    """用事先固定的第二周五换月规则选合约。"""
    month = day.to_period("M") if cycle == "month" else day.to_period("Q").end_time.to_period("M")
    first = month.start_time
    roll_date = first + pd.Timedelta(days=(4-first.weekday())%7+7)
    if day >= roll_date:
        month += 1 if cycle == "month" else 3
    return "IF" + month.strftime("%y%m")


def load_prices():
    """仅保留交易所实际交易日，使印在节假日的指数占位行不参与计算。"""
    panel = pd.read_csv(DATA/"IF_panel.csv.gz",parse_dates=["date"]).set_index(["date","contract"]).sort_index()
    spot = pd.concat(pd.DataFrame(json.loads(f.read_bytes())["data"])
                     for f in sorted(DATA.glob("csi300_*.json")))
    spot = spot.set_index(pd.to_datetime(spot.tradeDate,format="%Y%m%d")).sort_index()
    calendar = panel.index.get_level_values("date").unique()
    dates = calendar[(calendar >= "2021-01-01") & (calendar <= "2025-12-31")]
    assert panel.index.is_unique and spot.index.is_unique and dates.isin(spot.index).all()
    return panel, spot.loc[dates,["open","close"]].astype(float)


def interval(panel, spot, date, following):
    """两个端点均取持有合约的开盘，同期价格基差变化可直接核算期货盈亏。"""
    s0, s1 = spot.at[date,"open"], spot.at[following,"open"]
    s_close = spot.at[date,"close"]
    record = dict(date=date,next_date=following,index_open=s0,index_close=s_close,delta_index_points=s1-s0)
    for cycle in ("month", "quarter"):
        code = select_contract(date, cycle)
        f0, f1 = panel.at[(date,code),"open"], panel.at[(following,code),"open"]
        f_close = panel.at[(date,code),"close"]
        delta_basis = (f1-s1)-(f0-s0)
        assert abs(f1-f0 - (s1-s0) - delta_basis)<1e-9
        record.update({cycle+"_contract":code,cycle+"_open":f0,cycle+"_close":f_close,
                       cycle+"_next_open_same":f1,cycle+"_basis_open_points":f0-s0,
                       cycle+"_basis_close_bp":10000*(f_close-s_close)/s_close,
                       cycle+"_delta_basis_points":delta_basis,cycle+"_delta_f_points":f1-f0})
    months = [pd.Period("20"+record[c+"_contract"][2:4]+"-"+record[c+"_contract"][4:6],freq="M")
              for c in ("month","quarter")]
    distance = months[1].ordinal-months[0].ordinal
    spread = record["quarter_close"]-record["month_close"]
    record.update(same_contract=distance==0,maturity_months=distance,spread_close_points=spread,
                  spread_close_bp=1e4*spread/s_close,spread_bp_per_month=(1e4*spread/s_close/distance if distance else np.nan))
    return record


def market_summary(paths):
    """按是否同一合约报告共同指数走势与期限基差。"""
    rows=[]
    for year, group in [("all",paths),*list(paths.groupby(paths.date.dt.year))]:
        different = group.loc[~group.same_contract]
        for scope, subset in [("all",group),("different",different),("same",group.loc[group.same_contract])]:
            rows.append(dict(year=year,scope=scope,days=len(subset),
                month_basis_median_bp=subset.month_basis_close_bp.median(),
                quarter_basis_median_bp=subset.quarter_basis_close_bp.median(),
                month_discount_days_share=subset.month_basis_close_bp.lt(0).mean(),
                quarter_discount_days_share=subset.quarter_basis_close_bp.lt(0).mean(),
                spread_close_median_bp=subset.spread_close_bp.median(),
                spread_negative_share=subset.spread_close_bp.lt(0).mean(),
                spread_bp_per_month_median=subset.spread_bp_per_month.median(),
                index_points=subset.delta_index_points.sum(),
                month_basis_change_points=subset.month_delta_basis_points.sum(),
                quarter_basis_change_points=subset.quarter_delta_basis_points.sum(),
                month_f_points=subset.month_delta_f_points.sum(),quarter_f_points=subset.quarter_delta_f_points.sum(),
                quarter_minus_month_f_points=(subset.quarter_delta_f_points-subset.month_delta_f_points).sum()))
    return pd.DataFrame(rows)


def account_bridge(path):
    """按实际持仓将账户财富差拆成指数、基差与费用三项。"""
    accounts = pd.read_csv(RESULTS/"accounts_daily.csv.gz",parse_dates=["date"]).query('scenario=="base"')
    details=[]
    for (cycle,frequency), ledger in accounts.groupby(["cycle","frequency"]):
        joint = ledger.merge(path,on="date",validate="one_to_one").sort_values("date")
        assert len(joint)==len(path) and joint.contract.eq(joint[cycle+"_contract"]).all()
        weight = joint.lots*MULTIPLIER
        joint["index_component_yuan"] = weight*joint.delta_index_points
        joint["basis_component_yuan"] = weight*joint[cycle+"_delta_basis_points"]
        np.testing.assert_allclose(joint.gross_return*joint.nav_start,joint.index_component_yuan+joint.basis_component_yuan,atol=1e-6)
        for year, sample in [("all",joint),*list(joint.groupby(joint.date.dt.year))]:
            index, basis, fee = sample[["index_component_yuan","basis_component_yuan","fee"]].sum()
            actual = sample.nav_end.iloc[-1]-sample.nav_start.iloc[0]
            residue = actual-(index+basis-fee)
            assert abs(residue) < .001
            details.append(dict(cycle=cycle,frequency=frequency,year=year,intervals=len(sample),index_component_yuan=index,
                                basis_component_yuan=basis,fee_yuan=fee,net_change_yuan=actual,residual_yuan=residue))
    return pd.DataFrame(details)


def order_liquidity(panel):
    """预交易比较采用上一交易日的量仓，执行当日完整成交量仅用于事后诊断。"""
    orders = pd.read_csv(RESULTS/"orders.csv.gz",parse_dates=["date"]).query('scenario=="base"').copy()
    history = panel[["volume","oi"]].reset_index().sort_values(["contract","date"])
    calendar=panel.index.get_level_values("date").unique().sort_values()
    previous_session=dict(zip(calendar[1:],calendar[:-1]))
    orders["prior_date"] = orders.date.map(previous_session)
    joint = orders.merge(history,how="left",on=["contract","date"],validate="many_to_one")
    prior = history.rename(columns={"date":"prior_date","volume":"prior_volume","oi":"prior_oi"})
    joint = joint.merge(prior,how="left",on=["contract","prior_date"],validate="many_to_one")
    assert (joint.qty>0).all() and joint.volume.eq(joint.panel_volume).all() and joint.oi.eq(joint.panel_oi).all()
    for column in ("volume","oi","prior_volume","prior_oi"):
        joint["qty_pct_"+column] = joint.qty.div(joint[column].where(joint[column]>0))*100
    statistic = []
    groups = joint.groupby(["cycle","frequency"])
    for (cycle,freq), frame in groups:
        for scope, legs in [("all",frame),*list(frame.groupby("leg"))]:
            row=dict(cycle=cycle,frequency=freq,leg=scope,legs=len(legs),missing_prior=int(legs.prior_volume.isna().sum()))
            for name in ["qty","volume","oi","prior_volume","prior_oi","qty_pct_volume","qty_pct_oi","qty_pct_prior_volume","qty_pct_prior_oi"]:
                values=legs[name]
                row.update({name+"_median":values.median(),name+"_p95":values.quantile(.95),name+"_max":values.max()})
            statistic.append(row)
    return joint,pd.DataFrame(statistic)


def run(only_prices):
    """生成基差与期限斜率、逐腿流动性、账户盈亏解释三类结果。"""
    panel, spot = load_prices()
    paths=pd.DataFrame(interval(panel,spot,date,next_date) for date,next_date in zip(spot.index,spot.index[1:]))
    paths.to_csv(RESULTS/"market_daily.csv",index=False)
    market_summary(paths).to_csv(RESULTS/"market_year.csv",index=False)
    if not only_prices:
        account_bridge(paths).to_csv(RESULTS/"market_account_bridge.csv",index=False)
        legs,liquidity=order_liquidity(panel)
        legs.to_csv(RESULTS/"market_order_legs.csv.gz",index=False)
        liquidity.to_csv(RESULTS/"market_liquidity.csv",index=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices-only",action="store_true")
    run(parser.parse_args().prices_only)
