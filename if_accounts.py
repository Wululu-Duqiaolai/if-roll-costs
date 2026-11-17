"""用中金所IF实盘合约日行情比较调仓与到期月份的执行成本。"""
from dataclasses import dataclass
from pathlib import Path
from math import ceil, floor

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CAPITAL = 100_000_000
MULTIPLIER = 300
TICK_POINTS = 0.2
MARGIN = 0.12
ANNUAL = 252


@dataclass(frozen=True)
class Fees:
    name: str
    near_ticks: float
    far_ticks: float
    exchange_bps: float = 0.23
    broker_bps: float = 0.46
    filing_yuan: float = 1.0


def selected_contract(date, cycle):
    """第二个周五开盘执行预先排定的移仓。"""
    maturity = date.to_period("M") if cycle == "month" else date.to_period("Q").end_time.to_period("M")
    first = maturity.start_time
    switch = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 7)
    if date >= switch:
        maturity += 1 if cycle == "month" else 3
    return "IF" + maturity.strftime("%y%m")


def contract_path(panel, cycle):
    """同合约持仓区间计价，隔夜旧仓与日间新仓分段计算历史波动。"""
    dates = sorted(panel.index.get_level_values("date").unique())
    records = []
    for i in range(1, len(dates)-1):
        previous, date, following = dates[i-1:i+2]
        active = selected_contract(date, cycle)
        old = selected_contract(previous, cycle)
        opening = panel.at[(date, active), "open"]
        old_open = panel.at[(date, old), "open"]
        closing = panel.at[(date, active), "close"]
        old_close = panel.at[(previous, old), "close"]
        records.append(dict(date=date, next_date=following, prev_date=previous,
                            contract=active, old_contract=old, roll=active != old,
                            opening=opening, closing=closing, old_open=old_open,
                            next_open=panel.at[(following, active), "open"],
                            prev_close=panel.at[(previous, active), "close"],
                            settle=panel.at[(date, active), "settle"],
                            historic_return=old_open / old_close * closing / opening - 1))
    return pd.DataFrame(records)


def order(date, previous, panel, code, signed_qty, leg, fees):
    """按实际合约月份分类交易腿，保存成交后及前一日可得流动性。"""
    quote = panel.loc[(date, code)]
    maturity = pd.Period("20"+code[2:4]+"-"+code[4:6], freq="M")
    months_ahead = maturity.ordinal - date.to_period("M").ordinal
    far = months_ahead >= 2
    qty = abs(signed_qty)
    notional = qty * MULTIPLIER * quote.open
    exchange_fee = notional * fees.exchange_bps / 10000
    broker_fee = notional * fees.broker_bps / 10000
    slippage = (fees.far_ticks if far else fees.near_ticks) * qty * TICK_POINTS * MULTIPLIER
    children = ceil(qty/20)
    filing = children * fees.filing_yuan
    prior = panel.loc[(previous,code)]
    return dict(date=date, contract=code, leg=leg, buy_sell="buy" if signed_qty>0 else "sell",
                qty=qty, signed_qty=signed_qty, price=quote.open, notional=notional,
                months_ahead=months_ahead, is_far=far, panel_volume=quote.volume, panel_oi=quote.oi,
                lag_volume=prior.volume, lag_oi=prior.oi, child_orders=children, filing=filing,
                exchange_fee=exchange_fee,broker_fee=broker_fee,slippage=slippage,
                fee=exchange_fee+broker_fee+filing+slippage)


def simulate(path, panel, cycle, frequency, fees):
    """根据前收已知权益报单，用真实开盘估值，并按日累计扣费。"""
    tag = dict(scenario=fees.name, cycle=cycle, frequency=frequency)
    nav = known_nav = CAPITAL
    held = 0
    prev_week = None
    daily, orders = [], []
    for row in path.itertuples(index=False):
        week = row.date.to_period("W")
        size_now = frequency=="daily" or week!=prev_week
        lots = floor(known_nav * row.target / (row.prev_close * MULTIPLIER)) if size_now else held
        if row.roll:
            changes = [(row.old_contract,-held,"close_old"), (row.contract,lots,"open_new")]
        else:
            changes = [(row.contract, lots-held, "resize")]
        legs = [order(row.date, row.prev_date, panel, code, qty, leg, fees)
                for code,qty,leg in changes if qty]
        orders.extend({**tag,**leg,"nav_start":nav} for leg in legs)
        costs = {col:sum(leg[col] for leg in legs) for col in
                 ("fee","exchange_fee","broker_fee","filing","slippage")}
        traded = sum(leg["notional"] for leg in legs)
        matched = min(held,lots) * MULTIPLIER * (row.old_open+row.opening) if row.roll else 0.
        pnl = lots * MULTIPLIER * (row.next_open-row.opening)
        end = nav + pnl - costs["fee"]
        settlement_equity = nav - costs["fee"] + lots * MULTIPLIER * (row.settle-row.opening)
        margin = lots * MULTIPLIER * row.settle * MARGIN
        daily.append(dict(**tag,date=row.date,next_date=row.next_date,contract=row.contract,
                          old_contract=row.old_contract,roll=row.roll,target=row.target,lots=lots,old_lots=held,
                          nav_start=nav,nav_end=end,preclose_nav=known_nav,
                          open=row.opening,next_open_same_contract=row.next_open,
                          exposure=lots*MULTIPLIER*row.opening/nav,net_return=(end-nav)/nav,
                          gross_return=pnl/nav,turnover=traded/nav,matched_roll_turnover=matched/nav,
                          exchange_settlement=row.settle,settlement_equity=settlement_equity,
                          margin_use=margin/settlement_equity, **costs))
        known_nav = nav - costs["fee"] + lots * MULTIPLIER * (row.closing-row.opening)
        nav,held,prev_week=end,lots,week
    return pd.DataFrame(daily),pd.DataFrame(orders)


def summarize(p, orders):
    """计算可比较的年化资金指标及历史路径最大回撤。"""
    r=p.net_return
    wealth = np.r_[CAPITAL,p.nav_end]
    years=len(p)/ANNUAL
    normalizer=ANNUAL / len(p)
    component = {col+"_annual_bp": 10000 * (p[col]/p.nav_start).sum()*normalizer for col in
                 ("fee","exchange_fee","broker_fee","filing","slippage")}
    return dict(annual_net_arithmetic=r.mean()*ANNUAL,cagr=(wealth[-1]/wealth[0])**(1/years)-1,
                ann_vol=r.std()*np.sqrt(ANNUAL),max_drawdown=(wealth/np.maximum.accumulate(wealth)-1).min(),
                mean_exposure=p.exposure.mean(),turnover_annual=p.turnover.mean()*ANNUAL,
                matched_roll_turnover_annual=p.matched_roll_turnover.mean()*ANNUAL,
                matched_roll_share_money=(p.matched_roll_turnover*p.nav_start).sum()/orders.notional.sum(),
                gross_arithmetic=p.gross_return.mean()*ANNUAL,peak_margin_usage=p.margin_use.max(),
                target_rmse=np.sqrt(((p.target-p.exposure)**2).mean()),rolls=p.roll.sum(),trade_days=(p.turnover>0).sum(),
                legs=len(orders),child_orders=orders.child_orders.sum(),terminal=wealth[-1]/wealth[0],**component)


def break_even(stats, orders):
    """固定基准订单和每日交易前净值，推算同频率两路线的费用交点。"""
    rows=[]
    base=stats[stats.scenario.eq("base")].set_index(["cycle","frequency"])
    for freq in ("daily","weekly"):
        m,q=(base.loc[cycle,freq] for cycle in ("month","quarter"))
        legs=orders[(orders.scenario=="base") & (orders.frequency==freq) & orders.is_far]
        sensitivities=(legs.qty * TICK_POINTS * MULTIPLIER/legs.nav_start).groupby(legs.cycle).sum() * ANNUAL/1211*10000
        diff=sensitivities.get("quarter",0)-sensitivities.get("month",0)
        gap=m.fee_annual_bp-q.fee_annual_bp
        rows.append(dict(frequency=freq,base_fee_saving_annual_bp=gap,
                         month_turnover=m.turnover_annual,quarter_turnover=q.turnover_annual,
                         equal_spend_unit_price_ratio=m.turnover_annual/q.turnover_annual,
                         month_unit_cost_bp=m.fee_annual_bp/m.turnover_annual,
                         observed_quarter_unit_cost_bp=q.fee_annual_bp/q.turnover_annual,
                         breakeven_quarter_unit_cost_bp=m.fee_annual_bp/q.turnover_annual,
                         incremental_far_tick_annual_bp=diff,
                         additional_far_ticks_breakeven=gap/diff,
                         total_far_ticks_breakeven=1+gap/diff))
    return pd.DataFrame(rows)


def factor_effects(stats):
    """报告条件效应、两水平等权边际差和差分之差。"""
    rows=[]
    indicators=["turnover_annual","matched_roll_turnover_annual","fee_annual_bp","cagr","annual_net_arithmetic","ann_vol","max_drawdown","peak_margin_usage","target_rmse"]
    for scenario,block in stats.groupby("scenario"):
        indexed=block.set_index(["cycle","frequency"])
        for metric in indicators:
            md,mw,qd,qw=(indexed.at[key,metric] for key in
                          (("month","daily"),("month","weekly"),("quarter","daily"),("quarter","weekly")))
            rows.append(dict(scenario=scenario,metric=metric,month_week_minus_day=mw-md,quarter_week_minus_day=qw-qd,
                             daily_quarter_minus_month=qd-md,weekly_quarter_minus_month=qw-mw,
                             frequency_effect=((mw-md)+(qw-qd))/2,cycle_effect=((qd-md)+(qw-mw))/2,
                             interaction=(qw-qd)-(mw-md)))
    return pd.DataFrame(rows)


def main():
    """构建统一风险信号，将价格差异与交易腿的成本分开入账。"""
    panel=pd.read_csv(ROOT/"data/IF_panel.csv.gz",parse_dates=["date"]).set_index(["date","contract"]).sort_index()
    series={cycle:contract_path(panel,cycle) for cycle in ("month","quarter")}
    signal=(0.15/(series["month"].historic_return.rolling(30).std().shift(1)*np.sqrt(ANNUAL))).clip(upper=1.)
    paths={c:p.assign(target=signal).query("date >= '2021-01-01'") for c,p in series.items()}
    assert all(len(p)==1211 and p.target.notna().all() for p in paths.values())
    costs=[Fees("legacy",0,0,1,0,0), Fees("base",1,1), Fees("low",.5,.5), Fees("high",2,2), Fees("far_premium",1,2)]
    histories,orders,summaries=[],[],[]
    for fees in costs:
        for cycle,path in paths.items():
            for frequency in ("daily","weekly"):
                days,legs=simulate(path,panel,cycle,frequency,fees)
                histories.append(days)
                orders.append(legs)
                summaries.append(dict(scenario=fees.name,cycle=cycle,frequency=frequency,**summarize(days,legs)))
        if fees.name=="base":
            cross=break_even(pd.DataFrame(summaries),pd.concat(orders,ignore_index=True))
            stress_ticks=int(np.ceil(cross.total_far_ticks_breakeven.max()))+1
            costs.append(Fees("far_stress",1,stress_ticks))
    stats=pd.DataFrame(summaries)
    daily=pd.concat(histories,ignore_index=True)
    legs=pd.concat(orders,ignore_index=True)
    np.testing.assert_allclose(daily.net_return,daily.gross_return-daily.fee/daily.nav_start,atol=1e-13)
    old=pd.read_csv(ROOT/"legacy_reference.csv").set_index(["cycle","frequency"])
    new=stats.query("scenario=='legacy'").set_index(["cycle","frequency"])
    diff=pd.DataFrame({key:new[f1]-old[f2] for key,f1,f2 in
                       [("turnover_diff","turnover_annual","turnover_annual"),("vol_diff","ann_vol","vol_net"),
                        ("terminal_diff","terminal","terminal"),("roll_share_diff","matched_roll_share_money","roll_share_notional")]})
    assert diff.abs().max().max()<1e-11
    diff.to_csv(OUT/"legacy_check.csv")
    stats.to_csv(OUT/"accounts_summary.csv",index=False)
    daily.to_csv(OUT/"accounts_daily.csv.gz",index=False)
    legs.to_csv(OUT/"orders.csv.gz",index=False)
    factor_effects(stats).to_csv(OUT/"factor_effects.csv",index=False)
    cross.to_csv(OUT/"breakeven.csv",index=False)
    pd.DataFrame([vars(fee) for fee in costs]).to_csv(OUT/"cost_scenarios.csv",index=False)


if __name__=="__main__":
    main()
