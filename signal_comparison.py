"""在同一持仓时钟上比较三种风险信号及其移仓代价。"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler


SIGNALS = ("rolling30", "ewma94", "regularized")
WINDOWS = (1, 5, 22)
FEATURES = [f"log1p_msq_{k}" for k in WINDOWS]


def forecasts(path):
    """每年以开始预测前最后一个收盘为截点，使用届时已结束的持有区间。"""
    frame = path[["date", "prev_date", "next_date", "contract"]].copy()
    historical = path.historic_return
    squares = (100 * historical).pow(2)
    frame["observed_pct_squared"] = (100 * (path.next_open / path.opening - 1)).pow(2)
    frame["rolling30"] = (100 * historical.rolling(30).std().shift(1)).pow(2)
    frame["ewma94"] = squares.ewm(alpha=.06, adjust=False).mean().shift(1)
    for k, name in zip(WINDOWS, FEATURES):
        # 百分比平方的零观测仍有明确的特征值
        frame[name] = np.log1p(squares.rolling(k).mean().shift(1))

    frame["regularized"] = np.nan
    fits = []
    for year in range(2021, 2026):
        test = frame.loc[frame.date.dt.year.eq(year)]
        cutoff = test.prev_date.iloc[0]
        train = frame.loc[frame.next_date.le(cutoff)].dropna(subset=FEATURES)
        scaler = StandardScaler().fit(train[FEATURES])
        model = TweedieRegressor(power=1, link="log", alpha=.1,
                                 solver="lbfgs", max_iter=200, tol=1e-9)
        model.fit(scaler.transform(train[FEATURES]), train.observed_pct_squared)
        frame.loc[test.index, "regularized"] = model.predict(scaler.transform(test[FEATURES]))
        fits.append(dict(year=year, train_start=train.date.iloc[0],
                         last_label_open=train.next_date.iloc[-1], prediction_start=test.date.iloc[0],
                         previous_close=cutoff, train_n=len(train), test_n=len(test),
                         intercept=model.intercept_,
                         **{f"coef_{k}": c for k,c in zip(WINDOWS,model.coef_)},
                         **{f"mean_{k}": m for k,m in zip(WINDOWS,scaler.mean_)},
                         **{f"scale_{k}": s for k,s in zip(WINDOWS,scaler.scale_)}))
    result = frame.loc[frame.date.ge("2021-01-01")].copy()
    assert len(result) == 1211 and (result[list(SIGNALS)] > 0).all().all()
    return result, pd.DataFrame(fits)


def loss_table(predictions):
    """相对损失抵消共同的响应项，因此零收益日也可纳入。"""
    rows = []
    years = predictions.groupby(predictions.date.dt.year)
    for year, part in [("all", predictions), *list(years)]:
        variance = part.observed_pct_squared
        for signal in SIGNALS:
            h = part[signal]
            score = np.log(h) + variance/h
            rows.append(dict(year=year, signal=signal, intervals=len(part),
                             mean_qlike_score=score.mean(),
                             delta_rolling30=(np.log(h/part.rolling30) + variance*(1/h-1/part.rolling30)).mean(),
                             delta_ewma94=(np.log(h/part.ewma94) + variance*(1/h-1/part.ewma94)).mean(),
                             mean_pred_pct_squared=h.mean(), mean_obs_pct_squared=variance.mean()))
    return pd.DataFrame(rows)


def account_comparison(account, panel, paths, predictions):
    """逐个重建真实整手账户，费用变化反馈到随后的可交易手数。"""
    all_summaries, thresholds, histories, far5_summaries = [], [], [], []
    base = account.Fees("base", 1, 1)
    far5 = account.Fees("far5", 1, 5)
    for signal in SIGNALS:
        target = (.15 / (np.sqrt(252 * predictions[signal]) / 100)).clip(upper=1.)
        summaries, orders = [], []
        for cycle, path in paths.items():
            active = path.loc[predictions.index].assign(target=target)
            assert active.date.equals(predictions.date)
            for frequency in ("daily", "weekly"):
                daily, legs = account.simulate(active, panel, cycle, frequency, base)
                summaries.append(dict(signal=signal, scenario="base", cycle=cycle,
                                      frequency=frequency, **account.summarize(daily, legs)))
                orders.append(legs)
                thin = daily[["date", "next_date", "cycle", "frequency", "net_return", "gross_return",
                              "exposure", "target", "lots", "turnover"]].copy()
                thin["signal"] = signal
                thin["cost_bp"] = 10000*daily.fee/daily.nav_start
                histories.append(thin)
                # 压力情景的费用逐日反馈给下一次定手，不能只调整期末净值
                daily5, legs5 = account.simulate(active, panel, cycle, frequency, far5)
                far5_summaries.append(dict(signal=signal, scenario="far5", cycle=cycle,
                                           frequency=frequency, **account.summarize(daily5, legs5)))
        stats = pd.DataFrame(summaries)
        thresholds.append(account.break_even(stats, pd.concat(orders)).assign(signal=signal))
        all_summaries.append(stats)
    return pd.concat(all_summaries), pd.concat(thresholds), pd.concat(histories), pd.DataFrame(far5_summaries)


def quarter_intervals(predictions):
    """成对抽取相同自然季度，按各季度的真实持有区间数加权。"""
    y = predictions.observed_pct_squared
    h = predictions.regularized
    paired = pd.DataFrame({
        name: np.log(h/predictions[name]) + y*(1/h-1/predictions[name])
        for name in ("rolling30", "ewma94")
    })
    blocks = paired.groupby(predictions.date.dt.to_period("Q"))
    counts, totals = blocks.size().to_numpy(), blocks.sum().to_numpy()
    assert len(counts) == 20
    picks = np.random.default_rng(20261114).integers(0, 20, size=(2000,20))
    draws = totals[picks].sum(axis=1)/counts[picks].sum(axis=1)[:,None]
    return pd.DataFrame(dict(reference=paired.columns, delta_ml_minus_reference=paired.mean().values,
                             q025=np.quantile(draws,.025,axis=0),q975=np.quantile(draws,.975,axis=0),
                             blocks=len(counts), draws=len(draws), seed=20261114))


def run(project, output):
    """复用主分析的选约、盈亏及费用逻辑，保持三种信号的执行条件一致。"""
    import if_accounts as account

    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(project/"data/IF_panel.csv.gz", parse_dates=["date"]).set_index(["date", "contract"]).sort_index()
    paths = {cycle: account.contract_path(panel, cycle) for cycle in ("month", "quarter")}
    predictions, fits = forecasts(paths["month"])
    loss = loss_table(predictions)
    stats, thresholds, histories, far5 = account_comparison(account, panel, paths, predictions)
    keys = ["cycle", "frequency"]
    measures = ["ann_vol","cagr","fee_annual_bp","turnover_annual","terminal"]
    original = pd.read_csv(project/"results/accounts_summary.csv").query("scenario=='base'").set_index(keys).sort_index()
    replica = stats.query("signal=='rolling30'").set_index(keys).sort_index()
    np.testing.assert_allclose(replica[measures],original[measures],rtol=1e-12)
    for name, table in [("signal_predictions.csv.gz",predictions), ("signal_fits.csv",fits),
                        ("signal_loss.csv",loss), ("signal_accounts.csv",stats),
                        ("signal_loss_intervals.csv",quarter_intervals(predictions)),
                        ("signal_far5.csv",far5),
                        ("signal_thresholds.csv",thresholds), ("signal_daily.csv.gz",histories)]:
        table.to_csv(output/name, index=False)
    print(f"Python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}, NumPy {np.__version__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    run(project, args.output or project/"results")
