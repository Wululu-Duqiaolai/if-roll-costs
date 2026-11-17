"""用四天合成行情离线核对一手 IF 多头的移仓与持仓盈亏。"""
from io import StringIO

import numpy as np
import pandas as pd

from if_accounts import Fees, contract_path, simulate


QUOTES = """date,contract,open,close,settle,volume,oi
2025-06-11,IF2506,4000,4000,4000,1000,2000
2025-06-11,IF2507,3960,3960,3960,800,1500
2025-06-12,IF2506,4000,4005,4004,1100,2100
2025-06-12,IF2507,3960,3965,3964,850,1550
2025-06-13,IF2506,4010,4010,4010,1200,2200
2025-06-13,IF2507,3970,3972,3971,900,1600
2025-06-16,IF2506,4014,4014,4014,1000,2000
2025-06-16,IF2507,3974,3974,3974,950,1650
"""


def main():
    """以可手算的两段持仓盈亏和三条交易腿检验正式账户函数。"""
    panel = pd.read_csv(StringIO(QUOTES), parse_dates=["date"]).set_index(["date", "contract"])
    # 固定目标敞口，使一亿元账户只持一手，单独观察换月的影响
    path = contract_path(panel, "month").assign(target=0.02)
    ledger, orders = simulate(path, panel, "month", "daily", Fees("base", 1, 1))

    np.testing.assert_array_equal(ledger.contract, ["IF2506", "IF2507"])
    np.testing.assert_array_equal(orders.leg, ["resize", "close_old", "open_new"])
    np.testing.assert_array_equal(orders.signed_qty, [1, -1, 1])
    np.testing.assert_allclose(ledger.gross_return * ledger.nav_start, [3000, 1200], rtol=0, atol=1e-8)
    np.testing.assert_allclose(ledger.fee, [143.8, 287.186], rtol=0, atol=1e-8)
    np.testing.assert_allclose(ledger.nav_end.iloc[-1], 100_003_769.014, rtol=0, atol=1e-7)
    print("PASS: roll fees = 287.186 CNY; net P&L = 3,769.014 CNY")


if __name__ == "__main__":
    main()
