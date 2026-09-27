"""Compare 1/N, InvVol, Risk Parity, ERC and HRP on a multi-asset ETF universe.

    python examples/run_backtest.py                 # Yahoo Finance data
    python examples/run_backtest.py --synthetic     # offline, simulated data (portopt.data.simulate_returns)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portopt import ERC, HRP, EqualWeight, InverseVolatility, RiskParity, load_returns, run_backtests, summary
from portopt.data import SIM_ASSETS, simulate_returns

UNIVERSE = {
    "SPY": "US equity", "EFA": "DM ex-US equity", "EEM": "EM equity", "IWM": "US small caps",
    "TLT": "US 20y+ Treasuries", "IEF": "US 7-10y Treasuries", "LQD": "IG credit",
    "GLD": "Gold", "DBC": "Commodities", "VNQ": "US REITs",
}
assert list(UNIVERSE) == SIM_ASSETS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--in-sample", type=int, default=252)
    ap.add_argument("--out-of-sample", type=int, default=21)
    ap.add_argument("--tc-bps", type=float, default=5.0)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if args.synthetic:
        returns = simulate_returns()
    else:
        returns = load_returns(list(UNIVERSE), start=args.start, end=args.end, cache_dir="data_cache")

    budgets = {t: 1.0 for t in UNIVERSE} | {"TLT": 2.0, "IEF": 2.0, "LQD": 2.0}  # tilt risk towards bonds

    estimators = {
        "1/N": EqualWeight(),
        "InvVol": InverseVolatility(),
        "RB bond tilt": RiskParity(budgets=budgets),
        "ERC": ERC(),
        "ERC LW": ERC(cov_method="ledoit_wolf"),
        "HRP": HRP(),
    }

    rets, bts = run_backtests(
        returns, estimators, args.in_sample, args.out_of_sample, transaction_cost_bps=args.tc_bps
    )
    turnover = {k: bt.annualized_turnover() for k, bt in bts.items()}
    table = summary(rets, turnover=turnover)

    pd.set_option("display.width", 200)
    print(f"\nOOS period: {rets.index[0].date()} -> {rets.index[-1].date()} ({len(rets)} obs)\n")
    print(table.round(3).to_string())

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    table.to_csv(out / "summary.csv")
    rets.to_csv(out / "oos_returns.csv")
    try:
        from portopt import plot
    except ImportError:
        print("matplotlib not installed: skipping charts")
    else:
        paths = plot.save_report(rets, bts, out / "charts")
        print(f"\n{len(paths)} charts written to {(out / 'charts').resolve()}/")


if __name__ == "__main__":
    main()
