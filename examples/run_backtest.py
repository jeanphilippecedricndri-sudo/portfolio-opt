"""Compare 1/N, InvVol, Risk Parity, ERC and HRP on a multi-asset ETF universe.

    python examples/run_backtest.py                 # Yahoo Finance data
    python examples/run_backtest.py --synthetic     # offline, simulated data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from portopt import ERC, HRP, EqualWeight, InverseVolatility, RiskParity, load_returns, run_backtests, summary
from portopt.metric import drawdown_series, wealth_index

UNIVERSE = {
    "SPY": "US equity", "EFA": "DM ex-US equity", "EEM": "EM equity", "IWM": "US small caps",
    "TLT": "US 20y+ Treasuries", "IEF": "US 7-10y Treasuries", "LQD": "IG credit",
    "GLD": "Gold", "DBC": "Commodities", "VNQ": "US REITs",
}


def synthetic_returns(T: int = 252 * 15, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(UNIVERSE)
    vols = np.array([0.18, 0.19, 0.24, 0.23, 0.15, 0.07, 0.08, 0.16, 0.20, 0.25])
    mus = np.array([0.09, 0.06, 0.07, 0.08, 0.04, 0.03, 0.04, 0.05, 0.02, 0.07])
    k = 3  # 3-factor structure -> realistic block correlations
    B = rng.normal(0, 1, (n, k)) * [1.0, 0.6, 0.4]
    B[4:7, 0] *= -0.3
    corr = B @ B.T + np.diag(np.full(n, 1.0))
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    cov = np.outer(vols, vols) * corr / 252
    x = rng.multivariate_normal(mus / 252, cov, size=T)
    x *= np.sqrt(rng.standard_t(5, size=(T, 1)) ** 2 / 1.67)  # fat tails, common vol shocks
    idx = pd.bdate_range("2010-01-01", periods=T)
    return pd.DataFrame(x, index=idx, columns=list(UNIVERSE))


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
        returns = synthetic_returns()
    else:
        returns = load_returns(list(UNIVERSE), start=args.start, end=args.end, cache_dir="data_cache")

    budgets = {t: 1.0 for t in UNIVERSE} | {"TLT": 2.0, "IEF": 2.0, "LQD": 2.0}  # tilt risk towards bonds

    estimators = {
        "1/N": EqualWeight(),
        "InvVol": InverseVolatility(),
        "RiskParity (bond tilt)": RiskParity(budgets=budgets),
        "ERC": ERC(),
        "ERC (Ledoit-Wolf)": ERC(cov_method="ledoit_wolf"),
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
    bts["ERC"].weights_.to_csv(out / "erc_weights.csv")
    _plot(rets, bts, out / "backtest.png")
    print(f"\nSaved to {out.resolve()}/")


def _plot(rets: pd.DataFrame, bts: dict, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [3, 1.5, 2]})
    wealth_index(rets).plot(ax=axes[0], lw=1.2, logy=True)
    axes[0].set_title("Wealth (log scale), net of costs")
    drawdown_series(rets).plot(ax=axes[1], lw=0.9, legend=False)
    axes[1].set_title("Drawdown")
    bts["ERC"].weights_.plot.area(ax=axes[2], lw=0, alpha=0.85)
    axes[2].set_title("ERC target weights at each rebalancing")
    axes[2].legend(ncol=5, fontsize=7, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
