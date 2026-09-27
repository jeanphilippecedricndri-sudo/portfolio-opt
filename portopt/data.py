"""Data module: download prices from Yahoo Finance and turn them into returns.

Conventions
-----------
* Prices are a (T x N) DataFrame indexed by a tz-naive DatetimeIndex, one column per ticker.
* ``r_t`` at date ``t`` is the return from close ``t-1`` to close ``t``. A weight decided with
  information up to close ``t-1`` earns ``r_t``: the backtester relies on this alignment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

ReturnMethod = Literal["simple", "log"]


def download_prices(
    tickers: str | Iterable[str],
    start: str | None = None,
    end: str | None = None,
    field: str = "Close",
    auto_adjust: bool = True,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download daily prices from Yahoo Finance.

    Parameters
    ----------
    tickers : ticker or list of tickers.
    start, end : dates "YYYY-MM-DD" (end exclusive, yfinance convention).
    field : OHLCV field. With ``auto_adjust=True`` "Close" is already adjusted for splits and
        dividends, i.e. it yields total returns. Never compute returns on unadjusted closes.
    cache_dir : if given, results are cached as CSV so reruns are offline and reproducible.
    """
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    if not tickers:
        raise ValueError("`tickers` is empty.")

    cache_file = None
    if cache_dir is not None:
        key = f"{','.join(tickers)}|{start}|{end}|{field}|{auto_adjust}"
        cache_file = Path(cache_dir) / f"prices_{hashlib.md5(key.encode()).hexdigest()[:12]}.csv"
        if cache_file.exists():
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    import yfinance as yf  # lazy import: the rest of the package works without network

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned by Yahoo Finance for {tickers}.")

    if isinstance(raw.columns, pd.MultiIndex):
        if field not in raw.columns.get_level_values(0):
            raise KeyError(f"Field '{field}' not in {sorted(set(raw.columns.get_level_values(0)))}")
        prices = raw[field]
    else:
        prices = raw[[field]].rename(columns={field: tickers[0]})

    prices = prices.reindex(columns=tickers)
    missing = prices.columns[prices.isna().all()].tolist()
    if missing:
        raise ValueError(f"No data for tickers: {missing}")

    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index)).tz_localize(None)
    prices = prices.sort_index().dropna(how="all")
    prices.index.name = "date"
    prices.columns.name = None

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_file)
    return prices


def clean_prices(
    prices: pd.DataFrame,
    ffill_limit: int | None = 5,
    min_history: int | None = None,
) -> pd.DataFrame:
    """Light cleaning with no look-ahead.

    * forward-fill short gaps (exchange holidays when mixing markets), at most ``ffill_limit`` days.
      Never back-fill: it copies future prices into the past.
    * drop non-positive prices (bad ticks).
    * optionally drop assets with fewer than ``min_history`` valid observations.

    Leading NaNs (asset not yet listed) are kept: the backtester only invests in assets with a
    full in-sample window, which handles staggered listings without survivorship tricks.
    """
    out = prices.where(prices > 0)
    out = out.ffill(limit=ffill_limit)
    if min_history is not None:
        keep = out.notna().sum() >= min_history
        out = out.loc[:, keep]
    return out


def compute_returns(
    prices: pd.DataFrame | pd.Series,
    method: ReturnMethod = "simple",
    period: int = 1,
) -> pd.DataFrame | pd.Series:
    """Compute returns from prices.

    simple : R_t = P_t / P_{t-period} - 1, aggregates across assets (R_p = w' R).
    log    : r_t = ln(P_t / P_{t-period}), aggregates across time (sum), NOT across assets.

    Portfolio construction and backtests must use simple returns.
    """
    if method == "simple":
        rets = prices.pct_change(periods=period, fill_method=None)
    elif method == "log":
        rets = np.log(prices).diff(periods=period)
    else:
        raise ValueError("method must be 'simple' or 'log'")
    return rets.iloc[period:].dropna(how="all")


def resample_returns(
    returns: pd.DataFrame | pd.Series,
    rule: str = "W-FRI",
    method: ReturnMethod = "simple",
) -> pd.DataFrame | pd.Series:
    """Aggregate returns to a lower frequency (e.g. 'W-FRI', 'ME').

    simple: compounding, prod(1 + R) - 1.  log: sum.
    ``min_count=1`` keeps NaN for periods where an asset has no data at all.
    """
    if method == "simple":
        return (1.0 + returns).resample(rule).prod(min_count=1) - 1.0
    if method == "log":
        return returns.resample(rule).sum(min_count=1)
    raise ValueError("method must be 'simple' or 'log'")


SIM_ASSETS = ["SPY", "EFA", "EEM", "IWM", "TLT", "IEF", "LQD", "GLD", "DBC", "VNQ"]


def simulate_returns(
    n_obs: int = 252 * 15,
    start: str = "2010-01-01",
    seed: int = 15,
) -> pd.DataFrame:
    """Realistic synthetic daily returns for a 10-ETF multi-asset universe (offline work, docs, tests).

    * hand-calibrated vols / correlations (equities ~0.8 together, equity-bond ~ -0.3, ...),
    * 2-state Markov regime: calm (x0.8 vol) / stress (x2 vol, equity drift negative),
    * Student-t(5) innovations for fat tails.
    """
    rng = np.random.default_rng(seed)
    vols = np.array([0.16, 0.17, 0.22, 0.21, 0.14, 0.065, 0.075, 0.15, 0.19, 0.22])
    mus = np.array([0.10, 0.07, 0.08, 0.09, 0.035, 0.025, 0.035, 0.05, 0.02, 0.08])
    c = np.array([
        # SPY   EFA   EEM   IWM   TLT   IEF   LQD   GLD   DBC   VNQ
        [1.00, 0.85, 0.75, 0.88, -0.30, -0.28, 0.20, 0.05, 0.40, 0.72],  # SPY
        [0.85, 1.00, 0.82, 0.78, -0.25, -0.24, 0.22, 0.12, 0.45, 0.65],  # EFA
        [0.75, 0.82, 1.00, 0.70, -0.20, -0.18, 0.25, 0.20, 0.50, 0.58],  # EEM
        [0.88, 0.78, 0.70, 1.00, -0.30, -0.28, 0.18, 0.03, 0.38, 0.75],  # IWM
        [-0.30, -0.25, -0.20, -0.30, 1.00, 0.92, 0.65, 0.25, -0.20, 0.05],  # TLT
        [-0.28, -0.24, -0.18, -0.28, 0.92, 1.00, 0.72, 0.28, -0.18, 0.05],  # IEF
        [0.20, 0.22, 0.25, 0.18, 0.65, 0.72, 1.00, 0.25, 0.05, 0.35],  # LQD
        [0.05, 0.12, 0.20, 0.03, 0.25, 0.28, 0.25, 1.00, 0.35, 0.10],  # GLD
        [0.40, 0.45, 0.50, 0.38, -0.20, -0.18, 0.05, 0.35, 1.00, 0.30],  # DBC
        [0.72, 0.65, 0.58, 0.75, 0.05, 0.05, 0.35, 0.10, 0.30, 1.00],  # VNQ
    ])
    ev, V = np.linalg.eigh(c)  # project to nearest PSD correlation, just in case
    c = V @ np.diag(np.clip(ev, 1e-4, None)) @ V.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    L = np.linalg.cholesky(c)

    # regimes: P(calm->stress)=1/250, P(stress->calm)=1/60
    state = np.zeros(n_obs, dtype=int)
    for t in range(1, n_obs):
        p = 1 / 250 if state[t - 1] == 0 else 1 - 1 / 60
        state[t] = rng.random() < p
    scale = np.where(state == 1, 2.0, 0.8)
    # risky assets lose 2x their drift in stress, safe havens (TLT, IEF, GLD) rally;
    # calm drift is set so that the long-run average drift equals `mus`
    p_s = (1 / 250) / (1 / 250 + 1 / 60)
    stress_mu = np.where(vols > 0.15, -2.0 * mus, mus)
    stress_mu[[4, 5, 7]] = [0.12, 0.06, 0.12]
    calm_mu = (mus - p_s * stress_mu) / (1 - p_s)
    drift = np.where(state[:, None] == 1, stress_mu, calm_mu)

    nu = 5
    z = rng.standard_normal((n_obs, len(vols))) @ L.T
    z *= np.sqrt((nu - 2) / rng.chisquare(nu, size=(n_obs, 1)))  # unit-variance multivariate t
    x = drift / 252 + z * vols / np.sqrt(252) * scale[:, None]
    idx = pd.bdate_range(start, periods=n_obs)
    return pd.DataFrame(x, index=idx, columns=SIM_ASSETS)


def load_returns(
    tickers: str | Iterable[str],
    start: str | None = None,
    end: str | None = None,
    method: ReturnMethod = "simple",
    cache_dir: str | Path | None = None,
    ffill_limit: int | None = 5,
) -> pd.DataFrame:
    """One-liner: download -> clean -> returns."""
    prices = download_prices(tickers, start=start, end=end, cache_dir=cache_dir)
    prices = clean_prices(prices, ffill_limit=ffill_limit)
    return compute_returns(prices, method=method)
