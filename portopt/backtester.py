"""Backtester module: walk-forward backtest of an estimator.

Timeline for one rebalancing (rolling window, in_sample=L, out_of_sample=H):

    |<------- in-sample: fit on r[t-L : t) ------->|<--- hold: earn r[t : t+H) --->|
                                                   t = rebalancing date

The weights are computed with returns up to t-1 (close) and applied from r_t onwards, so
there is no look-ahead. Then t <- t + H and the window rolls (or expands).
"""

from __future__ import annotations

import copy
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .estimator import BaseEstimator


class Backtester:
    """Walk-forward backtest.

    Parameters
    ----------
    returns : (T x N) simple returns. NaN allowed (assets not yet listed / delisted).
    estimator : any ``BaseEstimator``. It is deep-copied at every rebalancing.
    in_sample : number of observations used to fit the estimator.
    out_of_sample : holding period (number of observations) between two rebalancings.
    window : "rolling" (fixed length L) or "expanding" (from the first observation).
    transaction_cost_bps : proportional cost applied to one-way turnover sum|w_target - w_drifted|.
    drift : True -> buy-and-hold inside the holding period (weights drift with prices, realistic).
            False -> constant weights every day (implicit daily rebalancing, intra-period
            turnover is NOT charged).
    min_assets : skip allocation (stay in the previous portfolio) if fewer assets are eligible.

    Universe rule: at each date t, an asset is eligible iff it has no NaN in the in-sample
    window. During the holding period a NaN return is treated as 0 (asset price frozen).
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        estimator: BaseEstimator,
        in_sample: int,
        out_of_sample: int,
        window: Literal["rolling", "expanding"] = "rolling",
        transaction_cost_bps: float = 0.0,
        drift: bool = True,
        min_assets: int = 1,
    ):
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a DataFrame.")
        if in_sample < 2 or out_of_sample < 1:
            raise ValueError("need in_sample >= 2 and out_of_sample >= 1.")
        if in_sample >= len(returns):
            raise ValueError(f"in_sample ({in_sample}) >= number of observations ({len(returns)}).")
        if window not in ("rolling", "expanding"):
            raise ValueError("window must be 'rolling' or 'expanding'.")
        self.returns = returns.sort_index()
        self.estimator = estimator
        self.in_sample = int(in_sample)
        self.out_of_sample = int(out_of_sample)
        self.window = window
        self.tc = transaction_cost_bps / 1e4
        self.drift = drift
        self.min_assets = min_assets

        self.returns_: pd.Series | None = None
        self.gross_returns_: pd.Series | None = None
        self.weights_: pd.DataFrame | None = None
        self.turnover_: pd.Series | None = None

    @property
    def name(self) -> str:
        return getattr(self.estimator, "name", type(self.estimator).__name__)

    def run(self) -> pd.Series:
        """Run the backtest and return the out-of-sample (net of costs) portfolio returns."""
        r = self.returns
        cols = r.columns
        values = r.to_numpy(dtype=float)
        T, N = values.shape

        gross = np.full(T, np.nan)
        net = np.full(T, np.nan)
        w_held = np.zeros(N)  # start fully in cash
        rebal_dates, targets, turnovers = [], [], []

        t = self.in_sample
        while t < T:
            start = 0 if self.window == "expanding" else t - self.in_sample
            window = r.iloc[start:t]
            eligible = window.columns[window.notna().all().to_numpy()]

            if len(eligible) >= self.min_assets:
                est = copy.deepcopy(self.estimator).fit(window[eligible])
                target = est.weights_.reindex(cols, fill_value=0.0).to_numpy()
            else:  # not enough assets: keep drifted weights (or cash at the start)
                target = w_held.copy()

            turnover = np.abs(target - w_held).sum()
            end = min(t + self.out_of_sample, T)
            oos = np.nan_to_num(values[t:end], nan=0.0)

            if self.drift:
                growth = np.cumprod(1.0 + oos, axis=0)          # per-asset wealth relative to t-1
                cash = 1.0 - target.sum()                        # 0 unless no allocation yet
                v = growth @ target + cash                       # portfolio wealth path
                v_prev = np.concatenate(([1.0], v[:-1]))
                rp = v / v_prev - 1.0
                w_end = target * growth[-1] / v[-1]
            else:
                rp = oos @ target
                w_end = target.copy()

            gross[t:end] = rp
            net[t:end] = rp
            # cost paid at the rebalancing: wealth multiplied by (1 - c * turnover)
            net[t] = (1.0 - self.tc * turnover) * (1.0 + rp[0]) - 1.0

            rebal_dates.append(r.index[t])
            targets.append(target)
            turnovers.append(turnover)
            w_held = w_end
            t = end

        idx = r.index[self.in_sample:]
        self.gross_returns_ = pd.Series(gross[self.in_sample:], index=idx, name=self.name)
        self.returns_ = pd.Series(net[self.in_sample:], index=idx, name=self.name)
        self.weights_ = pd.DataFrame(targets, index=pd.DatetimeIndex(rebal_dates), columns=cols)
        self.turnover_ = pd.Series(turnovers, index=self.weights_.index, name=self.name)
        return self.returns_

    def annualized_turnover(self, periods_per_year: int = 252) -> float:
        """Average one-way turnover per year, excluding the initial build-up from cash."""
        if self.turnover_ is None:
            raise RuntimeError("call run() first.")
        years = len(self.returns_) / periods_per_year
        return float(self.turnover_.iloc[1:].sum() / years)

    def risk_contributions(self, cov_method="sample") -> pd.DataFrame:
        """Ex-ante relative risk contributions of the target weights at each rebalancing,
        with the covariance estimated on the same in-sample window the estimator saw."""
        from .estimator import estimate_cov, risk_contributions

        if self.weights_ is None:
            raise RuntimeError("call run() first.")
        r = self.returns
        pos = r.index.get_indexer(self.weights_.index)
        rows = []
        for p, (date, w) in zip(pos, self.weights_.iterrows()):
            start = 0 if self.window == "expanding" else p - self.in_sample
            live = w[w > 0].index
            window = r.iloc[start:p][live]
            rc = pd.Series(0.0, index=r.columns)
            if len(live):
                rc[live] = risk_contributions(w[live].to_numpy(), estimate_cov(window, cov_method))
            rows.append(rc.rename(date))
        return pd.DataFrame(rows)


def run_backtests(
    returns: pd.DataFrame,
    estimators: Mapping[str, BaseEstimator] | Sequence[BaseEstimator],
    in_sample: int,
    out_of_sample: int,
    **kwargs,
) -> tuple[pd.DataFrame, dict[str, Backtester]]:
    """Backtest several estimators on the same grid. Returns (returns DataFrame, backtesters)."""
    if not isinstance(estimators, Mapping):
        estimators = list(estimators)
        names = [e.name for e in estimators]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(
                f"Several estimators share the name {dup}: pass a dict {{label: estimator}} instead of a list."
            )
        estimators = dict(zip(names, estimators))
    bts = {
        name: Backtester(returns, est, in_sample, out_of_sample, **kwargs) for name, est in estimators.items()
    }
    rets = pd.DataFrame({name: bt.run() for name, bt in bts.items()})
    return rets, bts
