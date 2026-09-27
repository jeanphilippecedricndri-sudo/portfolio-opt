"""Metric module: performance and risk metrics of return series.

All functions accept a pd.Series (one strategy) or a pd.DataFrame (one column per strategy)
of periodic simple returns and return a float or a pd.Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PPY = 252  # periods per year (daily)


def _clean(r):
    return r.dropna(how="all") if isinstance(r, pd.DataFrame) else r.dropna()


def _per_column(func, r, *args, **kwargs):
    if isinstance(r, pd.DataFrame):
        return r.apply(lambda s: func(s.dropna(), *args, **kwargs))
    return func(r.dropna(), *args, **kwargs)


# ---------------------------------------------------------------- performance
def total_return(r):
    return (1.0 + _clean(r)).prod() - 1.0


def cagr(r, periods_per_year: int = PPY):
    """Geometric annualized return: (prod(1+r))^(ppy/T) - 1."""
    r = _clean(r)
    return (1.0 + r).prod() ** (periods_per_year / r.count()) - 1.0


def annualized_mean(r, periods_per_year: int = PPY):
    """Arithmetic annualized mean. CAGR ~ mean - sigma^2/2 (volatility drag)."""
    return _clean(r).mean() * periods_per_year


def annualized_volatility(r, periods_per_year: int = PPY):
    return _clean(r).std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(r, rf: float = 0.0, periods_per_year: int = PPY):
    """Annualized Sharpe = sqrt(ppy) * mean(r - rf_p) / std(r), rf annual (converted geometrically)."""
    r = _clean(r)
    excess = r - ((1.0 + rf) ** (1.0 / periods_per_year) - 1.0)
    return np.sqrt(periods_per_year) * excess.mean() / r.std(ddof=1)


def sortino_ratio(r, target: float = 0.0, periods_per_year: int = PPY):
    """sqrt(ppy) * mean(r - target) / DD, DD = sqrt(mean(min(r - target, 0)^2)) (full-sample denominator)."""
    r = _clean(r)
    downside = np.sqrt(((r - target).clip(upper=0.0) ** 2).mean())
    return np.sqrt(periods_per_year) * (r - target).mean() / downside


# ---------------------------------------------------------------- drawdowns
def wealth_index(r, start: float = 1.0):
    return start * (1.0 + _clean(r).fillna(0.0)).cumprod()


def drawdown_series(r):
    """DD_t = W_t / max_{s<=t} W_s - 1, with the initial wealth 1 counted as a peak."""
    w = wealth_index(r)
    peak = np.maximum(w.cummax(), 1.0)
    return w / peak - 1.0


def max_drawdown(r):
    """Most negative drawdown (a negative number)."""
    return drawdown_series(r).min()


def max_drawdown_duration(r):
    """Longest underwater stretch, in periods (from a peak until the next new high)."""

    def _dur(s):
        under = (drawdown_series(s) < 0).to_numpy()
        best = cur = 0
        for u in under:
            cur = cur + 1 if u else 0
            best = max(best, cur)
        return best

    return _per_column(_dur, r)


def calmar_ratio(r, periods_per_year: int = PPY):
    return cagr(r, periods_per_year) / -max_drawdown(r)


# ---------------------------------------------------------------- tail risk
def value_at_risk(r, alpha: float = 0.05):
    """Historical VaR at level alpha, reported as a positive loss: -q_alpha(r)."""
    return -_clean(r).quantile(alpha)


def conditional_value_at_risk(r, alpha: float = 0.05):
    """Historical CVaR / Expected Shortfall: -E[r | r <= q_alpha]. Coherent, VaR is not."""

    def _cvar(s):
        q = s.quantile(alpha)
        return -s[s <= q].mean()

    return _per_column(_cvar, r)


def skewness(r):
    return _clean(r).skew()


def excess_kurtosis(r):
    return _clean(r).kurt()


def hit_ratio(r):
    """Share of periods with a strictly positive return."""
    return _per_column(lambda s: (s > 0).mean(), r)


# ---------------------------------------------------------------- statistical significance
def probabilistic_sharpe_ratio(r, sr_benchmark: float = 0.0, periods_per_year: int = PPY):
    """PSR (Bailey & Lopez de Prado, 2012): P(true SR > sr_benchmark) given non-normal returns.

        PSR = Phi( (SR_hat - SR*) sqrt(T - 1) / sqrt(1 - g3 SR_hat + (g4 - 1)/4 SR_hat^2) )

    SR_hat, SR* are PER-PERIOD (non-annualized); sr_benchmark is given annualized.
    g3 = skewness, g4 = (non-excess) kurtosis.
    """

    def _psr(s):
        t = s.count()
        sr = s.mean() / s.std(ddof=1)
        sr_star = sr_benchmark / np.sqrt(periods_per_year)
        g3 = stats.skew(s, bias=False)
        g4 = stats.kurtosis(s, fisher=False, bias=False)
        denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2, 1e-12))
        return stats.norm.cdf((sr - sr_star) * np.sqrt(t - 1) / denom)

    return _per_column(_psr, r)


# ---------------------------------------------------------------- summary
def summary(
    returns: pd.Series | pd.DataFrame,
    rf: float = 0.0,
    periods_per_year: int = PPY,
    alpha: float = 0.05,
    turnover: dict[str, float] | pd.Series | None = None,
) -> pd.DataFrame:
    """One table, metrics in rows and strategies in columns."""
    r = returns.to_frame() if isinstance(returns, pd.Series) else returns
    table = {
        "CAGR": cagr(r, periods_per_year),
        "Ann. mean": annualized_mean(r, periods_per_year),
        "Ann. vol": annualized_volatility(r, periods_per_year),
        "Sharpe": sharpe_ratio(r, rf, periods_per_year),
        "Sortino": sortino_ratio(r, 0.0, periods_per_year),
        "Calmar": calmar_ratio(r, periods_per_year),
        "Max DD": max_drawdown(r),
        "Max DD duration": max_drawdown_duration(r),
        f"VaR {1 - alpha:.0%}": value_at_risk(r, alpha),
        f"CVaR {1 - alpha:.0%}": conditional_value_at_risk(r, alpha),
        "Skew": skewness(r),
        "Excess kurt": excess_kurtosis(r),
        "Hit ratio": hit_ratio(r),
        "PSR(SR>0)": probabilistic_sharpe_ratio(r, 0.0, periods_per_year),
    }
    out = pd.DataFrame(table).T
    if turnover is not None:
        out.loc["Ann. turnover"] = pd.Series(turnover).reindex(out.columns)
    return out
