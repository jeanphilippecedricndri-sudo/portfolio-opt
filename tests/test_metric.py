import numpy as np
import pandas as pd
import pytest

from portopt import metric as M


@pytest.fixture
def toy():
    idx = pd.bdate_range("2020-01-01", periods=5)
    return pd.Series([0.10, -0.20, 0.05, -0.10, 0.30], index=idx)


def test_total_return_and_cagr(toy):
    tr = 1.1 * 0.8 * 1.05 * 0.9 * 1.3 - 1
    assert np.isclose(M.total_return(toy), tr)
    assert np.isclose(M.cagr(toy, periods_per_year=5), tr)  # 5 obs = 1 year


def test_max_drawdown(toy):
    w = np.cumprod(1 + toy.to_numpy())  # 1.1, .88, .924, .8316, 1.081
    assert np.isclose(M.max_drawdown(toy), 0.8316 / 1.1 - 1)
    assert M.max_drawdown_duration(toy) == 4  # 1.081 < 1.1: still underwater at the end


def test_drawdown_counts_initial_capital():
    r = pd.Series([-0.1, 0.05], index=pd.bdate_range("2020-01-01", periods=2))
    assert np.isclose(M.max_drawdown(r), -0.1)


def test_var_cvar():
    r = pd.Series(np.linspace(-0.10, 0.09, 20))
    assert M.value_at_risk(r, 0.05) > 0
    assert M.conditional_value_at_risk(r, 0.05) >= M.value_at_risk(r, 0.05)


def test_sharpe_scaling():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 5000))
    assert np.isclose(M.sharpe_ratio(r), np.sqrt(252) * r.mean() / r.std())


def test_psr_bounds_and_monotonicity():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.001, 0.01, 2000))
    bad = pd.Series(rng.normal(-0.001, 0.01, 2000))
    assert M.probabilistic_sharpe_ratio(good) > 0.95
    assert M.probabilistic_sharpe_ratio(bad) < 0.05


def test_summary_dataframe(returns):
    s = M.summary(returns, turnover={"EQ1": 1.0})
    assert list(s.columns) == list(returns.columns)
    assert "Sharpe" in s.index and "Ann. turnover" in s.index
    assert s.loc["Max DD"].le(0).all()
