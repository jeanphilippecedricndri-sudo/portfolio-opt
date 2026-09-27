import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from portopt import ERC, HRP, EqualWeight, plot, run_backtests  # noqa: E402
from portopt.data import SIM_ASSETS, simulate_returns  # noqa: E402


def test_simulate_returns_shape_and_moments():
    r = simulate_returns(n_obs=2000)
    assert list(r.columns) == SIM_ASSETS and len(r) == 2000
    vol = r.std() * np.sqrt(252)
    assert ((vol > 0.03) & (vol < 0.6)).all()
    c = r.corr()
    assert c.loc["SPY", "IWM"] > 0.6 and c.loc["SPY", "TLT"] < 0


def test_backtester_risk_contributions(returns):
    rets, bts = run_backtests(returns, {"ERC": ERC()}, 250, 63)
    rc = bts["ERC"].risk_contributions()
    assert np.allclose(rc.sum(axis=1), 1.0)
    assert np.allclose(rc, 1 / returns.shape[1], atol=1e-6)  # same cov as the estimator


def test_colors_fixed_by_entity():
    a = plot.colors_for(["A", "B", "C"])
    b = plot.colors_for(["A", "B", "C", "D"])
    assert all(a[k] == b[k] for k in a)
    assert plot.colors_for([str(i) for i in range(10)])["9"] == plot.MUTED  # no 9th hue


def test_save_report(tmp_path, returns):
    rets, bts = run_backtests(returns, {"1/N": EqualWeight(), "HRP": HRP()}, 250, 63)
    paths = plot.save_report(rets, bts, tmp_path)
    assert len(paths) == 9 and all(p.stat().st_size > 10_000 for p in paths)
