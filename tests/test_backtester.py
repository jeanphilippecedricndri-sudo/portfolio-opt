import numpy as np
import pandas as pd
import pytest

from portopt.backtester import Backtester, run_backtests
from portopt.estimator import ERC, HRP, BaseEstimator, EqualWeight


class SpyEstimator(BaseEstimator):
    """Records the last date seen at each fit, to test for look-ahead."""

    seen: list = []

    def _compute_weights(self, returns):
        SpyEstimator.seen.append(returns.index[-1])
        return np.ones(returns.shape[1])


def test_no_lookahead(returns):
    SpyEstimator.seen = []
    bt = Backtester(returns, SpyEstimator(), in_sample=250, out_of_sample=21)
    bt.run()
    for last_seen, rebal in zip(SpyEstimator.seen, bt.weights_.index):
        assert last_seen < rebal
    assert len(SpyEstimator.seen) == len(bt.weights_)


def test_output_alignment(returns):
    bt = Backtester(returns, EqualWeight(), in_sample=250, out_of_sample=21)
    out = bt.run()
    assert out.index[0] == returns.index[250]
    assert out.index[-1] == returns.index[-1]
    assert out.notna().all()


def test_drift_matches_manual_buy_and_hold(returns):
    L, H = 100, 20
    bt = Backtester(returns, EqualWeight(), in_sample=L, out_of_sample=H, drift=True)
    out = bt.run()
    block = returns.iloc[L : L + H]
    wealth = ((1 + block).cumprod() / 6).sum(axis=1)
    manual = wealth.pct_change()
    manual.iloc[0] = wealth.iloc[0] - 1
    assert np.allclose(out.iloc[:H], manual)


def test_no_drift_is_constant_mix(returns):
    bt = Backtester(returns, EqualWeight(), in_sample=100, out_of_sample=20, drift=False)
    out = bt.run()
    assert np.allclose(out, returns.iloc[100:].mean(axis=1))


def test_transaction_costs(returns):
    kw = dict(in_sample=250, out_of_sample=21)
    gross = Backtester(returns, ERC(), **kw).run()
    bt = Backtester(returns, ERC(), transaction_cost_bps=10, **kw)
    net = bt.run()
    assert (net <= gross + 1e-15).all()
    first = bt.turnover_.iloc[0]  # 1.0: build-up from cash
    assert np.isclose(first, 1.0)
    assert np.isclose((1 + net.iloc[0]), (1 - 10e-4 * first) * (1 + gross.iloc[0]))


def test_staggered_listing(returns):
    r = returns.copy()
    r.iloc[:400, 0] = np.nan  # EQ1 listed later
    bt = Backtester(r, HRP(), in_sample=250, out_of_sample=21)
    bt.run()
    early = bt.weights_.loc[bt.weights_.index < r.index[400]]
    late = bt.weights_.loc[bt.weights_.index >= r.index[650]]
    assert (early["EQ1"] == 0).all()
    assert (late["EQ1"] > 0).all()
    assert np.allclose(bt.weights_.sum(axis=1), 1.0)


def test_expanding_window(returns):
    SpyEstimator.seen = []

    class Len(BaseEstimator):
        lens = []

        def _compute_weights(self, r):
            Len.lens.append(len(r))
            return np.ones(r.shape[1])

    Backtester(returns, Len(), in_sample=250, out_of_sample=100, window="expanding").run()
    assert Len.lens[0] == 250 and Len.lens[1] == 350


def test_run_backtests(returns):
    rets, bts = run_backtests(returns, [EqualWeight(), ERC()], 250, 21, transaction_cost_bps=5)
    assert list(rets.columns) == ["EqualWeight", "ERC"]
    assert bts["ERC"].annualized_turnover() > 0


def test_run_backtests_duplicate_names_raise(returns):
    with pytest.raises(ValueError, match="dict"):
        run_backtests(returns, [ERC(), ERC(cov_method="ledoit_wolf")], 250, 21)
    rets, _ = run_backtests(returns, {"ERC": ERC(), "ERC LW": ERC(cov_method="ledoit_wolf")}, 250, 21)
    assert list(rets.columns) == ["ERC", "ERC LW"]


def test_bad_params(returns):
    with pytest.raises(ValueError):
        Backtester(returns, ERC(), in_sample=len(returns), out_of_sample=5)
