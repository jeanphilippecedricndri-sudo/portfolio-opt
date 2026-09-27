import numpy as np
import pandas as pd
import pytest

from portopt import estimator as E
from portopt.estimator import ERC, HRP, EqualWeight, InverseVolatility, RiskParity, risk_contributions

ALL = [EqualWeight(), InverseVolatility(), RiskParity(), ERC(), HRP(), HRP(split="dendrogram")]


@pytest.mark.parametrize("est", ALL, ids=lambda e: repr(e))
def test_weights_are_long_only_and_sum_to_one(est, returns):
    w = est.fit(returns).weights_
    assert list(w.index) == list(returns.columns)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_equal_weight(returns):
    assert np.allclose(EqualWeight().fit(returns).weights_, 1 / 6)


def test_inverse_vol_proportional(returns):
    w = InverseVolatility().fit(returns).weights_
    vol = returns.std()
    assert np.allclose(w * vol, (w * vol).iloc[0])


def test_erc_equal_risk_contributions(returns):
    est = ERC().fit(returns)
    rc = risk_contributions(est.weights_, est.cov_)
    assert np.allclose(rc, 1 / 6, atol=1e-8)


def test_risk_budgeting_matches_budgets(returns):
    b = {"EQ1": 0.3, "EQ2": 0.1, "EQ3": 0.1, "BD1": 0.2, "BD2": 0.2, "BD3": 0.1}
    est = RiskParity(budgets=b).fit(returns)
    rc = risk_contributions(est.weights_, est.cov_)
    assert np.allclose(rc, pd.Series(b)[returns.columns], atol=1e-8)


def test_erc_equals_invvol_when_correlations_equal():
    rng = np.random.default_rng(1)
    vols = np.array([0.1, 0.2, 0.3, 0.4])
    corr = np.full((4, 4), 0.3) + 0.7 * np.eye(4)
    cov = np.outer(vols, vols) * corr
    x = pd.DataFrame(rng.multivariate_normal(np.zeros(4), cov, 50))
    fixed = lambda _: cov  # noqa: E731 - use the true covariance
    w_erc = ERC(cov_method=fixed).fit(x).weights_
    w_iv = InverseVolatility(cov_method=fixed).fit(x).weights_
    assert np.allclose(w_erc, w_iv, atol=1e-8)


def test_erc_vol_between_minvar_and_equal_weight(returns):
    cov = returns.cov().to_numpy()
    ones = np.ones(6)
    w_mv = np.linalg.solve(cov, ones)
    w_mv /= w_mv.sum()  # unconstrained MV; long-only here since blocks are well behaved
    vol = lambda w: np.sqrt(w @ cov @ w)  # noqa: E731
    w_erc = ERC().fit(returns).weights_.to_numpy()
    assert vol(w_mv) <= vol(w_erc) <= vol(ones / 6)


def test_hrp_clusters_blocks(returns):
    est = HRP().fit(returns)
    order = est.order_
    # the two correlation blocks must be contiguous after quasi-diagonalisation
    first_half = set(order[:3])
    assert first_half in ({"EQ1", "EQ2", "EQ3"}, {"BD1", "BD2", "BD3"})


def test_hrp_two_assets_is_inverse_variance():
    rng = np.random.default_rng(2)
    x = pd.DataFrame(rng.normal(0, [0.01, 0.02], size=(500, 2)), columns=["A", "B"])
    w = HRP().fit(x).weights_
    iv = 1 / x.var()
    assert np.allclose(w, iv / iv.sum())


def test_ledoit_wolf_properties(returns):
    cov, delta = E.ledoit_wolf_cov(returns.iloc[:60])
    assert 0.0 <= delta <= 1.0
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    # shrinkage improves conditioning vs sample covariance
    s = np.cov(returns.iloc[:60].to_numpy(), rowvar=False)
    assert np.linalg.cond(cov) <= np.linalg.cond(s)


def test_nan_raises(returns):
    r = returns.copy()
    r.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        ERC().fit(r)
