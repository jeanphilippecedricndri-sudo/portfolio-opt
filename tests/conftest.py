"""Shared fixtures: synthetic multi-asset returns with a known block correlation structure."""

import numpy as np
import pandas as pd
import pytest


def make_returns(T=1500, seed=0):
    rng = np.random.default_rng(seed)
    # 2 blocks of 3 correlated assets (equities-like / bonds-like) + different vols
    vols = np.array([0.20, 0.25, 0.18, 0.06, 0.05, 0.08]) / np.sqrt(252)
    corr = np.eye(6)
    corr[:3, :3] = 0.7
    corr[3:, 3:] = 0.6
    corr[:3, 3:] = corr[3:, :3] = -0.1
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    mu = np.full(6, 0.05 / 252)
    x = rng.multivariate_normal(mu, cov, size=T)
    idx = pd.bdate_range("2015-01-01", periods=T)
    return pd.DataFrame(x, index=idx, columns=["EQ1", "EQ2", "EQ3", "BD1", "BD2", "BD3"]), cov


@pytest.fixture
def returns_and_cov():
    return make_returns()


@pytest.fixture
def returns(returns_and_cov):
    return returns_and_cov[0]
