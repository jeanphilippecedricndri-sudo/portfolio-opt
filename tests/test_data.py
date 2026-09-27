from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portopt import data as D


@pytest.fixture
def prices():
    idx = pd.bdate_range("2020-01-01", periods=6)
    return pd.DataFrame(
        {"A": [100, 110, 99, 99, 108.9, 108.9], "B": [np.nan, np.nan, 50, 55, np.nan, 60.5]},
        index=idx,
    )


def test_simple_returns(prices):
    r = D.compute_returns(prices["A"])
    assert np.allclose(r.to_numpy(), [0.1, -0.1, 0.0, 0.1, 0.0])


def test_log_returns_sum_over_time(prices):
    lr = D.compute_returns(prices["A"], method="log")
    assert np.isclose(lr.sum(), np.log(108.9 / 100))


def test_clean_no_backfill(prices):
    c = D.clean_prices(prices, ffill_limit=1)
    assert c["B"].iloc[:2].isna().all()      # leading NaN kept (no back-fill)
    assert c["B"].iloc[4] == 55              # gap forward-filled


def test_resample_compounds(prices):
    r = D.compute_returns(prices["A"])
    w = D.resample_returns(r, "YE")
    assert np.isclose(w.iloc[0], 108.9 / 100 - 1)


def _fake_download(tickers, **kwargs):
    idx = pd.bdate_range("2021-01-01", periods=4)
    cols = pd.MultiIndex.from_product([["Close", "Open"], tickers])
    return pd.DataFrame(np.arange(1, 4 * len(cols) + 1, dtype=float).reshape(4, -1), index=idx, columns=cols)


def test_download_multiindex_and_cache(tmp_path):
    with patch("yfinance.download", side_effect=_fake_download) as m:
        p = D.download_prices(["SPY", "TLT"], "2021-01-01", "2021-02-01", cache_dir=tmp_path)
        p2 = D.download_prices(["SPY", "TLT"], "2021-01-01", "2021-02-01", cache_dir=tmp_path)
    assert list(p.columns) == ["SPY", "TLT"]
    assert m.call_count == 1                 # second call hit the cache
    pd.testing.assert_frame_equal(p, p2, check_freq=False)
