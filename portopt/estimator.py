"""Estimator module: allocation methods.

Every estimator follows the scikit-learn pattern:

    est = ERC(cov_method="ledoit_wolf").fit(returns_in_sample)
    est.weights_          # pd.Series indexed by asset, sums to 1, long-only

Implemented: EqualWeight (1/N), InverseVolatility, RiskParity (general risk budgeting),
ERC (equal risk contribution), HRP (Lopez de Prado, 2016).
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform

CovMethod = Literal["sample", "ledoit_wolf", "ewma"] | Callable[[pd.DataFrame], np.ndarray]


# --------------------------------------------------------------------------------------
# Covariance estimation
# --------------------------------------------------------------------------------------
def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    """Unbiased sample covariance (ddof=1)."""
    return np.cov(returns.to_numpy(), rowvar=False, ddof=1)


def ledoit_wolf_cov(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004) shrinkage towards the scaled identity  mu * I.

    Sigma_hat = delta * mu * I + (1 - delta) * S, with the optimal delta in closed form.
    Returns (Sigma_hat, delta).
    """
    x = returns.to_numpy()
    n, p = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / n
    mu = np.trace(s) / p
    # normalized Frobenius norm ||A||^2 = tr(A A') / p
    d2 = np.sum((s - mu * np.eye(p)) ** 2) / p
    if d2 <= 0:
        return s, 0.0
    # b2_bar = 1/n^2 sum_k ||x_k x_k' - S||^2 = 1/n^2 [sum_k (x_k'x_k)^2 - n ||S||^2]
    b2_bar = (np.sum(np.sum(x**2, axis=1) ** 2) - n * np.sum(s**2)) / (n**2 * p)
    b2 = min(b2_bar, d2)
    delta = b2 / d2
    return delta * mu * np.eye(p) + (1.0 - delta) * s, float(delta)


def ewma_cov(returns: pd.DataFrame, halflife: float = 63.0) -> np.ndarray:
    """Exponentially weighted covariance (RiskMetrics style), most recent obs weighs most."""
    x = returns.to_numpy()
    n = x.shape[0]
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n - 1, -1, -1)
    w /= w.sum()
    xc = x - w @ x
    return (xc * w[:, None]).T @ xc


def estimate_cov(returns: pd.DataFrame, method: CovMethod = "sample") -> np.ndarray:
    if callable(method):
        cov = np.asarray(method(returns), dtype=float)
    elif method == "sample":
        cov = sample_cov(returns)
    elif method == "ledoit_wolf":
        cov = ledoit_wolf_cov(returns)[0]
    elif method == "ewma":
        cov = ewma_cov(returns)
    else:
        raise ValueError(f"Unknown cov method: {method}")
    return 0.5 * (cov + cov.T)  # enforce exact symmetry


def risk_contributions(weights: pd.Series | np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Relative risk contributions RC_i / sigma_p^2 = w_i (Sigma w)_i / (w' Sigma w). Sum to 1."""
    w = np.asarray(weights, dtype=float)
    mrc = cov @ w
    return w * mrc / (w @ mrc)


# --------------------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------------------
class BaseEstimator(ABC):
    """Common interface. Subclasses implement ``_compute_weights(returns) -> np.ndarray``."""

    def __init__(self, cov_method: CovMethod = "sample"):
        self.cov_method = cov_method
        self.weights_: pd.Series | None = None
        self.cov_: np.ndarray | None = None

    @property
    def name(self) -> str:
        return type(self).__name__

    def fit(self, returns: pd.DataFrame) -> "BaseEstimator":
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame (T x N).")
        if returns.isna().any().any():
            raise ValueError("returns contain NaN: select a clean universe before fitting.")
        if returns.shape[1] == 0:
            raise ValueError("returns has no column.")
        w = np.asarray(self._compute_weights(returns), dtype=float)
        if not np.all(np.isfinite(w)):
            raise FloatingPointError(f"{self.name} produced non-finite weights.")
        w = np.clip(w, 0.0, None)
        w /= w.sum()
        self.weights_ = pd.Series(w, index=returns.columns, name=self.name)
        return self

    def _cov(self, returns: pd.DataFrame) -> np.ndarray:
        self.cov_ = estimate_cov(returns, self.cov_method)
        return self.cov_

    @abstractmethod
    def _compute_weights(self, returns: pd.DataFrame) -> np.ndarray: ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.endswith("_"))
        return f"{self.name}({params})"


# --------------------------------------------------------------------------------------
# Allocation methods
# --------------------------------------------------------------------------------------
class EqualWeight(BaseEstimator):
    """w_i = 1/N. No estimation at all, hence zero estimation error (DeMiguel et al. 2009)."""

    def _compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        n = returns.shape[1]
        return np.full(n, 1.0 / n)


class InverseVolatility(BaseEstimator):
    """w_i proportional to 1 / sigma_i^exponent.

    exponent=1 : inverse volatility ("naive risk parity"). Equals ERC if all pairwise
                 correlations are equal.
    exponent=2 : inverse variance. Equals min-variance if Sigma is diagonal.
    """

    def __init__(self, cov_method: CovMethod = "sample", exponent: float = 1.0):
        super().__init__(cov_method)
        self.exponent = exponent

    def _compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        vol = np.sqrt(np.diag(self._cov(returns)))
        if np.any(vol <= 0):
            raise ValueError("Zero volatility asset in the window.")
        return vol ** (-self.exponent)


class RiskParity(BaseEstimator):
    """Risk budgeting: find long-only w with RC_i / sigma_p^2 = b_i for given budgets b.

    Solved through the convex program of Spinu (2013) / Roncalli:

        y* = argmin_{y > 0}  1/2 y' Sigma y - sum_i b_i ln(y_i),     w = y* / sum(y*)

    First-order condition (Sigma y)_i = b_i / y_i  =>  y_i (Sigma y)_i = b_i, i.e. risk
    contributions proportional to b. The problem is strictly convex, so the solution is
    unique, and cyclical coordinate descent (Griveau-Billion, Richard & Roncalli, 2013)
    has a closed-form update per coordinate:

        y_i = [ -c_i + sqrt(c_i^2 + 4 Sigma_ii b_i) ] / (2 Sigma_ii),   c_i = sum_{j!=i} Sigma_ij y_j
    """

    def __init__(
        self,
        cov_method: CovMethod = "sample",
        budgets: Mapping[str, float] | Sequence[float] | pd.Series | None = None,
        tol: float = 1e-10,
        max_iter: int = 10_000,
    ):
        super().__init__(cov_method)
        self.budgets = budgets
        self.tol = tol
        self.max_iter = max_iter
        self.n_iter_: int | None = None

    def _budget_vector(self, columns: pd.Index) -> np.ndarray:
        n = len(columns)
        if self.budgets is None:
            b = np.full(n, 1.0 / n)
        elif isinstance(self.budgets, (Mapping, pd.Series)):
            b = pd.Series(self.budgets, dtype=float).reindex(columns).to_numpy()
            if np.isnan(b).any():
                raise ValueError("budgets missing for some assets of the window.")
        else:
            b = np.asarray(self.budgets, dtype=float)
            if b.shape != (n,):
                raise ValueError(f"budgets must have length {n}.")
        if np.any(b <= 0):
            raise ValueError("budgets must be strictly positive (log barrier).")
        return b / b.sum()

    def _compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        cov = self._cov(returns)
        b = self._budget_vector(returns.columns)
        y, n_iter = self._ccd(cov, b)
        self.n_iter_ = n_iter
        return y / y.sum()

    def _ccd(self, cov: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
        diag = np.diag(cov).copy()
        if np.any(diag <= 0):
            raise ValueError("Zero variance asset in the window.")
        y = b / np.sqrt(diag)
        y /= np.sqrt(y @ cov @ y)  # start on the right scale: y'Sigma y = 1 at optimum
        sigma_y = cov @ y
        for it in range(1, self.max_iter + 1):
            y_old = y.copy()
            for i in range(len(y)):
                c = sigma_y[i] - diag[i] * y[i]
                new = (-c + np.sqrt(c * c + 4.0 * diag[i] * b[i])) / (2.0 * diag[i])
                sigma_y += cov[:, i] * (new - y[i])  # O(N) incremental update of Sigma y
                y[i] = new
            if np.linalg.norm(y - y_old) <= self.tol * np.linalg.norm(y):
                return y, it
        warnings.warn(f"{self.name}: CCD did not converge in {self.max_iter} iterations.")
        return y, self.max_iter


class ERC(RiskParity):
    """Equal Risk Contribution: risk budgeting with b_i = 1/N.

    Known bounds (Maillard, Roncalli & Teiletche 2010):
        sigma_MV <= sigma_ERC <= sigma_1/N
    and w_ERC = w_InvVol when all correlations are equal.
    """

    def __init__(self, cov_method: CovMethod = "sample", tol: float = 1e-10, max_iter: int = 10_000):
        super().__init__(cov_method=cov_method, budgets=None, tol=tol, max_iter=max_iter)


class HRP(BaseEstimator):
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    1. Distance  d_ij = sqrt((1 - rho_ij) / 2)  (a proper metric on correlations).
    2. Hierarchical clustering (``linkage``), then quasi-diagonalisation = leaf order.
    3. Top-down allocation: at each split between clusters L and R,
           alpha = 1 - V_L / (V_L + V_R),   w_L *= alpha,  w_R *= 1 - alpha
       with V_c = w_c' Sigma_c w_c and w_c the inverse-variance weights inside c.

    split="bisection"  : original paper, halves the ordered list (ignores the tree shape).
    split="dendrogram" : follows the actual dendrogram splits (more faithful to the clusters).

    No matrix inversion anywhere, so it is robust when Sigma is ill-conditioned.
    """

    def __init__(
        self,
        cov_method: CovMethod = "sample",
        linkage: Literal["single", "complete", "average", "ward"] = "single",
        split: Literal["bisection", "dendrogram"] = "bisection",
    ):
        super().__init__(cov_method)
        self.linkage = linkage
        self.split = split
        self.order_: list | None = None
        self.linkage_matrix_: np.ndarray | None = None

    @staticmethod
    def _cluster_var(cov: np.ndarray, idx: Sequence[int]) -> float:
        sub = cov[np.ix_(idx, idx)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    def _compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        cov = self._cov(returns)
        n = cov.shape[0]
        if n == 1:
            return np.ones(1)
        std = np.sqrt(np.diag(cov))
        corr = np.clip(cov / np.outer(std, std), -1.0, 1.0)
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
        np.fill_diagonal(dist, 0.0)
        link = sch.linkage(squareform(dist, checks=False), method=self.linkage)
        self.linkage_matrix_ = link
        order = sch.leaves_list(link).tolist()
        self.order_ = [returns.columns[i] for i in order]

        w = np.ones(n)
        if self.split == "bisection":
            clusters = [order]
            while clusters:
                clusters = [
                    part
                    for c in clusters
                    if len(c) > 1
                    for part in (c[: len(c) // 2], c[len(c) // 2 :])
                ]
                for left, right in zip(clusters[::2], clusters[1::2]):
                    self._split(cov, w, left, right)
        elif self.split == "dendrogram":
            stack = [sch.to_tree(link)]
            while stack:
                node = stack.pop()
                if node.is_leaf():
                    continue
                self._split(cov, w, node.get_left().pre_order(), node.get_right().pre_order())
                stack.extend([node.get_left(), node.get_right()])
        else:
            raise ValueError("split must be 'bisection' or 'dendrogram'")
        return w

    def _split(self, cov: np.ndarray, w: np.ndarray, left: list, right: list) -> None:
        v_l, v_r = self._cluster_var(cov, left), self._cluster_var(cov, right)
        alpha = 1.0 - v_l / (v_l + v_r)
        w[left] *= alpha
        w[right] *= 1.0 - alpha
