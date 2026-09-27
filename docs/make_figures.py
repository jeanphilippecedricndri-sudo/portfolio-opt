"""Generate every figure used in docs/METHODOLOGY.md.

    python docs/make_figures.py

Pedagogical figures (walk-forward, ERC vs MV vs 1/N, HRP steps, Ledoit-Wolf spectrum, drawdown
anatomy, VaR/CVaR, Sharpe uncertainty) + the standard backtest report on simulated data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.cluster import hierarchy as sch

from portopt import ERC, HRP, EqualWeight, InverseVolatility, RiskParity, run_backtests
from portopt import metric as M
from portopt import plot as P
from portopt.data import simulate_returns
from portopt.estimator import ledoit_wolf_cov, risk_contributions

OUT = Path(__file__).parent / "figures"
BLUE, ORANGE, AQUA, YELLOW = P.PALETTE[:4]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("  ", name)


# ------------------------------------------------------------------ 1. walk-forward
def fig_walk_forward():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    L, H, n = 12, 3, 5
    for k in range(n):
        y = n - 1 - k
        s = k * H
        ax.add_patch(FancyBboxPatch((s, y - 0.3), L, 0.6, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="#cde2fb", ec="none"))
        ax.add_patch(FancyBboxPatch((s + L + 0.08, y - 0.3), H - 0.08, 0.6,
                                    boxstyle="round,pad=0,rounding_size=0.12", fc=ORANGE, ec="none"))
        ax.text(s + L / 2, y, "in-sample: fit", ha="center", va="center", fontsize=8.5, color="#184f95")
        ax.text(s + L + H / 2, y, "hold", ha="center", va="center", fontsize=8.5, color="white")
        ax.plot([s + L, s + L], [y - 0.42, y + 0.42], color=P.INK, lw=1.0)
        ax.text(-0.6, y, f"rebal. {k + 1}", ha="right", va="center", fontsize=9, color=P.INK_2)
    ax.annotate("", xy=(L, -0.75), xytext=(0, -0.75), arrowprops=dict(arrowstyle="<->", color=P.INK_2, lw=0.9))
    ax.text(L / 2, -0.95, "L = in_sample", ha="center", va="top", fontsize=9, color=P.INK_2)
    ax.annotate("", xy=(L + H, -0.75), xytext=(L, -0.75), arrowprops=dict(arrowstyle="<->", color=P.INK_2, lw=0.9))
    ax.text(L + H / 2, -0.95, "H = out_of_sample", ha="center", va="top", fontsize=9, color=P.INK_2)
    ax.text(L, n - 0.35, "t", ha="center", va="bottom", fontsize=9, color=P.INK)
    ax.set_xlim(-4.5, (n - 1) * H + L + H + 0.5)
    ax.set_ylim(-1.5, n)
    ax.axis("off")
    P._header(ax, "Walk-forward backtest (rolling window)",
              "Weights fitted on returns up to t-1 are held on [t, t+H), then the window rolls by H")
    save(fig, "walk_forward.png")


# ------------------------------------------------------------------ 2. MV <= ERC <= 1/N (2 assets)
def fig_erc_two_assets():
    s1, s2, rho = 0.18, 0.06, -0.2
    cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w1 = np.linspace(0, 1, 401)
    vol = np.sqrt(w1**2 * s1**2 + (1 - w1) ** 2 * s2**2 + 2 * w1 * (1 - w1) * rho * s1 * s2)
    w_mv = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    w_erc = (1 / s1) / (1 / s1 + 1 / s2)  # 2 assets: ERC = inverse vol for any rho
    pts = {"Min variance": w_mv, "ERC = InvVol (N = 2)": w_erc, "1/N": 0.5}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(w1, vol, color=P.MUTED, lw=1.6)
    for (name, w), c in zip(pts.items(), [AQUA, BLUE, ORANGE]):
        v = np.interp(w, w1, vol)
        ax.plot([w], [v], "o", ms=8, color=c, mec="white", mew=2, zorder=4, label=name)
        ax.annotate(f"{v:.1%}", (w, v), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xlabel("Weight in the risky asset (σ = 18%)")
    P._pct(ax, decimals=0)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    P._header(ax, "Portfolio volatility, 2 assets", "σ1 = 18%, σ2 = 6%, ρ = -0.2", legend_ncol=3)

    ax = axes[1]
    names = list(pts)
    rc = np.array([risk_contributions(np.array([w, 1 - w]), cov) for w in pts.values()])
    y = np.arange(len(names))[::-1]
    ax.barh(y, rc[:, 0], color=BLUE, height=0.55, label="risky asset")
    ax.barh(y, rc[:, 1], left=rc[:, 0], color="#9ec5f4", height=0.55, label="safe asset")
    for yi, r0 in zip(y, rc[:, 0]):
        ax.text(min(max(r0 / 2, 0.06), 0.9), yi, f"{r0:.0%}", va="center", ha="center", fontsize=8.5,
                color="white" if r0 > 0.12 else P.INK)
    ax.set_yticks(y, ["MV", "ERC", "1/N"])
    ax.axvline(0.5, color=P.INK_2, lw=0.8, ls=(0, (2, 2)))
    ax.set_xlim(min(0, rc.min()) - 0.02, 1.02)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    P._header(ax, "Share of total risk", "Relative risk contributions", legend_ncol=2)
    fig.tight_layout(w_pad=3)
    save(fig, "erc_two_assets.png")


# ------------------------------------------------------------------ 3. HRP steps
def fig_hrp_steps(returns):
    rng = np.random.default_rng(3)
    cols = list(returns.columns)
    rng.shuffle(cols)
    r = returns[cols].iloc[-756:]
    est = HRP(linkage="single").fit(r)
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.8, 1], wspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    P.plot_correlation(r, title="1. Original order", subtitle="Correlation, arbitrary asset order", ax=ax0,
                       annot_size=6.3)
    ax1 = fig.add_subplot(gs[1])
    with plt.rc_context({"lines.linewidth": 1.4}):
        sch.dendrogram(est.linkage_matrix_, labels=cols, ax=ax1, color_threshold=0,
                       above_threshold_color=BLUE, leaf_font_size=8.5)
    ax1.set_ylabel("distance  d = √((1-ρ)/2)")
    ax1.grid(axis="x", visible=False)
    ax1.tick_params(axis="x", rotation=90)
    P._header(ax1, "2. Single-linkage tree", "Leaves order = quasi-diagonalisation", extra=20)
    ax2 = fig.add_subplot(gs[2])
    P.plot_correlation(r, order=est.order_, title="3. Quasi-diagonal order",
                       subtitle="Correlated assets become neighbours", ax=ax2, annot_size=6.3)
    for ax in (ax0, ax2):
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    save(fig, "hrp_steps.png")


# ------------------------------------------------------------------ 4. Ledoit-Wolf spectrum
def fig_lw_spectrum():
    rng = np.random.default_rng(0)
    N, T = 50, 75
    beta = rng.normal(1, 0.3, N)
    true = np.outer(beta, beta) * 0.01**2 + np.diag(rng.uniform(0.5, 2.0, N) * 0.01**2)
    x = pd.DataFrame(rng.multivariate_normal(np.zeros(N), true, T))
    S = np.cov(x.to_numpy(), rowvar=False)
    LW, delta = ledoit_wolf_cov(x)
    ev = {n: np.sort(np.linalg.eigvalsh(m))[::-1] for n, m in [("True Σ", true), ("Sample S", S),
                                                                ("Ledoit-Wolf", LW)]}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    k = np.arange(1, N + 1)
    for (n, e), c in zip(ev.items(), [P.MUTED, ORANGE, BLUE]):
        ax.plot(k, e, color=c, marker="o", ms=4, mec="white", mew=1, label=n, lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank")
    ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    cond = {n: e[0] / e[-1] for n, e in ev.items()}
    P._header(ax, f"Eigenvalue spectrum: N = {N} assets, T = {T} days",
              f"Condition number: true {cond['True Σ']:,.0f}, sample {cond['Sample S']:,.0f}, "
              f"Ledoit-Wolf {cond['Ledoit-Wolf']:,.0f} (δ = {delta:.2f})", legend_ncol=3)
    ax.margins(x=0.01)
    save(fig, "lw_spectrum.png")


# ------------------------------------------------------------------ 5. drawdown anatomy
def fig_drawdown_anatomy(r):
    w = M.wealth_index(r)
    dd = M.drawdown_series(r)
    trough = dd.idxmin()
    peak = w.loc[:trough].idxmax()
    after = w.loc[trough:]
    rec = after[after >= w.loc[peak]].index
    recovery = rec[0] if len(rec) else w.index[-1]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(w.index, w, color=BLUE)
    ax.plot(w.index, w.cummax().clip(lower=1), color=P.MUTED, lw=1.0, ls=(0, (3, 2)), label="running peak")
    for d, lab, va in [(peak, "peak", "bottom"), (trough, "trough", "top"), (recovery, "recovery", "bottom")]:
        ax.plot([d], [w.loc[d]], "o", ms=7, color=BLUE, mec="white", mew=2, zorder=4)
        ax.annotate(lab, (d, w.loc[d]), xytext=(0, 8 if va == "bottom" else -10), textcoords="offset points",
                    ha="center", va=va, fontsize=9)
    ax.annotate("", xy=(trough, w.loc[trough]), xytext=(trough, w.loc[peak]),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    ax.annotate(f"Max DD = {dd.min():.1%}", (trough, (w.loc[trough] + w.loc[peak]) / 2), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    ax.legend(loc="upper left")
    P._header(ax, "Anatomy of a drawdown", "1/N portfolio, simulated data")
    ax = axes[1]
    ax.fill_between(dd.index, dd, 0, color=BLUE, alpha=0.12, lw=0)
    ax.plot(dd.index, dd, color=BLUE, lw=1.1)
    ax.axvspan(peak, recovery, color=ORANGE, alpha=0.08, lw=0)
    ax.annotate("", xy=(recovery, dd.min() * 1.08), xytext=(peak, dd.min() * 1.08),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.2))
    n_days = w.loc[peak:recovery].shape[0] - 1
    ax.text(peak + (recovery - peak) / 2, dd.min() * 1.12, f"underwater {n_days} days", ha="center", va="top",
            fontsize=9)
    ax.set_ylim(dd.min() * 1.35, 0.02)
    P._pct(ax, decimals=0)
    ax.set_title("Drawdown", fontsize=10.5, loc="left")
    fig.tight_layout()
    save(fig, "drawdown_anatomy.png")


# ------------------------------------------------------------------ 6. VaR / CVaR
def fig_var_cvar(r):
    alpha = 0.05
    var, cvar = M.value_at_risk(r, alpha), M.conditional_value_at_risk(r, alpha)
    mu, sd = r.mean(), r.std()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    bins = np.linspace(r.quantile(0.001), r.quantile(0.999), 90)
    ax.hist(r, bins=bins, density=True, color="#9ec5f4", rwidth=0.85, label="empirical")
    xx = np.linspace(bins[0], bins[-1], 400)
    ax.plot(xx, stats.norm.pdf(xx, mu, sd), color=P.INK_2, lw=1.2, ls=(0, (3, 2)), label="normal, same μ and σ")
    top = ax.get_ylim()[1]
    for v, lab, c, dx, ha, yf in [(-var, f"VaR 95% = {var:.2%}", BLUE, 6, "left", 0.92),
                                   (-cvar, f"CVaR 95% = {cvar:.2%}", ORANGE, -6, "right", 0.78)]:
        ax.axvline(v, color=c, lw=1.6)
        ax.annotate(lab, (v, top * yf), xytext=(dx, 0), textcoords="offset points", ha=ha, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))
    ax.axvspan(bins[0], -var, color=ORANGE, alpha=0.07, lw=0)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("Daily return")
    k = M.excess_kurtosis(r)
    P._header(ax, "Daily return distribution and tail risk",
              f"1/N portfolio, excess kurtosis {k:.1f}: the normal curve underestimates the tail (shaded)",
              legend_ncol=2)
    save(fig, "var_cvar.png")


# ------------------------------------------------------------------ 7. Sharpe uncertainty
def fig_sharpe_ci():
    years = np.linspace(1, 30, 300)
    fig, ax = plt.subplots(figsize=(11, 4))
    for sr, c in [(0.5, BLUE), (1.0, ORANGE)]:
        se = np.sqrt((1 + 0.5 * (sr / np.sqrt(252)) ** 2) / (years * 252)) * np.sqrt(252)
        ax.fill_between(years, sr - 1.96 * se, sr + 1.96 * se, color=c, alpha=0.12, lw=0)
        ax.plot(years, np.full_like(years, sr), color=c, label=f"true Sharpe {sr}")
        y_sig = years[np.argmax(sr - 1.96 * se > 0)]
        ax.plot([y_sig], [0], "o", ms=7, color=c, mec="white", mew=2, zorder=4)
        ax.annotate(f"{y_sig:.0f} y to be\nsignificant", (y_sig, 0), xytext=(6, -26), textcoords="offset points",
                    fontsize=8.5, va="center")
    ax.axhline(0, color=P.MUTED, lw=0.8)
    ax.set_xlabel("Track record length (years)")
    ax.set_xlim(1, 30)
    P._header(ax, "How precise is an estimated Sharpe ratio?",
              "95% confidence band, iid returns: se ≈ √((1 + SR²/2) / T), T in years", legend_ncol=2)
    save(fig, "sharpe_ci.png")


def main():
    P.use_style()
    print("figures ->", OUT)
    returns = simulate_returns()
    fig_walk_forward()
    fig_erc_two_assets()
    fig_hrp_steps(returns)
    fig_lw_spectrum()
    ew = returns.mean(axis=1)
    fig_drawdown_anatomy(ew)
    fig_var_cvar(ew)
    fig_sharpe_ci()

    budgets = {t: 1.0 for t in returns.columns} | {"TLT": 2.0, "IEF": 2.0, "LQD": 2.0}
    estimators = {"1/N": EqualWeight(), "InvVol": InverseVolatility(), "RB bond tilt": RiskParity(budgets=budgets),
                  "ERC": ERC(), "ERC LW": ERC(cov_method="ledoit_wolf"), "HRP": HRP()}
    rets, bts = run_backtests(returns, estimators, 252, 21, transaction_cost_bps=5)
    tmp = OUT / "_report"
    for p in P.save_report(rets, bts, tmp):
        if p.name.startswith("08_") and "erc" not in p.name.split("_", 2)[-1] and "hrp" not in p.name:
            continue
        shutil.copy(p, OUT / f"report_{p.name}")
        print("  ", f"report_{p.name}")
    shutil.rmtree(tmp)
    summ = M.summary(rets, turnover={k: b.annualized_turnover() for k, b in bts.items()})
    summ.to_csv(OUT / "report_summary.csv")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
