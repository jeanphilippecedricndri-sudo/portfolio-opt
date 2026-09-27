"""Plot module: white-background, publication-grade charts for backtests.

Design rules (applied everywhere):
* white surface, recessive hairline grid (y only), no top/right spines;
* categorical colors assigned by entity in a fixed, colorblind-validated order (never cycled),
  so a strategy keeps its color in every chart;
* 2px lines, 10% area washes, sequential blue ramp for magnitudes, blue/gray/red for signed values;
* text in ink tokens, never in series colors; direct end labels with leader lines when they collide.

    from portopt import plot
    plot.use_style()
    plot.save_report(oos_returns, backtesters, "outputs/")
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb

from . import metric as M

# ----------------------------------------------------------------------------- tokens
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#ebeae6"

# categorical order validated for CVD separation on white (adjacent pairs)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEUTRAL = "#f0efec"
DIV_NEG, DIV_POS = "#e34948", "#2a78d6"

CMAP_SEQ = LinearSegmentedColormap.from_list("portopt_seq", SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "portopt_div", ["#a8201f", DIV_NEG, "#f3b1b0", DIV_NEUTRAL, "#9ec5f4", DIV_POS, "#104281"]
)

_FONT_DIR = Path(__file__).parent / "fonts"


def use_style() -> None:
    """Register the bundled Space Grotesk / Space Mono fonts and set a clean white style."""
    for f in _FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
        "font.family": ["Space Grotesk", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 12.5,
        "axes.titleweight": 600,
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.handlelength": 1.4,
        "figure.titlesize": 15,
        "figure.titleweight": 600,
    })


def colors_for(names: Sequence[str]) -> dict[str, str]:
    """Fixed entity -> color map (in the given order). Folds past 8 into muted gray."""
    return {n: (PALETTE[i] if i < len(PALETTE) else MUTED) for i, n in enumerate(names)}


def _header(ax, title: str, subtitle: str = "", legend_ncol: int = 0, extra: float = 0.0) -> None:
    """Title, subtitle and (optional) legend stacked ABOVE the plot area, never over the data."""
    y = extra
    if legend_ncol:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=legend_ncol, borderaxespad=0.4,
                  columnspacing=1.6, handletextpad=0.6)
        y += 20
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, y + 4), textcoords="offset points",
                    color=INK_2, fontsize=9, va="bottom", ha="left", annotation_clip=False)
        y += 16
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, y + 6), textcoords="offset points",
                color=INK, fontsize=12.5, fontweight=600, va="bottom", ha="left", annotation_clip=False)


def _fig_header(fig, title: str, subtitle: str = "") -> float:
    """Figure-level title/subtitle at a fixed distance from the top edge. Returns the rect top."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.10 / h, title, fontsize=14, fontweight=600, va="top", ha="left")
    if subtitle:
        fig.text(0.01, 1 - 0.40 / h, subtitle, fontsize=9, color=INK_2, va="top", ha="left")
    return 1 - (0.70 if subtitle else 0.45) / h


def _pct(ax, axis: str = "y", decimals: int | None = None) -> None:
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _end_labels(ax, series: Mapping[str, pd.Series], colors: Mapping[str, str], fmt, min_gap_px: float = 17):
    """Direct labels at line ends; spread vertically when they collide, with leader lines."""
    fig = ax.figure
    fig.canvas.draw()
    ends = []
    for name, s in series.items():
        s = s.dropna()
        x, y = s.index[-1], s.iloc[-1]
        px = ax.transData.transform((mpl.dates.date2num(x), y))[1]
        ends.append([name, x, y, px])
    ends.sort(key=lambda e: e[3])
    adj = [e[3] for e in ends]
    for i in range(1, len(adj)):  # push up
        adj[i] = max(adj[i], adj[i - 1] + min_gap_px)
    shift = (adj[-1] - ends[-1][3]) / 2 if len(adj) > 1 else 0  # recentre the block
    adj = [a - shift for a in adj]
    for i in range(len(adj) - 2, -1, -1):  # re-check downward after recentring
        adj[i] = min(adj[i], adj[i + 1] - min_gap_px)
    pt = 72.0 / fig.dpi
    for (name, x, y, px), a in zip(ends, adj):
        ax.annotate(
            f"{name}  {fmt(y)}", xy=(x, y), xytext=(22, (a - px) * pt), textcoords="offset points",
            va="center", fontsize=8.8, color=INK, annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=colors[name], lw=1.0, shrinkA=0, shrinkB=2),
        )
        ax.plot([x], [y], "o", ms=5, color=colors[name], mec=SURFACE, mew=1.5, zorder=5, clip_on=False)


# ----------------------------------------------------------------------------- charts
def plot_wealth(returns: pd.DataFrame, ax=None, log: bool = True, colors=None, title="Growth of 1"):
    colors = colors or colors_for(returns.columns)
    ax = ax or plt.subplots(figsize=(11, 5.2))[1]
    wealth = M.wealth_index(returns)
    for c in returns.columns:
        ax.plot(wealth.index, wealth[c], color=colors[c], label=c)
    if log:
        ax.set_yscale("log")
        lo, hi = wealth.min().min(), wealth.max().max()
        nice = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100]
        ticks = [t for t in nice if lo * 0.97 <= t <= hi * 1.03]
        if len(ticks) > 9:
            ticks = ticks[:: int(np.ceil(len(ticks) / 9))]
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.axhline(1.0, color=MUTED, lw=0.8, zorder=1)
    _header(ax, title, "Out-of-sample wealth, net of transaction costs" + (", log scale" if log else ""),
            legend_ncol=min(len(returns.columns), 6))
    ax.margins(x=0)
    _end_labels(ax, {c: wealth[c] for c in returns.columns}, colors, lambda v: f"×{v:.2f}")
    return ax


def plot_drawdowns(returns: pd.DataFrame, colors=None, ncols: int = 3):
    """Small multiples: one panel per strategy, shared y, deepest trough annotated."""
    colors = colors or colors_for(returns.columns)
    dd = M.drawdown_series(returns)
    n = len(returns.columns)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.3 * nrows + 0.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, returns.columns):
        s = dd[c]
        ax.fill_between(s.index, s, 0, color=colors[c], alpha=0.12, lw=0)
        ax.plot(s.index, s, color=colors[c], lw=1.2)
        t = s.idxmin()
        ax.plot([t], [s.min()], "o", ms=5, color=colors[c], mec=SURFACE, mew=1.5)
        ax.annotate(f"{s.min():.1%}", (t, s.min()), xytext=(6, -2), textcoords="offset points",
                    fontsize=8.5, color=INK, va="top")
        ax.set_title(c, fontsize=10.5, pad=6)
        _pct(ax, decimals=0)
    for ax in axes[n:]:
        ax.set_visible(False)
    top = _fig_header(fig, "Drawdowns", "Peak-to-trough loss of wealth, dot = maximum drawdown")
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def plot_rolling(returns: pd.DataFrame, stat: str = "vol", window: int = 252, colors=None, ax=None):
    """Rolling annualized volatility or Sharpe. One measure per chart (never dual axes)."""
    colors = colors or colors_for(returns.columns)
    ax = ax or plt.subplots(figsize=(11, 4))[1]
    r = returns
    if stat == "vol":
        val = r.rolling(window).std() * np.sqrt(M.PPY)
        title, sub = "Rolling volatility", f"Annualized, {window}-day window"
    elif stat == "sharpe":
        val = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(M.PPY)
        title, sub = "Rolling Sharpe ratio", f"Annualized, {window}-day window, rf = 0"
    else:
        raise ValueError("stat must be 'vol' or 'sharpe'")
    for c in r.columns:
        ax.plot(val.index, val[c], color=colors[c], label=c, lw=1.3)
    if stat == "vol":
        _pct(ax)
    else:
        ax.axhline(0, color=MUTED, lw=0.8)
    _header(ax, title, sub, legend_ncol=min(len(r.columns), 6))
    ax.margins(x=0)
    return ax


def plot_heatmap(
    values: pd.DataFrame, title: str, subtitle: str = "", fmt: str = "{:.0%}", diverging: bool = False,
    ax=None, vmax: float | None = None, annot_size: float = 8.5,
):
    """Annotated heatmap (rows x cols). Sequential blue, or blue/gray/red for signed values."""
    h, w = values.shape
    ax = ax or plt.subplots(figsize=(max(6, 0.85 * w + 2.5), 0.55 * h + 1.6))[1]
    v = values.to_numpy(dtype=float)
    if diverging:
        m = vmax or np.nanmax(np.abs(v))
        norm, cmap = TwoSlopeNorm(0, -m, m), CMAP_DIV
    else:
        norm, cmap = mpl.colors.Normalize(0, vmax or np.nanmax(v)), CMAP_SEQ
    ax.imshow(v, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(w), values.columns)
    ax.set_yticks(range(h), values.index)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, w), minor=True)
    ax.set_yticks(np.arange(-0.5, h), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)  # 2px surface gap between cells
    ax.tick_params(which="minor", length=0)
    for i in range(h):
        for j in range(w):
            if np.isnan(v[i, j]):
                continue
            rgb = np.array(to_rgb(cmap(norm(v[i, j]))))
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            label = fmt.format(v[i, j])
            if label.startswith("-") and not label.strip("-0.%"):
                label = label[1:]  # no "-0.00"
            ax.text(j, i, label, ha="center", va="center", fontsize=annot_size,
                    color=SURFACE if lum < 0.5 else INK)
    _header(ax, title, subtitle, extra=20)  # 20pt clears the top tick labels
    return ax


def plot_weights_through_time(weights: pd.DataFrame, title: str, ax=None):
    """Assets x rebalancing dates heatmap of target weights (readable for any number of assets)."""
    ax = ax or plt.subplots(figsize=(11, 0.33 * weights.shape[1] + 1.4))[1]
    x = mpl.dates.date2num(weights.index)
    extent = (x[0], x[-1], weights.shape[1] - 0.5, -0.5)
    im = ax.imshow(weights.T.to_numpy(), aspect="auto", cmap=CMAP_SEQ, extent=extent, vmin=0,
                   interpolation="nearest")
    ax.xaxis_date()
    ax.set_yticks(range(weights.shape[1]), weights.columns)
    ax.grid(False)
    for k in range(1, weights.shape[1]):
        ax.axhline(k - 0.5, color=SURFACE, lw=2)  # surface gap between asset rows
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.01, format=mpl.ticker.PercentFormatter(1.0, decimals=0))
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    _header(ax, title, "Target weight at each rebalancing")
    return ax


def plot_metrics(summary: pd.DataFrame, metrics: Sequence[str] | None = None, colors=None, ncols: int = 3):
    """Small multiples of horizontal bars, one panel per metric, strategies keep their color."""
    metrics = metrics or ["CAGR", "Ann. vol", "Sharpe", "Max DD", "CVaR 95%", "Ann. turnover"]
    metrics = [m for m in metrics if m in summary.index]
    colors = colors or colors_for(summary.columns)
    pct = {"CAGR", "Ann. mean", "Ann. vol", "Max DD", "VaR 95%", "CVaR 95%", "Hit ratio", "Ann. turnover"}
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 0.34 * len(summary.columns) * nrows + 1.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    names = list(summary.columns)[::-1]
    for ax, m in zip(axes, metrics):
        vals = summary.loc[m, names].astype(float)
        ax.barh(range(len(names)), vals, color=[colors[n] for n in names], height=0.62)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(range(len(names)), names, fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_title(m, fontsize=10.5, pad=6)
        lo, hi = min(vals.min(), 0), max(vals.max(), 0)
        span = hi - lo or 1
        ax.set_xlim(lo - 0.02 * span - (0.28 * span if lo < 0 else 0), hi + 0.3 * span)
        for k, v in enumerate(vals):
            ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), k,
                    f"{v:.1%}" if m in pct else f"{v:.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.3, color=INK)
        ax.set_xticklabels([])
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    top = _fig_header(fig, "Key metrics", "Out-of-sample, net of costs. Turnover is one-way, per year")
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def plot_correlation(returns: pd.DataFrame, order: Sequence[str] | None = None, title="Correlation matrix",
                     subtitle: str = "", ax=None, annot_size: float = 8.5):
    corr = returns.corr()
    if order is not None:
        corr = corr.loc[list(order), list(order)]
    return plot_heatmap(corr, title, subtitle, fmt="{:.2f}", diverging=True, vmax=1.0, ax=ax, annot_size=annot_size)


# ----------------------------------------------------------------------------- report
def save_report(returns: pd.DataFrame, backtesters: Mapping, out_dir: str | Path, rolling_window: int = 252) -> list[Path]:
    """Write the standard set of PNGs for a comparison of strategies. Returns the paths."""
    use_style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    colors = colors_for(returns.columns)
    turnover = {k: bt.annualized_turnover() for k, bt in backtesters.items()}
    summ = M.summary(returns, turnover=turnover)
    paths = []

    def _save(fig, name):
        p = out / name
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    ax = plot_wealth(returns, colors=colors)
    _save(ax.figure, "01_wealth.png")
    _save(plot_drawdowns(returns, colors=colors), "02_drawdowns.png")
    ax = plot_rolling(returns, "vol", rolling_window, colors=colors)
    _save(ax.figure, "03_rolling_vol.png")
    ax = plot_rolling(returns, "sharpe", rolling_window, colors=colors)
    _save(ax.figure, "04_rolling_sharpe.png")
    _save(plot_metrics(summ, colors=colors), "05_metrics.png")

    avg_w = pd.DataFrame({k: bt.weights_.iloc[1:].mean() for k, bt in backtesters.items()}).T
    ax = plot_heatmap(avg_w, "Average capital allocation", "Mean target weight across rebalancings")
    _save(ax.figure, "06_avg_weights.png")
    avg_rc = pd.DataFrame({k: bt.risk_contributions().iloc[1:].mean() for k, bt in backtesters.items()}).T
    ax = plot_heatmap(avg_rc, "Average risk allocation",
                      "Mean relative risk contribution, in-sample covariance at each rebalancing")
    _save(ax.figure, "07_avg_risk_contrib.png")
    for k, bt in backtesters.items():
        safe = "".join(ch if ch.isalnum() else "_" for ch in k).strip("_").lower()
        ax = plot_weights_through_time(bt.weights_, f"{k}: weights through time")
        _save(ax.figure, f"08_weights_{safe}.png")
    return paths
