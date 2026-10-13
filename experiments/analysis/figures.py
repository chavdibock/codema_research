"""All figures of spec §10 as PDF and PNG.

    python -m experiments.analysis.figures --summary summary --out summary/figures \
        [--cache $CDX_CACHE]   # E10 dataset plots read the preprocessing cache

One fixed color per model in every figure (our 4 models: categorical slots 1-4 of the
validated reference palette, in fixed order; baselines in grays) plus a fixed marker and
line style per model, so identity never relies on color alone. Missing runs leave empty
panels marked "no data"; nothing is interpolated or filled in.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.analysis.aggregate import load  # noqa: E402
from experiments.analysis.stats import (BASELINES, DISPLAY, OUR_MODELS, TASK_ORDER,  # noqa: E402
                                        TASK_SHORT)

COLORS = {"cdt": "#2a78d6", "cdt_light": "#eb6834", "cdmamba": "#1baf7a", "cdconv": "#eda100",
          "mt_bc": "#5f5f5f", "dt_cost": "#3f3f3f", "bc_safe": "#7a7a7a", "bcq_lag": "#959595",
          "bear_lag": "#aaaaaa", "cpq": "#4f4f4f", "coptidice": "#b5b5b5"}
MARKERS = {"cdt": "o", "cdt_light": "s", "cdmamba": "D", "cdconv": "^", "mt_bc": "x",
           "dt_cost": "v", "bc_safe": "P", "bcq_lag": "<", "bear_lag": ">", "cpq": "*",
           "coptidice": "h"}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 4,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out / name}.pdf/png")


def no_data(ax, msg: str = "no data") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, transform=ax.transAxes)


def legend_below(fig, axes, models: Sequence[str]) -> None:
    handles = [plt.Line2D([], [], color=COLORS.get(m, MUTED), marker=MARKERS.get(m, "o"),
                          label=DISPLAY.get(m, m)) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               frameon=False, bbox_to_anchor=(0.5, -0.04))


def seed_curve(df: pd.DataFrame, x: str, y: str):
    """Mean and std over seeds of per-seed values (already episode means)."""
    g = df.groupby([x, "seed"])[y].mean().groupby(x)
    return g.mean(), g.std().fillna(0.0)


# ------------------------------------------------------------- Figure 1 (E4)
def fig_zero_shot(runs: pd.DataFrame, out: Path) -> None:
    d = runs[runs["exp"] == "E4"] if not runs.empty else runs
    models = OUR_MODELS + ["mt_bc"]
    fig, axes = plt.subplots(2, len(TASK_ORDER), figsize=(7.0, 3.2), sharex=True)
    for j, task in enumerate(TASK_ORDER):
        ax_c, ax_r = axes[0, j], axes[1, j]
        dt = d[d["task"] == task] if len(d) else d
        ax_c.set_title(TASK_SHORT[task])
        if not len(dt):
            no_data(ax_c)
            no_data(ax_r)
            continue
        ks = np.sort(dt["target_C"].unique())
        ax_c.plot(ks, ks, color=MUTED, lw=1, ls=":", label="y = x")
        cmax = dt["data_c_max"].dropna()
        if len(cmax):
            for ax in (ax_c, ax_r):
                ax.axvline(cmax.iloc[0], color=MUTED, lw=1, ls="--")
        for m in models:
            dm = dt[dt["model"] == m]
            if not len(dm):
                continue
            for ax, y in ((ax_c, "C"), (ax_r, "R_norm")):
                mu, sd = seed_curve(dm, "target_C", y)
                ax.plot(mu.index, mu.values, color=COLORS[m], marker=MARKERS[m],
                        ls="--" if m == "mt_bc" else "-")
                ax.fill_between(mu.index, mu - sd, mu + sd, color=COLORS[m], alpha=0.15, lw=0)
        ax_r.set_xlabel("target cost κ")
    axes[0, 0].set_ylabel("achieved cost")
    axes[1, 0].set_ylabel("normalized reward")
    legend_below(fig, axes, models)
    fig.tight_layout()
    save(fig, out, "fig1_zero_shot")


# ------------------------------------------------------------- Figure 2 (E5)
def fig_conflicting(runs: pd.DataFrame, out: Path) -> None:
    d = runs[runs["exp"] == "E5"] if not runs.empty else runs
    fig, axes = plt.subplots(2, len(TASK_ORDER), figsize=(7.0, 3.2))
    for j, task in enumerate(TASK_ORDER):
        ax_r, ax_c = axes[0, j], axes[1, j]
        ax_r.set_title(TASK_SHORT[task])
        dt = d[d["task"] == task] if len(d) else d
        if not len(dt):
            no_data(ax_r)
            no_data(ax_c)
            continue
        ax_c.axhline(10.0, color=MUTED, lw=1, ls="--")  # kappa
        for m in OUR_MODELS:
            for var, ls in (("", "-"), ("noaug", "--")):
                dm = dt[(dt["model"] == m) & (dt["variant"] == var)]
                if not len(dm):
                    continue
                for ax, y in ((ax_r, "R"), (ax_c, "C")):
                    mu, sd = seed_curve(dm, "target_R", y)
                    ax.plot(mu.index, mu.values, color=COLORS[m], marker=MARKERS[m], ls=ls)
        ax_c.set_xlabel("target reward")
    axes[0, 0].set_ylabel("reward")
    axes[1, 0].set_ylabel("cost")
    legend_below(fig, axes, OUR_MODELS)
    fig.text(0.99, 0.01, "solid: full model, dashed: w/o augmentation", ha="right",
             color=MUTED, fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig2_conflicting_targets")


# ------------------------------------------------------------- Figure 3
def fig_scatter(runs: pd.DataFrame, out: Path) -> None:
    d = runs[((runs["exp"] == "E1") | (runs["exp"] == "E2")) & (runs["target_C"] == 10.0) &
             (runs["variant"] == "")] if not runs.empty else runs
    models = OUR_MODELS + BASELINES
    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(7.0, 1.9))
    for j, task in enumerate(TASK_ORDER):
        ax = axes[j]
        ax.set_title(TASK_SHORT[task])
        ax.axvspan(0, 1, color="#2a78d6", alpha=0.06, lw=0)  # safe region
        ax.axvline(1.0, color=MUTED, lw=1, ls="--")
        dt = d[d["task"] == task] if len(d) else d
        if not len(dt):
            no_data(ax)
            continue
        for m in models:
            dm = dt[dt["model"] == m]
            ax.scatter(dm["C_norm"], dm["R_norm"], s=14, color=COLORS[m], marker=MARKERS[m],
                       edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xlabel("normalized cost")
    axes[0].set_ylabel("normalized reward")
    legend_below(fig, axes, models)
    fig.tight_layout()
    save(fig, out, "fig3_reward_cost_scatter")


# ------------------------------------------------------------- Figure 4
def fig_learning_curves(vals: pd.DataFrame, out: Path) -> None:
    d = vals[(vals["exp"] == "E1") & (vals["variant"].fillna("") == "")] if not vals.empty else vals
    for xcol, xlabel, name in (("step", "gradient steps", "fig4_learning_curves_steps"),
                               ("wall_time", "wall-clock (s, packed jobs: indicative)",
                                "fig4_learning_curves_wallclock")):
        fig, axes = plt.subplots(2, len(TASK_ORDER), figsize=(7.0, 3.2))
        for j, task in enumerate(TASK_ORDER):
            axes[0, j].set_title(TASK_SHORT[task])
            dt = d[d["task"] == task] if len(d) else d
            if not len(dt):
                no_data(axes[0, j])
                no_data(axes[1, j])
                continue
            axes[1, j].axhline(1.0, color=MUTED, lw=1, ls="--")
            for m in OUR_MODELS:
                dm = dt[dt["model"] == m].sort_values("step")
                axes[0, j].plot(dm[xcol], dm["R_norm"], color=COLORS[m])
                axes[1, j].plot(dm[xcol], dm["C_norm"], color=COLORS[m])
            axes[1, j].set_xlabel(xlabel)
        axes[0, 0].set_ylabel("val. normalized reward")
        axes[1, 0].set_ylabel("val. normalized cost")
        legend_below(fig, axes, OUR_MODELS)
        fig.tight_layout()
        save(fig, out, name)


# ------------------------------------------------------------- Figure 5 (E6.6)
def _eff_series(eff: pd.DataFrame, model: str, metric: str, stat: str, **kw) -> pd.Series:
    d = eff[(eff["model"] == model) & (eff["metric"] == metric) & (eff["stat"] == stat)]
    for k, v in kw.items():
        d = d[d[k] == v]
    if not len(d):
        return pd.Series(dtype=float)
    d = d.assign(K=pd.to_numeric(d["K"]))
    return d.groupby("K")["value"].mean().sort_index()


def fig_scaling(eff: pd.DataFrame, out: Path, batch: int = 256) -> None:
    e = eff[eff["weights"] == "fresh_scaling"] if not eff.empty else eff
    panels = [("scaling_train_step_s", "median", "training time / step (s)",
               dict(mode="sampler_excluded", batch=batch, precision="fp32")),
              ("train_mem_allocated", "peak", "training memory (MB)",
               dict(batch=batch, precision="fp32")),
              ("infer_latency_forward", "median", "inference time / prediction (ms)",
               dict(mode="scaling_window_full")),
              ("infer_mem_allocated", "peak", "inference memory (MB)",
               dict(mode="scaling_window"))]
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.0))
    for ax, (metric, stat, label, kw) in zip(axes.flat, panels):
        drawn = False
        for m in OUR_MODELS:
            for dev, ls in (("cuda", "-"), ("cpu", "--")):
                if len(e) == 0:
                    continue
                s = _eff_series(e, m, metric if not (dev == "cpu" and "mem" in metric)
                                else "infer_mem_rss_increase", stat, device=dev, **kw)
                s = s[s > 0]
                if len(s):
                    ax.plot(s.index, s.values, color=COLORS[m], marker=MARKERS[m], ls=ls)
                    drawn = True
        ax.set_xscale("log")
        if drawn:
            ax.set_yscale("log")
        else:
            no_data(ax)
        ax.set_xlabel("context length K (timesteps)")
        ax.set_ylabel(label)
    legend_below(fig, axes, OUR_MODELS)
    fig.text(0.99, 0.01, f"GPU solid, CPU dashed; batch {batch} for training; "
             "fresh weights (computational scaling only)", ha="right", color=MUTED, fontsize=6)
    fig.tight_layout()
    save(fig, out, "fig5_context_scaling")


# ------------------------------------------------------------- Figure 6 / 7b
def _k10(eff: pd.DataFrame, model: str, metric: str, stat: str, **kw) -> float:
    if eff.empty:
        return np.nan
    e = eff[(eff["K"].astype(str) == "10") & (eff["weights"] != "fresh_scaling")]
    d = e[(e["model"] == model) & (e["metric"] == metric) & (e["stat"] == stat)]
    for k, v in kw.items():
        d = d[d[k] == v]
    return float(d["value"].mean()) if len(d) else np.nan


def core_metrics(eff: pd.DataFrame, precision: str = "bf16") -> pd.DataFrame:
    rows = []
    for m in OUR_MODELS:
        rows.append({
            "model": m,
            "M1 training time": _k10(eff, m, "train_step_s", "median", device="cuda",
                                     mode="sampler_included", precision=precision, batch=2048),
            "M2 inference time": _k10(eff, m, "infer_latency_forward", "median", device="cuda",
                                      mode="window_full"),
            "M3 training memory": _k10(eff, m, "train_mem_allocated", "peak", device="cuda",
                                       precision=precision, batch=2048),
            "M4 inference memory": _k10(eff, m, "infer_mem_allocated", "peak", device="cuda",
                                        mode="window"),
        })
    return pd.DataFrame(rows).set_index("model")


def fig_tradeoff(runs: pd.DataFrame, eff: pd.DataFrame, out: Path) -> None:
    cm = core_metrics(eff)
    d = runs[(runs["exp"] == "E1") & (runs["target_C"] == 10.0)] if not runs.empty else runs
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4))
    any_pt = False
    for m in OUR_MODELS:
        dm = d[d["model"] == m] if len(d) else d
        if not len(dm):
            continue
        per_task = dm.groupby("task")[["R_norm", "C_norm"]].mean()
        safe = per_task[per_task["C_norm"] <= 1.0]
        r = safe["R_norm"].mean() if len(safe) else np.nan
        mem = cm.loc[m, "M3 training memory"]
        size = 30 if np.isnan(mem) else 20 + 80 * mem / np.nanmax(cm["M3 training memory"])
        for ax, x in ((axes[0], cm.loc[m, "M2 inference time"]),
                      (axes[1], cm.loc[m, "M1 training time"] * 1e5 / 3600)):
            if not (np.isnan(x) or np.isnan(r)):
                ax.scatter([x], [r], s=size, color=COLORS[m], marker=MARKERS[m],
                           edgecolors="white", linewidths=0.8, zorder=3)
                ax.annotate(DISPLAY[m], (x, r), textcoords="offset points", xytext=(5, 3),
                            color=INK, fontsize=7)
                any_pt = True
    axes[0].set_xlabel("inference time per prediction, GPU (ms)")
    axes[1].set_xlabel("training time for 100k steps (h)")
    axes[0].set_ylabel("avg. normalized reward\n(tasks where safe)")
    if not any_pt:
        for ax in axes:
            no_data(ax)
    fig.text(0.99, -0.02, "marker size = training memory", ha="right", color=MUTED, fontsize=6)
    fig.tight_layout()
    save(fig, out, "fig6_efficiency_tradeoff")


def fig_eff_bars(eff: pd.DataFrame, out: Path) -> None:
    cm = core_metrics(eff)
    rel = cm / cm.loc["cdt"] if "cdt" in cm.index else cm * np.nan
    fig, ax = plt.subplots(figsize=(5.0, 2.2))
    metrics = list(cm.columns)
    w = 0.8 / len(OUR_MODELS)
    x = np.arange(len(metrics))
    for i, m in enumerate(OUR_MODELS):
        vals = rel.loc[m].values.astype(float)
        ax.bar(x + (i - 1.5) * w, np.nan_to_num(vals, nan=0.0), width=w * 0.92,
               color=COLORS[m], label=DISPLAY[m], edgecolor="white", linewidth=1)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("relative to CDT (= 1)")
    if np.isnan(rel.values.astype(float)).all():
        no_data(ax)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    save(fig, out, "fig7b_efficiency_bars")


# ------------------------------------------------------------- Figure 7 (ablations)
def fig_ablation_deltas(runs: pd.DataFrame, out: Path) -> None:
    d = runs[runs["target_C"] == 10.0] if not runs.empty else runs
    variants = [("noaug", "w/o aug."), ("noent", "w/o entropy"), ("det", "deterministic")]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.2))
    x = np.arange(len(OUR_MODELS))
    w = 0.8 / len(variants)
    hatches = ["", "////", "...."]
    drawn = False
    for k, (v, lab) in enumerate(variants):
        dr, dc = [], []
        for m in OUR_MODELS:
            full = d[(d["exp"] == "E1") & (d["model"] == m) & (d["variant"] == "") &
                     (d["seed"].isin([0, 1, 2]))] if len(d) else d
            abl = d[(d["exp"] == "E3") & (d["model"] == m) & (d["variant"] == v)] if len(d) else d
            if len(full) and len(abl):
                f = full.groupby("task")[["R_norm", "C_norm"]].mean()
                a = abl.groupby("task")[["R_norm", "C_norm"]].mean()
                common = f.index.intersection(a.index)
                dr.append((a.loc[common, "R_norm"] - f.loc[common, "R_norm"]).mean())
                dc.append((a.loc[common, "C_norm"] - f.loc[common, "C_norm"]).mean())
                drawn = True
            else:
                dr.append(np.nan)
                dc.append(np.nan)
        for ax, vals in ((axes[0], dr), (axes[1], dc)):
            ax.bar(x + (k - 1) * w, np.nan_to_num(vals), width=w * 0.92, color=MUTED,
                   hatch=hatches[k], edgecolor="white", label=lab)
    for ax, lab in ((axes[0], "Δ normalized reward"), (axes[1], "Δ normalized cost")):
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[m] for m in OUR_MODELS], rotation=15)
        ax.set_ylabel(lab)
        if not drawn:
            no_data(ax)
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, out, "fig7_ablation_deltas")


# ------------------------------------------------------------- Figure 8 (E10)
def fig_datasets(cache: Optional[Path], out: Path) -> None:
    """Dataset cost-reward plots from the preprocessing cache (no training)."""
    from experiments.common.data import (data_cache_key, epsilon_hat, trajectory_returns)
    from experiments.common.expdefs import resolve_run_config

    def load_seq(cfg, seed=0):
        if cache is None:
            return None
        p = cache / cfg.task / f"{data_cache_key(cfg, seed, 'seq')}.npz"
        return np.load(p) if p.exists() else None

    def load_pre(cfg, seed=0):
        if cache is None:
            return None
        p = cache / cfg.task / f"{data_cache_key(cfg, seed, 'pre')}.npz"
        return dict(np.load(p)) if p.exists() else None

    # (a) every dataset with its Pareto frontier ([CDT] Fig. 7)
    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(7.0, 1.9))
    for ax, task in zip(axes, TASK_ORDER):
        ax.set_title(TASK_SHORT[task])
        z = load_seq(resolve_run_config("E1", "cdt", task, 0).data)
        if z is None:
            no_data(ax, "cache missing")
            continue
        c, r = z["orig_cost_ret"], z["orig_rew_ret"]
        ax.scatter(c, r, s=3, color="#86b6ef", lw=0)
        order = np.argsort(c)
        pf = np.maximum.accumulate(r[order])
        ax.plot(c[order], pf, color="#2a78d6", lw=1.5)
        ax.axvline(10, color=MUTED, lw=1, ls="--")
        ax.set_xlabel("cost return")
    axes[0].set_ylabel("reward return")
    fig.tight_layout()
    save(fig, out, "fig8a_datasets")

    # (b) full vs reduced (E7), eps-hat in the titles ([CDT] Fig. 6)
    tasks7 = ["OfflineAntRun-v0", "OfflineDroneRun-v0"]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0))
    for ax, task in zip(axes, tasks7):
        full = load_pre(resolve_run_config("E7", "dt", task, 0, "full").data)
        red = load_pre(resolve_run_config("E7", "dt", task, 0, "reduced").data)
        if full is None or red is None:
            no_data(ax, "cache missing")
            ax.set_title(TASK_SHORT[task])
            continue
        cf, rf = trajectory_returns(full)
        cr, rr = trajectory_returns(red)
        removed = ~np.isin(np.round(cf, 4) * 1e6 + np.round(rf, 4), np.round(cr, 4) * 1e6 + np.round(rr, 4))
        ax.scatter(cr, rr, s=3, color="#2a78d6", lw=0, label="reduced dataset")
        ax.scatter(cf[removed], rf[removed], s=3, color="#aaaaaa", lw=0, label="removed")
        ax.axvline(10, color=MUTED, lw=1, ls="--")
        ef, er = epsilon_hat(cf, rf, 10)["eps_hat"], epsilon_hat(cr, rr, 10)["eps_hat"]
        ax.set_title(f"{TASK_SHORT[task]}: ε̂ full {ef:.3f}, reduced {er:.3f}")
        ax.set_xlabel("cost return")
    axes[0].set_ylabel("reward return")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False, markerscale=3)
    fig.tight_layout()
    save(fig, out, "fig8b_reduced_datasets")

    # (c) Drone-Run with 2 % outliers: original, association lines, density filtered
    task = "OfflineDroneRun-v0"
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True)
    base = load_seq(resolve_run_config("E1", "cdt", task, 0).data)
    assoc = load_seq(resolve_run_config("E8", "cdt", task, 0, "a2_assoc").data)
    dens = load_seq(resolve_run_config("E8", "cdt", task, 0, "a2_density").data)
    titles = ["original", "2 % outliers: association", "2 % outliers: density filter"]
    for ax, z, title in zip(axes, (base, assoc, dens), titles):
        ax.set_title(title)
        if z is None:
            no_data(ax, "cache missing")
            continue
        c, r = z["orig_cost_ret"], z["orig_rew_ret"]
        ax.scatter(c, r, s=3, color="#86b6ef", lw=0)
        if z is not base and base is not None:
            key = lambda cc, rr: np.round(cc, 3) * 1e6 + np.round(rr, 3)  # noqa: E731
            out_mask = ~np.isin(key(c, r), key(base["orig_cost_ret"], base["orig_rew_ret"]))
            ax.scatter(c[out_mask], r[out_mask], s=8, color="#e34948", lw=0, label="outliers")
        if z is assoc:
            off = z["offsets"]
            aug = np.where(z["is_aug"])[0]
            for i in aug[::max(1, len(aug) // 60)]:
                src = z["aug_source"][i]
                tc, tr = z["ctg"][off[i]], z["rtg"][off[i]]
                ax.plot([c[src], tc], [r[src], tr], color="#3f3f3f", lw=0.4)
        ax.axvline(10, color=MUTED, lw=1, ls="--")
        ax.set_xlabel("cost return")
    axes[0].set_ylabel("reward return")
    fig.tight_layout()
    save(fig, out, "fig8c_outliers")


# ------------------------------------------------------------- Figure 9 (E11)
def fig_capacity(runs: pd.DataFrame, out: Path) -> None:
    d = runs[(runs["exp"] == "E11") & (runs["target_C"] == 10.0)] if not runs.empty else runs
    tasks = ["OfflineCarCircle-v0", "OfflineDroneRun-v0"]
    fig, axes = plt.subplots(2, len(tasks), figsize=(5.0, 3.4), sharex=True)
    for j, task in enumerate(tasks):
        axes[0, j].set_title(TASK_SHORT[task])
        dt = d[d["task"] == task] if len(d) else d
        if not len(dt):
            no_data(axes[0, j])
            no_data(axes[1, j])
            continue
        axes[1, j].axhline(1.0, color=MUTED, lw=1, ls="--")
        for m in ("cdt", "cdmamba", "cdconv"):
            dm = dt[dt["model"] == m]
            if not len(dm):
                continue
            for ax, y in ((axes[0, j], "R_norm"), (axes[1, j], "C_norm")):
                mu, sd = seed_curve(dm, "params_total", y)
                ax.errorbar(mu.index, mu.values, yerr=sd.values, color=COLORS[m],
                            marker=MARKERS[m], capsize=2, lw=1.5)
        axes[1, j].set_xscale("log")
        axes[1, j].set_xlabel("trainable parameters")
    axes[0, 0].set_ylabel("normalized reward")
    axes[1, 0].set_ylabel("normalized cost")
    legend_below(fig, axes, ["cdt", "cdmamba", "cdconv"])
    fig.tight_layout()
    save(fig, out, "fig9_capacity_sweep")


def main() -> None:
    import os

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", default="summary")
    p.add_argument("--out", default="summary/figures")
    p.add_argument("--cache", default=os.environ.get("CDX_CACHE", "cache"))
    a = p.parse_args()
    s, out = Path(a.summary), Path(a.out)
    runs = load(s / "runs.parquet")
    vals = load(s / "val_logs.parquet")
    eff = load(s / "efficiency.parquet")
    for df in (runs, vals):
        if not df.empty:
            df["variant"] = df["variant"].fillna("")
    fig_zero_shot(runs, out)
    fig_conflicting(runs, out)
    fig_scatter(runs, out)
    fig_learning_curves(vals, out)
    fig_scaling(eff, out)
    fig_tradeoff(runs, eff, out)
    fig_eff_bars(eff, out)
    fig_ablation_deltas(runs, out)
    fig_datasets(Path(a.cache) if a.cache else None, out)
    fig_capacity(runs, out)


if __name__ == "__main__":
    main()
