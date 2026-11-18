"""Publication figures. Arial (live text in SVG), no frame boxes, grey axes, outward ticks,
large fonts, legends outside data areas. Every figure is skipped with a message if the
results it needs are missing, so figures can be rebuilt after any subset of stages."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from .data import apply_qc, by_group
from .likelihood import Model, cell_quad, gls_intercept, f_mean

GREY = "#7f7f7f"
LABEL = {"20200528_RFP": "May RFP", "20200528_YFP": "May YFP", "20201217_CFP": "Dec CFP",
         "20201217_GFP": "Dec GFP", "20201217_YFP": "Dec YFP"}
COL = {"20200528_RFP": "#D55E00", "20200528_YFP": "#E69F00", "20201217_CFP": "#0072B2",
       "20201217_GFP": "#009E73", "20201217_YFP": "#CC79A7"}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "mathtext.default": "regular",
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY, "axes.linewidth": 0.9,
        "xtick.color": GREY, "ytick.color": GREY, "xtick.labelcolor": "black", "ytick.labelcolor": "black",
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False})


def lab(g):
    return LABEL.get(g, g)


def col(g):
    return COL.get(g, "#555555")


def save(fig, out, name):
    for ext in ["svg", "pdf", "png"]:
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def letter(ax, s):
    ax.text(-0.18, 1.06, s, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")


def _need(paths):
    miss = [str(p) for p in paths if not Path(p).exists()]
    if miss:
        print(f"  skipped (missing: {', '.join(miss)})")
        return False
    return True


def make_all(cfg, cells, R, out):
    style()
    out.mkdir(parents=True, exist_ok=True)
    pq = f"qc{cfg['primary_qc']:g}"
    groups = sorted(by_group(cells).keys())
    fits_p = R / "main/hierarchical_fits.csv"
    fits = pd.read_csv(fits_p) if fits_p.exists() else None
    import json
    theta = json.loads((R / "main/theta.json").read_text()) if (R / "main/theta.json").exists() else None
    rng = np.random.default_rng(1)

    # ---------- Figure 1 ----------
    print("Fig 1")
    if _need([R / "covariates/cell_estimates.csv", R / "main/theta.json"]):
        eb = pd.read_csv(R / "covariates/cell_estimates.csv")
        kept, _ = apply_qc(cells, cfg["primary_qc"]); K = {c.cell: c for c in kept}
        G = sorted(eb.group.unique())
        fig, axes = plt.subplots(1, len(G), figsize=(2.9 * len(G), 3.4), sharey=True, squeeze=False)
        for ax, g in zip(axes[0], G):
            x = np.array(theta[f"{pq}|{g}|M1_random"])
            om, phi = float(np.exp(x[1])), float(np.exp(x[3]))
            e = eb[eb.group == g].sort_values("eb_gamma")
            for _, r in e.iterrows():
                c = K[r.cell]; a = gls_intercept(c.t, c.y, r.eb_gamma, om, phi)
                ax.plot(c.t, c.y - a, color="#c8c8c8", lw=0.6, zorder=1)
            for frac, s in zip([0.1, 0.5, 0.9], [0.45, 0.75, 1.0]):
                r = e.iloc[int(frac * (len(e) - 1))]; c = K[r.cell]
                a = gls_intercept(c.t, c.y, r.eb_gamma, om, phi)
                tt = np.linspace(0, c.t.max(), 200)
                ax.plot(tt, -r.eb_gamma * f_mean(tt, om), color=col(g), lw=2, alpha=s, zorder=3)
                ax.plot(c.t, c.y - a, "o", ms=4, color=col(g), alpha=s, mec="none", zorder=4)
            ax.set_title(f"{lab(g)} (n = {len(e)})"); ax.set_xlabel("Time since starvation (h)")
        axes[0][0].set_ylabel("log F, intercept removed")
        save(fig, out, "Fig1_trajectories")

    # ---------- Figure 2 ----------
    print("Fig 2")
    if _need([R / "cellwise/cellwise.csv"]):
        cw = pd.read_csv(R / "cellwise/cellwise.csv")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
        for i, g in enumerate(groups):
            v = cw.loc[cw.group == g, "dAICc_M1_minus_M0"].dropna()
            a1.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color="black"),
                       boxprops=dict(color=GREY), whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
            a1.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=col(g), alpha=0.6, lw=0, zorder=3)
        a1.axhline(0, color=GREY, lw=0.8, ls="--")
        a1.set_xticks(range(len(groups))); a1.set_xticklabels([lab(g) for g in groups], rotation=45, ha="right")
        a1.set_ylabel("ΔAICc (M1 − M0)"); letter(a1, "A")
        for g in groups:
            v = cw[cw.group == g]
            a2.scatter(v.holdout_RMSE_M0, v.holdout_RMSE_M1, s=12, color=col(g), alpha=0.7, lw=0, label=lab(g))
        lo = np.nanmin(cw[["holdout_RMSE_M0", "holdout_RMSE_M1"]].values) * 0.8
        hi = np.nanmax(cw[["holdout_RMSE_M0", "holdout_RMSE_M1"]].values) * 1.2
        a2.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--")
        a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlim(lo, hi); a2.set_ylim(lo, hi)
        a2.set_xlabel("Hold-out RMSE, constant rate (M0)"); a2.set_ylabel("Hold-out RMSE, relaxing rate (M1)")
        a2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.2); letter(a2, "B")
        fig.tight_layout(); save(fig, out, "Fig2_model_comparison")

    # ---------- Figure 3 ----------
    print("Fig 3")
    if _need([R / "cellwise/cellwise.csv", R / "synthetic/synthetic_summary.csv", fits_p,
              R / "bootstrap/parametric_bootstrap_summary.csv"]):
        cw = pd.read_csv(R / "cellwise/cellwise.csv")
        sy = pd.read_csv(R / "synthetic/synthetic_summary.csv"); sy = sy[sy.density == 1].set_index("group")
        bs = pd.read_csv(R / "bootstrap/parametric_bootstrap_summary.csv")
        fr = cw.groupby("group")[["gamma_profile_boundary", "omega_profile_boundary"]].mean()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2)); x = np.arange(len(groups))
        a1.bar(x - 0.27, [100 * fr.loc[g, "gamma_profile_boundary"] for g in groups], 0.26, color="#bdbdbd", label="γ$_0$, data")
        a1.bar(x, [100 * fr.loc[g, "omega_profile_boundary"] for g in groups], 0.26, color="#3a3a3a", label="ω, data")
        a1.bar(x + 0.27, [100 * sy.loc[g, "frac_omega_profile_boundary"] if g in sy.index else 0 for g in groups], 0.26,
               color="white", edgecolor="#3a3a3a", hatch="///", lw=0.8, label="ω, synthetic")
        a1.set_xticks(x); a1.set_xticklabels([lab(g) for g in groups], rotation=45, ha="right")
        a1.set_ylabel("Cells with boundary-limited\nprofile interval (%)")
        a1.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0)); letter(a1, "A")
        for i, g in enumerate(groups):
            for dx, qc, filled in [(-0.15, "all", False), (0.15, pq, True)]:
                f = fits[(fits.group == g) & (fits.qc == qc) & (fits.model == "M1_random")]
                b = bs[(bs.group == g) & (bs.qc == qc)]
                if f.empty or b.empty:
                    continue
                o = float(f.omega.iloc[0])
                kw = dict(color=col(g)) if filled else dict(mfc="white", mec=col(g), ecolor=col(g))
                a2.errorbar(i + dx, o, yerr=[[max(0, o - b.omega_lo.iloc[0])], [max(0, b.omega_hi.iloc[0] - o)]],
                            fmt="o", ms=7, capsize=3, **kw)
            if g in cfg["published_gamma_dec"]:
                a2.plot(i, cfg["published_gamma_dec"][g], marker="D", color="black", ms=6, ls="none")
        a2.set_xticks(x); a2.set_xticklabels([lab(g) for g in groups], rotation=45, ha="right")
        a2.set_ylabel("Relaxation rate ω (h$^{-1}$)")
        a2.legend([Line2D([], [], marker="o", ls="none", mfc="white", mec="black", ms=7),
                   Line2D([], [], marker="o", ls="none", color="black", ms=7),
                   Line2D([], [], marker="D", ls="none", color="black", ms=6)],
                  ["All cells", "After QC", "Published calibration"], loc="upper left", bbox_to_anchor=(1.0, 1.0))
        letter(a2, "B"); fig.tight_layout(); save(fig, out, "Fig3_omega_identifiability")

    # ---------- Figure 4 ----------
    print("Fig 4")
    if _need([fits_p, R / "bootstrap/parametric_bootstrap_summary.csv", R / "covariates/cell_estimates.csv"]):
        bs = pd.read_csv(R / "bootstrap/parametric_bootstrap_summary.csv")
        eb = pd.read_csv(R / "covariates/cell_estimates.csv")
        fig = plt.figure(figsize=(13, 4.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.1], wspace=0.45)
        a1, a2, a3 = [fig.add_subplot(gs[0, i]) for i in range(3)]
        for i, g in enumerate(groups):
            for dx, qc, filled in [(-0.15, "all", False), (0.15, pq, True)]:
                f = fits[(fits.group == g) & (fits.qc == qc) & (fits.model == "M1_random")]
                b = bs[(bs.group == g) & (bs.qc == qc)]
                if f.empty or b.empty:
                    continue
                c = float(f.CV.iloc[0])
                kw = dict(color=col(g)) if filled else dict(mfc="white", mec=col(g), ecolor=col(g))
                a1.errorbar(i + dx, c, yerr=[[max(0, c - b.CV_lo.iloc[0])], [max(0, b.CV_hi.iloc[0] - c)]],
                            fmt="o", ms=7, capsize=3, **kw)
        a1.plot([], [], "o", mfc="white", mec="black", label="All cells"); a1.plot([], [], "o", color="black", label="After QC")
        a1.set_xticks(range(len(groups))); a1.set_xticklabels([lab(g) for g in groups], rotation=45, ha="right")
        a1.set_ylabel("Between-cell CV of γ$_0$"); a1.set_ylim(bottom=0)
        a1.legend(loc="upper left", bbox_to_anchor=(0.0, 1.12), ncol=2, columnspacing=0.8, handletextpad=0.2)
        letter(a1, "A")
        for i, g in enumerate(groups):
            e = eb[eb.group == g]
            if e.empty:
                continue
            a2.scatter(i + rng.uniform(-0.2, 0.2, len(e)), e.eb_gamma, s=10, color=col(g), alpha=0.65, lw=0)
            f = fits[(fits.group == g) & (fits.qc == pq) & (fits.model == "M1_random")].iloc[0]
            q = np.exp(f.mu + f.scale * np.array([-1.96, 0, 1.96]))
            a2.plot([i - 0.3, i + 0.3], [q[1]] * 2, color="black", lw=1.5); a2.plot([i, i], [q[0], q[2]], color="black", lw=0.9)
        a2.set_yscale("log"); a2.set_xticks(range(len(groups))); a2.set_xticklabels([lab(g) for g in groups], rotation=45, ha="right")
        a2.set_ylabel("Cell-level γ$_0$ (h$^{-1}$)"); letter(a2, "B")
        zs = lambda v: (v - v.mean()) / v.std()
        allx, ally = [], []
        for g in groups:
            e = eb[eb.group == g]
            if len(e) < 3:
                continue
            zx, zy = zs(e.length_um), zs(e.eb_log_gamma); allx += list(zx); ally += list(zy)
            a3.scatter(zx, zy, s=10, color=col(g), alpha=0.6, lw=0, label=lab(g))
        k = np.polyfit(allx, ally, 1); xx = np.linspace(min(allx), max(allx), 10)
        a3.plot(xx, np.polyval(k, xx), color="black", lw=1.2)
        a3.set_xlabel("Cell length (z-score within group)"); a3.set_ylabel("log γ$_0$ (z-score within group)")
        a3.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.1, markerscale=1.6); letter(a3, "C")
        save(fig, out, "Fig4_heterogeneity")

    # ---------- Figure 5: robustness of the heterogeneity estimate ----------
    print("Fig 5")
    paths = [fits_p, R / "bootstrap/parametric_bootstrap_summary.csv"]
    if _need(paths):
        bs = pd.read_csv(R / "bootstrap/parametric_bootstrap_summary.csv")
        sc = pd.read_csv(R / "structure/structure_comparison.csv") if (R / "structure/structure_comparison.csv").exists() else None
        lb = pd.read_csv(R / "bootstrap/lane_bootstrap_summary.csv") if (R / "bootstrap/lane_bootstrap_summary.csv").exists() else None
        bl = pd.read_csv(R / "bleaching/bleaching_refits.csv") if (R / "bleaching/bleaching_refits.csv").exists() else None
        fig, ax = plt.subplots(figsize=(9, 4.4))
        offs = {"Primary": -0.3, "Lane CI": -0.15, "Within-lane": 0.0, "Gamma": 0.15, "Bleach": 0.3}
        for i, g in enumerate(groups):
            f = fits[(fits.group == g) & (fits.qc == pq) & (fits.model == "M1_random")]
            if f.empty:
                continue
            c = float(f.CV.iloc[0]); b = bs[(bs.group == g) & (bs.qc == pq)]
            if not b.empty:
                ax.errorbar(i + offs["Primary"], c, yerr=[[c - b.CV_lo.iloc[0]], [b.CV_hi.iloc[0] - c]], fmt="o", color=col(g), ms=7, capsize=3)
            if lb is not None:
                b2 = lb[(lb.group == g) & (lb.qc == pq)]
                if not b2.empty:
                    ax.errorbar(i + offs["Lane CI"], c, yerr=[[max(0, c - b2.CV_lo.iloc[0])], [max(0, b2.CV_hi.iloc[0] - c)]],
                                fmt="s", mfc="white", mec=col(g), ecolor=col(g), ms=6, capsize=3)
            if sc is not None:
                r = sc[(sc.group == g) & (sc.qc == pq) & (sc.comparison == "lane_vs_cell_only")]
                if not r.empty:
                    ax.plot(i + offs["Within-lane"], r.CV_within.iloc[0], "^", color=col(g), ms=7)
                r = sc[(sc.group == g) & (sc.qc == pq) & (sc.comparison == "gamma_vs_lognormal")]
                if not r.empty:
                    ax.plot(i + offs["Gamma"], r.CV_alt.iloc[0], "D", mfc="white", mec=col(g), ms=6)
            if bl is not None:
                r = bl[bl.group == g]
                if not r.empty:
                    ax.plot(i + offs["Bleach"], r.CV_corrected.iloc[0], "v", color=col(g), ms=7)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([lab(g) for g in groups], rotation=45, ha="right")
        ax.set_ylabel("Between-cell CV of γ$_0$"); ax.set_ylim(bottom=0)
        ax.legend([Line2D([], [], marker="o", ls="none", color="black"),
                   Line2D([], [], marker="s", ls="none", mfc="white", mec="black"),
                   Line2D([], [], marker="^", ls="none", color="black"),
                   Line2D([], [], marker="D", ls="none", mfc="white", mec="black"),
                   Line2D([], [], marker="v", ls="none", color="black")],
                  ["Primary fit, parametric 95% CI", "Lane-cluster bootstrap 95% CI", "Within-lane CV (lane model)",
                   "Gamma random effect", "Bleaching corrected"],
                  loc="upper left", bbox_to_anchor=(1.0, 1.0))
        save(fig, out, "Fig5_robustness")

    # ---------- Figure S1: excluded trajectories ----------
    print("Fig S1")
    _, dropped = apply_qc(cells, cfg["primary_qc"])
    if dropped:
        dropped = sorted(dropped, key=lambda c: (c.group, c.max_jump))
        n = len(dropped); ncol = 3; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.7 * nrow), sharex=True, squeeze=False)
        for ax, c in zip(axes.flat, dropped):
            j = int(np.argmax(np.abs(np.diff(c.y))))
            ax.plot(c.t, c.y, "-o", color=col(c.group), ms=4, lw=1.2); ax.plot(c.t[j:j + 2], c.y[j:j + 2], color="black", lw=2.2)
            ax.set_title(f"{lab(c.group)}, max |Δlog F| = {c.max_jump:.2f}", fontsize=10.5)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("Time since starvation (h)")
        for ax in axes[:, 0]:
            ax.set_ylabel("log F")
        fig.tight_layout(); save(fig, out, "FigS1_excluded_trajectories")

    # ---------- Figure S2: fixed vs adaptive quadrature ----------
    print("Fig S2")
    g = cfg["quadrature_demo_group"]
    if theta and f"all|{g}|M1_random" in theta:
        cs = by_group(cells)[g]
        x = np.array(theta[f"all|{g}|M1_random"]); m = Model(cs, nodes=15); p = m.unpack(x)
        qs = m.quads(p)
        sds = np.linspace(0.2, 0.8, 49)

        def fixed(sd, nq):
            xg, wg = hermgauss(nq); lw = np.log(wg) - 0.5 * np.log(np.pi); gs = np.exp(p["mu"] + np.sqrt(2) * sd * xg)
            tot = 0.0
            for c_, A, B, D in qs:
                tot += logsumexp(c_ - (A + 2 * gs * B + gs * gs * D) / (2 * p["sigma"] ** 2) + lw)
            return -tot
        cvs = np.sqrt(np.expm1(sds ** 2))
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(cvs, [fixed(s, 15) for s in sds], color="#E69F00", lw=1.4, label="Fixed nodes, 15")
        ax.plot(cvs, [fixed(s, 31) for s in sds], color="#56B4E9", lw=1.4, label="Fixed nodes, 31")
        xx = x.copy()
        ax.plot(cvs, [m.nll(np.r_[xx[:-1], np.log(s)]) for s in sds], color="black", lw=2, label="Adaptive, 15")
        ax.set_xlabel("Between-cell CV of γ$_0$"); ax.set_ylabel("Negative log-likelihood")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0)); save(fig, out, "FigS2_quadrature")
    else:
        print("  skipped (needs main fit of the demo group)")

    # ---------- Figure S3: simulation-based validation ----------
    print("Fig S3")
    if _need([R / "sbc/sbc_replicates.csv"]):
        d = pd.read_csv(R / "sbc/sbc_replicates.csv")
        main = d[(d.artefacts == True) & (d.misspec == False)]
        designs = sorted(main.design.unique())
        fig, axes = plt.subplots(1, len(designs) + 1, figsize=(4.2 * (len(designs) + 1), 4), squeeze=False)
        for ax, g in zip(axes[0], designs):
            x = main[main.design == g]
            for tau, mk in [(0.0, "o"), (max(x.tau), "s")]:
                y = x[x.tau == tau]
                if y.empty:
                    continue
                jit = rng.uniform(-0.015, 0.015, len(y))
                ax.scatter(y.true_CV + jit - 0.02, y.CV_noqc, s=12, marker=mk, facecolor="white", edgecolor=GREY, lw=0.8)
                ax.scatter(y.true_CV + jit + 0.02, y.CV_qc, s=12, marker=mk, color=col(g), lw=0)
            lim = [0, max(1.2, float(np.nanmax(x[["CV_qc", "CV_noqc"]].values)) * 1.05)]
            ax.plot(lim, lim, color=GREY, ls="--", lw=0.8); ax.set_xlim(0, max(x.true_CV) + 0.15); ax.set_ylim(lim)
            ax.set_title(f"Design: {lab(g)}"); ax.set_xlabel("True CV of γ$_0$"); ax.set_ylabel("Estimated CV")
        ax = axes[0][-1]
        pub = d[d.omega_source == "published"]
        if not pub.empty:
            for i, g in enumerate(sorted(pub.design.unique())):
                y = pub[pub.design == g]
                ax.scatter(i + rng.uniform(-0.15, 0.15, len(y)), y.omega_qc / y.true_omega, s=14, color=col(g), lw=0)
            ax.axhline(1, color=GREY, ls="--", lw=0.8)
            gg = sorted(pub.design.unique())
            ax.set_xticks(range(len(gg))); ax.set_xticklabels([lab(g) for g in gg], rotation=45, ha="right")
            ax.set_ylabel("Estimated / true ω")
        fig.legend([Line2D([], [], marker="o", ls="none", mfc="white", mec=GREY),
                    Line2D([], [], marker="o", ls="none", color="black"),
                    Line2D([], [], marker="s", ls="none", color="black")],
                   ["No QC", "With QC", "Lane effect present"], loc="upper left", bbox_to_anchor=(1.0, 0.95))
        fig.tight_layout(); save(fig, out, "FigS3_simulation_validation")
