"""Analysis stages. Each stage reads inputs, writes results/<stage>/ and is safe to re-run.
Worker functions are module level so they work with multiprocessing on Windows."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr, f_oneway

from .data import load_cells, apply_qc, by_group, Cell, read_table
from .likelihood import Model, eb_log_gamma, f_mean
from .fit import fit, summary_row
from .cellwise import analyse_cell, synthetic_task, ou_noise
from .util import pmap, Checkpoint, write_json, read_json, log


def _qc_label(q):
    return "all" if (q is None or q <= 0) else f"qc{q:g}"


# ======================================================================
# 1. cellwise: M0 vs M1, hold-out, profile likelihood
# ======================================================================
def stage_cellwise(cfg, cells, out, jobs):
    rows = pmap(analyse_cell, cells, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(out / "cellwise.csv", index=False)
    s = df.groupby("group").agg(
        cells=("cell", "size"), median_points=("n_points", "median"),
        median_dAICc=("dAICc_M1_minus_M0", "median"),
        frac_M1_AICc=("dAICc_M1_minus_M0", lambda v: (v < 0).mean()),
        frac_M1_holdout=("holdout_RMSE_M1", lambda v: np.nan),
        median_RMSE_M0=("holdout_RMSE_M0", "median"), median_RMSE_M1=("holdout_RMSE_M1", "median"),
        frac_gamma_boundary=("gamma_profile_boundary", "mean"),
        frac_omega_boundary=("omega_profile_boundary", "mean")).reset_index()
    s["frac_M1_holdout"] = df.groupby("group").apply(
        lambda x: (x.holdout_RMSE_M1 < x.holdout_RMSE_M0).mean()).values
    s.to_csv(out / "cellwise_summary.csv", index=False)
    return s


# ======================================================================
# 2. synthetic identifiability benchmark (OU noise from the primary fit)
# ======================================================================
def stage_synthetic(cfg, cells, out, jobs, main_params):
    G = by_group(cells)
    sc = cfg["synthetic"]
    tasks = []
    for i, (g, cs) in enumerate(G.items()):
        p = main_params[g]
        scheds = [c.t for c in cs[: sc["schedules_per_group"]]]
        tasks.append((g, scheds, p["sigma"], p["phi"], sc["gamma_true"], sc["omega_true"],
                      sc["densities"], sc["reps"], cfg["seed"] + 101 * i))
    rows = [r for res in pmap(synthetic_task, tasks, jobs) for r in res]
    df = pd.DataFrame(rows)
    df.to_csv(out / "synthetic_recovery.csv", index=False)
    s = df.groupby(["group", "density"]).agg(
        fits=("rep", "size"), median_gamma_rel_err=("gamma_rel_err", "median"),
        median_omega_rel_err=("omega_rel_err", "median"),
        frac_omega_profile_boundary=("omega_profile_boundary", "mean"),
        frac_gamma_profile_boundary=("gamma_profile_boundary", "mean")).reset_index()
    s.to_csv(out / "synthetic_summary.csv", index=False)
    return s


# ======================================================================
# 3. main hierarchical fits: lognormal single level, M1 vs M0 vs common gamma,
#    several QC thresholds, optional fixed omega, leave-one-out
# ======================================================================
def _fit_group_task(args):
    g, cs, kind, opts, seed = args
    m = Model(cs, **opts)
    res = fit(m, starts=kind["starts"], seed=seed)
    row = summary_row(g, res, kind["label"])
    row.update(qc=kind["qc"])
    return row, res["x"].tolist()


def stage_main(cfg, cells, out, jobs):
    rows, params = [], {}
    tasks = []
    for q in cfg["qc_thresholds"]:
        kept, dropped = apply_qc(cells, q)
        G = by_group(kept)
        for i, (g, cs) in enumerate(G.items()):
            base = dict(nodes=cfg["nodes"])
            specs = [("M1_random", dict(random=True)), ("M1_common", dict(random=False)),
                     ("M0_random", dict(random=True, omega_fixed=0.0))]
            if q == cfg["primary_qc"] and g in cfg.get("fix_omega", {}):
                specs.append(("M1_random_omega_fixed", dict(random=True, omega_fixed=cfg["fix_omega"][g])))
            for lab, o in specs:
                tasks.append((g, cs, dict(label=lab, qc=_qc_label(q), starts=cfg["starts"]), {**base, **o},
                              cfg["seed"] + 7 * i))
    res = pmap(_fit_group_task, tasks, jobs)
    for row, x in res:
        rows.append(row)
        params[f"{row['qc']}|{row['group']}|{row['model']}"] = x
    df = pd.DataFrame(rows)
    df.to_csv(out / "hierarchical_fits.csv", index=False)
    write_json(out / "theta.json", params)
    # model comparisons
    comp = []
    for (qc, g), x in df.groupby(["qc", "group"]):
        r = x.set_index("model")
        m1, m0, cm = r.loc["M1_random"], r.loc["M0_random"], r.loc["M1_common"]
        lr = max(0.0, 2 * (m0.nll - m1.nll))
        lr_re = max(0.0, 2 * (cm.nll - m1.nll))
        comp.append(dict(qc=qc, group=g, n_cells=int(m1.n_cells), CV=m1.CV, omega=m1.omega,
                         dAICc_M1_minus_M0=m1.AICc - m0.AICc, LR_M1_vs_M0=lr,
                         p_M1_vs_M0=0.5 * chi2.sf(lr, 1) if lr > 0 else 0.5,
                         dAICc_random_minus_common=m1.AICc - cm.AICc, LR_random_vs_common=lr_re,
                         p_random_vs_common=0.5 * chi2.sf(lr_re, 1) if lr_re > 0 else 0.5,
                         starts_agree=int(m1.starts_agree), starts=int(m1.starts)))
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "model_comparison.csv", index=False)
    return df, comp, params


def _loo_task(args):
    g, cs, idx, opts, starts, seed = args
    red = [c for j, c in enumerate(cs) if j != idx]
    res = fit(Model(red, **opts), starts=starts, seed=seed)
    return dict(group=g, removed_cell=cs[idx].cell, CV=res["cv"], omega=res["params"]["omega"])


def stage_loo(cfg, cells, out, jobs, theta):
    rows = []
    for q in [0.0, cfg["primary_qc"]]:
        kept, _ = apply_qc(cells, q)
        tasks = []
        for i, (g, cs) in enumerate(by_group(kept).items()):
            m = Model(cs, nodes=cfg["nodes"])
            x = np.array(theta[f"{_qc_label(q)}|{g}|M1_random"])
            ll = m.per_cell(x)
            for idx in np.argsort(ll)[: cfg["loo_cells"]]:
                tasks.append((g, cs, int(idx), dict(nodes=cfg["nodes"]), max(3, cfg["starts"] // 2), cfg["seed"] + i))
        for r in pmap(_loo_task, tasks, jobs):
            r["qc"] = _qc_label(q)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "leave_one_out.csv", index=False)
    return df


# ======================================================================
# 4. lane model and 5. gamma random effect (model-structure sensitivity)
# ======================================================================
def stage_structure(cfg, cells, out, jobs, which):
    tasks = []
    for q in [cfg["primary_qc"], 0.0]:
        kept, _ = apply_qc(cells, q)
        for i, (g, cs) in enumerate(by_group(kept).items()):
            if which == "lane":
                opts = dict(random=True, lane=True, nodes=cfg["nodes"], lane_nodes=cfg["lane_nodes"])
            else:
                opts = dict(random=True, dist="gamma", nodes=cfg["gamma_nodes"])
            tasks.append((g, cs, dict(label=which, qc=_qc_label(q), starts=cfg["starts_structure"]), opts,
                          cfg["seed"] + 13 * i))
    res = pmap(_fit_group_task, tasks, jobs)
    df = pd.DataFrame([r for r, _ in res])
    if which == "lane":
        # total and within-lane CV of gamma_0 under the lane model
        df["CV_within_lane"] = np.sqrt(np.expm1(df.scale ** 2))
        df["CV_total"] = np.sqrt(np.expm1(df.scale ** 2 + df.tau ** 2))
        df["frac_var_lane"] = df.tau ** 2 / (df.tau ** 2 + df.scale ** 2)
    df.to_csv(out / f"{which}_fits.csv", index=False)
    return df


def compare_structures(main_df, lane_df, gamma_df, out):
    rows = []
    for _, r in lane_df.iterrows():
        b = main_df[(main_df.group == r.group) & (main_df.qc == r.qc) & (main_df.model == "M1_random")].iloc[0]
        lr = max(0.0, 2 * (b.nll - r.nll))
        rows.append(dict(qc=r.qc, group=r.group, comparison="lane_vs_cell_only", CV_single=b.CV,
                         CV_within=r.CV_within_lane, CV_total=r.CV_total, frac_var_lane=r.frac_var_lane,
                         dAICc=r.AICc - b.AICc, LR=lr, p=0.5 * chi2.sf(lr, 1) if lr > 0 else 0.5,
                         omega_single=b.omega, omega_alt=r.omega))
    for _, r in gamma_df.iterrows():
        b = main_df[(main_df.group == r.group) & (main_df.qc == r.qc) & (main_df.model == "M1_random")].iloc[0]
        rows.append(dict(qc=r.qc, group=r.group, comparison="gamma_vs_lognormal", CV_single=b.CV,
                         CV_alt=r.CV, dAICc=r.AICc - b.AICc, omega_single=b.omega, omega_alt=r.omega))
    df = pd.DataFrame(rows)
    df.to_csv(out / "structure_comparison.csv", index=False)
    return df


# ======================================================================
# 6. parametric bootstrap and 7. lane-cluster bootstrap (resumable)
# ======================================================================
def _simulate(cs, p, rng, lane_sd=0.0, mean="exp"):
    """Simulate trajectories on the observed schedules. mean='exp': gamma(t)=gamma0 exp(-omega t);
    mean='hyper': gamma(t)=gamma0/(1+omega t) (misspecified alternative)."""
    lanes = sorted(set(c.lane for c in cs))
    leff = {l: rng.normal(0, lane_sd) if lane_sd > 0 else 0.0 for l in lanes}
    out = []
    for c in cs:
        u = p["mu"] + leff[c.lane] + p["scale"] * rng.standard_normal()
        shape = f_mean(c.t, p["omega"]) if mean == "exp" else np.log1p(p["omega"] * c.t) / p["omega"]
        y = 10.0 - np.exp(u) * shape + ou_noise(c.t, p["sigma"], p["phi"], rng)
        out.append(Cell(c.group, c.lane, c.cell, c.t, y, c.length_um, 0.0))
    return out


def _boot_task(args):
    key, kind, cs, p, x0, starts, nodes, seed = args
    rng = np.random.default_rng(seed)
    if kind == "parametric":
        data = _simulate(cs, p, rng)
    else:  # lane cluster: resample lanes with replacement, relabel duplicates
        lanes = sorted(set(c.lane for c in cs))
        pick = rng.choice(len(lanes), len(lanes), replace=True)
        data = []
        for k, li in enumerate(pick):
            for c in cs:
                if c.lane == lanes[li]:
                    data.append(Cell(c.group, f"{lanes[li]}#{k}", c.cell, c.t, c.y, c.length_um, c.max_jump))
    res = fit(Model(data, nodes=nodes), starts=starts, seed=seed, init=x0)
    q = res["params"]
    return dict(key=key, kind=kind, group=cs[0].group, CV=res["cv"], omega=q["omega"], sigma=q["sigma"],
                phi=q["phi"], gamma_median=float(np.exp(q["mu"])), nll=res["nll"],
                starts_agree=res["starts_agree"], omega_at_lower=bool(q["omega"] < 1.05e-6))


def stage_bootstrap(cfg, cells, out, jobs, theta, kind, reps, qcs, logfile=None):
    ck = Checkpoint(out / f"{kind}_bootstrap_replicates.csv")
    tasks = []
    for q in qcs:
        kept, _ = apply_qc(cells, q)
        for i, (g, cs) in enumerate(by_group(kept).items()):
            x0 = np.array(theta[f"{_qc_label(q)}|{g}|M1_random"])
            p = Model(cs).unpack(x0)
            for r in range(reps):
                key = f"{_qc_label(q)}|{g}|{r}"
                if key in ck.done:
                    continue
                tasks.append((key, kind, cs, p, x0, cfg["boot_starts"], cfg["nodes"],
                              cfg["seed"] * 1000 + 100000 * i + r + (0 if kind == "parametric" else 7777777)))
    log(f"{kind} bootstrap: {len(tasks)} replicates to run ({len(ck.done)} already done)", logfile)
    buf = []
    t0 = time.time()

    def on_res(i, res):
        buf.append(res)
        if len(buf) >= 10:
            ck.add(buf); buf.clear()
            log(f"  {kind}: {len(ck.done)} done, {time.time() - t0:.0f} s", logfile)
    pmap(_boot_task, tasks, jobs, on_result=on_res)
    ck.add(buf)
    df = ck.frame()
    df["qc"] = df.key.str.split("|").str[0]
    s = df.groupby(["qc", "group"]).agg(
        n=("CV", "size"), CV_median=("CV", "median"), CV_lo=("CV", lambda v: v.quantile(0.025)),
        CV_hi=("CV", lambda v: v.quantile(0.975)), omega_lo=("omega", lambda v: v.quantile(0.025)),
        omega_hi=("omega", lambda v: v.quantile(0.975)), frac_omega_at_lower=("omega_at_lower", "mean"),
        frac_starts_disagree=("starts_agree", lambda v: (v < cfg["boot_starts"]).mean())).reset_index()
    s.to_csv(out / f"{kind}_bootstrap_summary.csv", index=False)
    return s


# ======================================================================
# 8. photobleaching
# ======================================================================
def stage_bleaching(cfg, cells, out, jobs, theta):
    bc = cfg["bleaching"]
    b = read_table(cfg["bleaching_file"])
    need = ["date", "Fluo_sig", "idx", "time_sec", "signal", "cell"]
    miss = [c for c in need if c not in b.columns]
    if miss:
        raise ValueError(f"bleaching file missing columns {miss}")
    b = b[(pd.to_numeric(b.signal, errors="coerce") > 0) & (b.idx == bc["dense_idx"])]
    rates = []
    for (date, fl), g in b.groupby(["date", "Fluo_sig"]):
        ks, dts = [], []
        for _, x in g.groupby("cell"):
            x = x.sort_values("time_sec")
            if len(x) < 5:
                continue
            t = x.time_sec.to_numpy(float) / 3600; y = np.log(x.signal.to_numpy(float))
            ks.append(-np.polyfit(t, y, 1)[0]); dts.append(np.median(np.diff(t)))
        if ks:
            k, dt = float(np.median(ks)), float(np.median(dts))
            rates.append(dict(date=str(date), Fluo=fl, cells=len(ks), dense_interval_h=dt, decay_rate_dense=k,
                              loss_per_exposure=max(0.0, (k - bc["degradation_reference"]) * dt)))
    rates = pd.DataFrame(rates)
    rates.to_csv(out / "bleaching_calibration.csv", index=False)
    kept, _ = apply_qc(cells, cfg["primary_qc"])
    G = by_group(kept)
    tasks, meta = [], []
    for i, (g, cs) in enumerate(G.items()):
        date, fl = g.split("_")
        src = bc["calibration_date"].get(date, date)
        r = rates[(rates.date == str(src)) & (rates.Fluo == fl)]
        if r.empty:
            continue
        interval = float(np.median([np.median(np.diff(c.t)) for c in cs]))
        kb = float(r.loss_per_exposure.iloc[0]) / interval
        corrected = [Cell(c.group, c.lane, c.cell, c.t, c.y + kb * c.t, c.length_um, c.max_jump) for c in cs]
        tasks.append((g, corrected, dict(label="bleach_corrected", qc=_qc_label(cfg["primary_qc"]),
                                          starts=cfg["starts"]), dict(nodes=cfg["nodes"]), cfg["seed"] + i))
        meta.append(dict(group=g, calibration_date=src, upper_bound=str(src) != date,
                         imaging_interval_h=interval, k_bleach_per_h=kb))
    res = pmap(_fit_group_task, tasks, jobs)
    df = pd.DataFrame([{**m, "CV_corrected": r["CV"], "omega_corrected": r["omega"]} for m, (r, _) in zip(meta, res)])
    df.to_csv(out / "bleaching_refits.csv", index=False)
    return df


# ======================================================================
# 9. cell-level estimates and covariates
# ======================================================================
def stage_covariates(cfg, cells, out, theta):
    kept, _ = apply_qc(cells, cfg["primary_qc"])
    rows = []
    for g, cs in by_group(kept).items():
        p = Model(cs).unpack(np.array(theta[f"{_qc_label(cfg['primary_qc'])}|{g}|M1_random"]))
        for c in cs:
            m, s = eb_log_gamma(c, p)
            rows.append(dict(group=g, lane=c.lane, cell=c.cell, eb_log_gamma=m, eb_sd=s, eb_gamma=np.exp(m),
                             length_um=c.length_um, logF0=float(c.y[0])))
    eb = pd.DataFrame(rows)
    eb.to_csv(out / "cell_estimates.csv", index=False)
    summ = []
    for g, x in eb.groupby("group"):
        rl = spearmanr(x.eb_log_gamma, x.length_um); rf = spearmanr(x.eb_log_gamma, x.logF0)
        lanes = [v.eb_log_gamma.values for _, v in x.groupby("lane") if len(v) >= 2]
        k = len(lanes); N = sum(len(v) for v in lanes)
        icc, pF = np.nan, np.nan
        if k >= 2:
            grand = np.concatenate(lanes).mean()
            msb = sum(len(v) * (v.mean() - grand) ** 2 for v in lanes) / (k - 1)
            msw = sum(((v - v.mean()) ** 2).sum() for v in lanes) / (N - k)
            n0 = (N - sum(len(v) ** 2 for v in lanes) / N) / (k - 1)
            icc = max(0.0, (msb - msw) / (msb + (n0 - 1) * msw)); pF = f_oneway(*lanes).pvalue
        z = lambda v: (v - v.mean()) / v.std()
        X = np.column_stack([np.ones(len(x)), z(x.length_um), z(x.logF0)])
        beta = np.linalg.lstsq(X, x.eb_log_gamma.values, rcond=None)[0]
        res = x.eb_log_gamma.values - X @ beta
        r2 = 1 - (res ** 2).sum() / ((x.eb_log_gamma - x.eb_log_gamma.mean()) ** 2).sum()
        # length association within lanes (lane-demeaned, lanes with >= 2 cells)
        w = x[x.groupby("lane").cell.transform("size") > 1]
        wl = w.length_um - w.groupby("lane").length_um.transform("mean")
        wg = w.eb_log_gamma - w.groupby("lane").eb_log_gamma.transform("mean")
        rw = spearmanr(wg, wl) if len(w) > 3 else None
        summ.append(dict(group=g, n_cells=len(x), n_lanes=x.lane.nunique(), rho_length=rl.statistic,
                         p_length=rl.pvalue,
                         rho_length_within_lane=rw.statistic if rw else np.nan,
                         p_length_within_lane=rw.pvalue if rw else np.nan, rho_logF0=rf.statistic, p_logF0=rf.pvalue, R2_length_logF0=r2,
                         lane_ICC=icc, p_lane_anova=pF, median_posterior_sd=x.eb_sd.median(),
                         sd_between_cells=x.eb_log_gamma.std()))
    s = pd.DataFrame(summ)
    s.to_csv(out / "covariate_summary.csv", index=False)
    return s


# ======================================================================
# 10. simulation-based calibration of the whole pipeline
# ======================================================================
def _artefact(cell, rng, lo, hi):
    y = cell.y.copy()
    j = rng.integers(2, len(y) - 1)
    y[j:] -= rng.uniform(lo, hi)
    return Cell(cell.group, cell.lane, cell.cell, cell.t, y, cell.length_um, float(np.abs(np.diff(y)).max()))


def _sbc_task(args):
    key, cond, cs, truth, sc, nodes, starts, seed = args
    rng = np.random.default_rng(seed)
    p = dict(truth)
    data = _simulate(cs, p, rng, lane_sd=cond["tau"], mean="hyper" if cond.get("misspec") else "exp")
    n_art = int(round(sc["artefact_fraction"] * len(data))) if cond["artefacts"] else 0
    for j in rng.choice(len(data), n_art, replace=False):
        data[j] = _artefact(data[j], rng, *sc["artefact_drop"])
    for c in data:
        c.max_jump = float(np.abs(np.diff(c.y)).max())
    out = dict(key=key, **{k: v for k, v in cond.items()}, n_cells=len(data), n_artefacts=n_art,
               true_CV=float(np.sqrt(np.expm1(p["scale"] ** 2))), true_omega=p["omega"],
               CV_noqc=np.nan, omega_noqc=np.nan, dropped_noqc=np.nan, CV_qc=np.nan, omega_qc=np.nan,
               dropped_qc=np.nan, CV_within_lane=np.nan, tau_hat=np.nan)   # fixed column set
    for qc_on in ([False, True] if cond["artefacts"] else [True]):
        d2, dropped = apply_qc(data, sc["qc"] if qc_on else 0.0)
        r = fit(Model(d2, nodes=nodes), starts=starts, seed=seed)
        tag = "qc" if qc_on else "noqc"
        out[f"CV_{tag}"] = r["cv"]; out[f"omega_{tag}"] = r["params"]["omega"]; out[f"dropped_{tag}"] = len(dropped)
    if cond.get("fit_lane"):
        d2, _ = apply_qc(data, sc["qc"])
        r = fit(Model(d2, lane=True, nodes=nodes, lane_nodes=11), starts=max(2, starts - 1), seed=seed)
        out["CV_within_lane"] = float(np.sqrt(np.expm1(r["params"]["scale"] ** 2)))
        out["tau_hat"] = r["params"]["tau"]
    return out


def stage_sbc(cfg, cells, out, jobs, main_params, logfile=None):
    sc = cfg["sbc"]
    kept, _ = apply_qc(cells, cfg["primary_qc"])
    G = by_group(kept)
    ck = Checkpoint(out / "sbc_replicates.csv")
    tasks = []
    conds = []
    for design in sc["designs"]:
        for cv in sc["true_cv"]:
            for tau in sc["true_tau"]:
                conds.append(dict(design=design, cv=cv, tau=tau, artefacts=True, misspec=False,
                                  fit_lane=bool(cv == sc["lane_fit_cv"]), omega_source="fitted"))
        conds.append(dict(design=design, cv=sc["lane_fit_cv"], tau=0.0, artefacts=False, misspec=True,
                          fit_lane=False, omega_source="fitted"))
    for g, w in cfg["published_gamma_dec"].items():
        if g in G:
            conds.append(dict(design=g, cv=None, tau=0.0, artefacts=False, misspec=False, fit_lane=False,
                              omega_source="published"))
    for ci, cond in enumerate(conds):
        g = cond["design"]
        if g not in G:
            continue
        mp = main_params[g]
        cv = cond["cv"] if cond["cv"] is not None else mp["CV"]
        truth = dict(mu=mp["mu"], omega=mp["omega"] if cond["omega_source"] == "fitted" else cfg["published_gamma_dec"][g],
                     sigma=mp["sigma"], phi=mp["phi"], scale=float(np.sqrt(np.log1p(cv ** 2))))
        for r in range(sc["reps"]):
            key = f"c{ci}|{r}"
            if key in ck.done:
                continue
            tasks.append((key, {**cond, "cv": cv, "cond_id": ci}, G[g], truth, sc, cfg["nodes"], sc["starts"],
                          cfg["seed"] + 999 * ci + r))
    log(f"SBC: {len(tasks)} simulated datasets to fit ({len(ck.done)} already done)", logfile)
    buf = []

    def on_res(i, res):
        buf.append(res)
        if len(buf) >= 5:
            ck.add(buf); buf.clear(); log(f"  SBC: {len(ck.done)} done", logfile)
    pmap(_sbc_task, tasks, jobs, on_result=on_res)
    ck.add(buf)
    df = ck.frame()
    agg = {}
    for col in ["CV_qc", "CV_noqc", "omega_qc", "CV_within_lane", "tau_hat"]:
        if col in df:
            agg[f"median_{col}"] = (col, "median")
    s = df.groupby(["cond_id", "design", "cv", "tau", "artefacts", "misspec", "omega_source"]).agg(
        reps=("key", "size"), true_CV=("true_CV", "first"), true_omega=("true_omega", "first"), **agg).reset_index()
    s["bias_CV_qc"] = s.median_CV_qc - s.true_CV
    if "median_CV_noqc" in s:
        s["bias_CV_noqc"] = s.median_CV_noqc - s.true_CV
    s["rel_bias_omega_qc"] = s.median_omega_qc / s.true_omega - 1
    s.to_csv(out / "sbc_summary.csv", index=False)
    return s
