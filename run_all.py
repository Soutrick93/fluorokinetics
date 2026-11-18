#!/usr/bin/env python
"""Run the full fluorescence-decay heterogeneity analysis.

  python run_all.py --quick                 # ~5 min check, writes results_quick/
  python run_all.py --jobs 4                # full analysis (several hours; resumable)
  python run_all.py --stages main,figures   # selected stages only
  python run_all.py --list                  # show stages, inputs and outputs

Each stage writes results/<stage>/ and a DONE.json marker. Finished stages are skipped
on the next run unless --force is given. Bootstrap and SBC stages checkpoint every few
replicates, so an interrupted run continues where it stopped.
"""
import os
for _v in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")   # one BLAS thread per process; parallelism comes from --jobs

import argparse, copy, json, sys, time, traceback, platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STAGES = [
    ("cellwise",   "degradation file", "cellwise.csv, cellwise_summary.csv (Table 1, Fig 2, Fig 3A)"),
    ("main",       "degradation file", "hierarchical_fits.csv, model_comparison.csv, theta.json (Table 3, S1, S2; OU-likelihood M0 vs M1 test)"),
    ("loo",        "main",             "leave_one_out.csv"),
    ("synthetic",  "main",             "synthetic_recovery.csv, synthetic_summary.csv (Table 2, Fig 3A)"),
    ("structure",  "main",             "lane_fits.csv, gamma_fits.csv, structure_comparison.csv (lane model, gamma random effect)"),
    ("bootstrap",  "main",             "parametric and lane-cluster bootstrap summaries (CIs)"),
    ("bleaching",  "main + bleaching file", "bleaching_calibration.csv, bleaching_refits.csv (Table S3)"),
    ("covariates", "main",             "cell_estimates.csv, covariate_summary.csv (Table S4, Fig 4C)"),
    ("sbc",        "main",             "sbc_replicates.csv, sbc_summary.csv (pipeline validation, Fig S3)"),
    ("figures",    "any of the above", "figures/*.svg|pdf|png"),
]


def deep_update(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_update(a[k], v)
        else:
            a[k] = v
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--stages", default="all")
    ap.add_argument("--quick", action="store_true", help="small fast run to check everything works")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true", help="re-run stages that are already DONE")
    ap.add_argument("--data", help="override degradation_file")
    ap.add_argument("--bleaching", help="override bleaching_file")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pack", action="store_true", help="zip the result files into results_archive.zip")
    a = ap.parse_args()

    if a.pack:
        import zipfile
        cfg0 = json.loads(Path(a.config).read_text(encoding="utf-8"))
        R0 = Path(a.config).resolve().parent / (cfg0["results_dir"] + ("_quick" if a.quick else ""))
        z = R0.parent / (R0.name + "_archive.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in R0.rglob("*"):
                if p.is_file() and p.stat().st_size < 30e6:
                    zf.write(p, p.relative_to(R0.parent))
        print(f"Wrote {z}")
        return
    if a.list:
        print(f"{'stage':<11} {'needs':<24} outputs")
        for s, need, outp in STAGES:
            print(f"{s:<11} {need:<24} {outp}")
        return

    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required.")
    try:
        import numpy, scipy, pandas, matplotlib  # noqa
    except ImportError as e:
        sys.exit(f"Missing package: {e.name}. Run: python -m pip install -r requirements.txt")

    from fk.data import load_cells, apply_qc, by_group
    from fk import stages as S
    from fk.util import log, write_json, read_json
    from fk import figures as F

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.quick:
        cfg = deep_update(copy.deepcopy(cfg), cfg["quick"])
        cfg["results_dir"] += "_quick"; cfg["figures_dir"] += "_quick"
    if a.data:
        cfg["degradation_file"] = a.data
    if a.bleaching:
        cfg["bleaching_file"] = a.bleaching
    base = Path(a.config).resolve().parent
    for k in ["degradation_file", "bleaching_file", "results_dir", "figures_dir"]:
        p = Path(cfg[k])
        cfg[k] = str(p if p.is_absolute() else base / p)
    R = Path(cfg["results_dir"]); R.mkdir(parents=True, exist_ok=True)
    logfile = R / "run_log.txt"
    want = [s for s, _, _ in STAGES] if a.stages == "all" else [s.strip() for s in a.stages.split(",")]
    bad = [s for s in want if s not in [x for x, _, _ in STAGES]]
    if bad:
        sys.exit(f"Unknown stage(s): {bad}. Use --list.")

    log(f"Python {platform.python_version()}, jobs={a.jobs}, quick={a.quick}, stages={want}", logfile)
    log(f"Loading {cfg['degradation_file']}", logfile)
    cells = load_cells(cfg["degradation_file"], min_points=cfg["min_points"])
    if cfg.get("groups"):
        cells = [c for c in cells if c.group in cfg["groups"]]
    counts = {g: len(v) for g, v in by_group(cells).items()}
    log(f"{len(cells)} trajectories: {counts}", logfile)
    if not a.quick and len(cells) != 393:
        log(f"NOTE: expected 393 trajectories for the paper data, got {len(cells)}.", logfile)

    def done(stage):
        return (R / stage / "DONE.json").exists() and not a.force

    def mark(stage, t0):
        write_json(R / stage / "DONE.json", dict(stage=stage, seconds=round(time.time() - t0), quick=a.quick,
                                                 finished=time.strftime("%Y-%m-%d %H:%M:%S")))

    def theta():
        return read_json(R / "main/theta.json")

    def main_params():
        import pandas as pd
        df = pd.read_csv(R / "main/hierarchical_fits.csv")
        pq = f"qc{cfg['primary_qc']:g}"
        out = {}
        for _, r in df[(df.qc == pq) & (df.model == "M1_random")].iterrows():
            out[r.group] = dict(mu=r.mu, omega=r.omega, sigma=r.sigma, phi=r.phi_per_h, scale=r.scale, CV=r.CV)
        return out

    failed = []
    for stage in want:
        out = R / stage; out.mkdir(exist_ok=True)
        if stage != "figures" and done(stage):
            log(f"== {stage}: already done (use --force to redo)", logfile)
            continue
        if stage not in ("cellwise", "main", "figures") and not (R / "main/theta.json").exists():
            log(f"== {stage}: SKIPPED, needs stage 'main' first", logfile); failed.append(stage); continue
        if a.force and stage != "figures":
            for f in out.glob("*"):
                if f.is_file():
                    f.unlink()   # --force: start this stage from scratch (old checkpoints removed)
        log(f"== {stage}: start", logfile); t0 = time.time()
        try:
            if stage == "cellwise":
                print(S.stage_cellwise(cfg, cells, out, a.jobs).round(3).to_string(index=False))
            elif stage == "main":
                _, comp, _ = S.stage_main(cfg, cells, out, a.jobs)
                print(comp.round(4).to_string(index=False))
            elif stage == "loo":
                S.stage_loo(cfg, cells, out, a.jobs, theta())
            elif stage == "synthetic":
                print(S.stage_synthetic(cfg, cells, out, a.jobs, main_params()).round(3).to_string(index=False))
            elif stage == "structure":
                import pandas as pd
                lane = S.stage_structure(cfg, cells, out, a.jobs, "lane")
                gam = S.stage_structure(cfg, cells, out, a.jobs, "gamma")
                comp = S.compare_structures(pd.read_csv(R / "main/hierarchical_fits.csv"), lane, gam, out)
                print(comp.round(3).to_string(index=False))
            elif stage == "bootstrap":
                S.stage_bootstrap(cfg, cells, out, a.jobs, theta(), "parametric", cfg["bootstrap_reps"], [cfg["primary_qc"]], logfile)
                print(S.stage_bootstrap(cfg, cells, out, a.jobs, theta(), "parametric", cfg["bootstrap_reps_all"], [0.0], logfile).round(4).to_string(index=False))
                print(S.stage_bootstrap(cfg, cells, out, a.jobs, theta(), "lane", cfg["lane_bootstrap_reps"], [cfg["primary_qc"]], logfile).round(4).to_string(index=False))
            elif stage == "bleaching":
                if not Path(cfg["bleaching_file"]).exists():
                    raise FileNotFoundError(f"bleaching file not found: {cfg['bleaching_file']}")
                print(S.stage_bleaching(cfg, cells, out, a.jobs, theta()).round(4).to_string(index=False))
            elif stage == "covariates":
                print(S.stage_covariates(cfg, cells, out, theta()).round(3).to_string(index=False))
            elif stage == "sbc":
                print(S.stage_sbc(cfg, cells, out, a.jobs, main_params(), logfile).round(3).to_string(index=False))
            elif stage == "figures":
                F.make_all(cfg, cells, R, Path(cfg["figures_dir"]))
            if stage != "figures":
                mark(stage, t0)
            log(f"== {stage}: finished in {time.time() - t0:.0f} s", logfile)
        except Exception as e:
            failed.append(stage)
            log(f"== {stage}: FAILED: {e}", logfile)
            with open(logfile, "a", encoding="utf-8") as fh:
                traceback.print_exc(file=fh)
            traceback.print_exc()
    if failed:
        log(f"Finished with problems in: {failed}. Details are in {logfile}.", logfile)
    else:
        log("All requested stages finished.", logfile)


if __name__ == "__main__":
    main()
