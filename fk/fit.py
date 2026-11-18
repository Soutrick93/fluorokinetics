"""Multi-start maximum-likelihood fitting with convergence diagnostics."""
import numpy as np
from scipy.optimize import minimize
from .likelihood import Model


def _starts(model, n, rng, init=None):
    lo = np.array([b[0] for b in model.bounds]); hi = np.array([b[1] for b in model.bounds])
    om = [0.03, 0.005, 0.1, 0.0005, 0.05, 0.015, 0.08, 0.002]
    ph = [3.0, 0.5, 10.0, 1.0, 0.2, 5.0]
    sc = [0.4, 0.2, 0.7, 1.0, 0.3, 0.55, 0.15, 0.85]
    ta = [0.2, 0.05, 0.5, 0.1]
    out = []
    if init is not None:
        out.append(np.clip(np.asarray(init, float), lo + 1e-9, hi - 1e-9))
    for s in range(n):
        x = [np.log(0.02) + rng.normal(0, 0.4)]
        if model.omega_fixed is None:
            x.append(np.log(om[s % len(om)]))
        x += [np.log(0.08), np.log(ph[s % len(ph)])]
        if model.random:
            x.append(np.log(sc[s % len(sc)]))
        if model.lane:
            x.append(np.log(ta[s % len(ta)]))
        out.append(np.clip(x, lo + 1e-9, hi - 1e-9))
    return out[: max(n, 1) + (init is not None)]


def fit(model, starts=6, seed=0, init=None, maxiter=4000):
    rng = np.random.default_rng(seed)
    fits = []
    for x0 in _starts(model, starts, rng, init):
        try:
            r = minimize(model.nll, x0, method="L-BFGS-B", bounds=model.bounds,
                         options={"maxiter": maxiter, "ftol": 1e-10, "maxls": 40})
            fits.append(r)
        except Exception as e:  # never let one start kill the run
            print(f"    start failed: {e}", flush=True)
    if not fits:
        raise RuntimeError("all optimiser starts failed")
    best = min(fits, key=lambda r: r.fun)
    objs = np.array([r.fun for r in fits])
    p = model.unpack(best.x)
    n, k = model.n_cells, model.k
    aicc = 2 * best.fun + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)
    lo = [b[0] for b in model.bounds]; hi = [b[1] for b in model.bounds]
    at_bound = [i for i, (v, a, b) in enumerate(zip(best.x, lo, hi)) if v - a < 1e-3 or b - v < 1e-3]
    return dict(x=best.x, nll=float(best.fun), aicc=float(aicc), k=k, n=n, params=p, cv=model.cv(p),
                starts=len(fits), starts_agree=int((objs - objs.min() < 0.1).sum()),
                objective_spread=float(objs.max() - objs.min()), success=bool(best.success),
                at_bound=at_bound)


def summary_row(group, res, label=""):
    p = res["params"]
    return dict(group=group, model=label, n_cells=res["n"], mu=p["mu"], gamma_median=float(np.exp(p["mu"])),
                omega=p["omega"], sigma=p["sigma"], phi_per_h=p["phi"],
                OU_half_life_h=float(np.log(2) / p["phi"]), scale=p["scale"], tau=p["tau"], CV=res["cv"],
                nll=res["nll"], AICc=res["aicc"], k=res["k"], starts=res["starts"],
                starts_agree=res["starts_agree"], objective_spread=res["objective_spread"],
                success=res["success"], omega_at_lower=bool(p["omega"] < 1.05e-6))
