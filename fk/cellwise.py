"""Cellwise (one trajectory at a time) analyses with independent Gaussian residuals:
M0 vs M1 by AICc, 70/30 temporal hold-out, profile-likelihood intervals, and the
synthetic identifiability benchmark with OU noise.

For fixed omega the mean is linear in (a, gamma), so the residual sum of squares is a
closed-form quadratic in gamma. Fits and profiles are computed on a dense (gamma, omega)
grid and then refined, which avoids local optima.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

OMEGA_GRID = np.exp(np.linspace(np.log(1e-5), np.log(5.0), 160))
GAMMA_MAX = 0.3
GAMMA_GRID = np.linspace(0.0, GAMMA_MAX, 601)


def _fmat(t, omegas):
    om = omegas[None, :]
    return -np.expm1(-om * t[:, None]) / om        # (n, n_omega)


def _rss_grid(t, y):
    F = _fmat(t, OMEGA_GRID)
    yc = y - y.mean()
    Fc = F - F.mean(axis=0)
    Syy = yc @ yc
    Syf = yc @ Fc                                    # (n_omega,)
    Sff = (Fc * Fc).sum(axis=0)
    g = GAMMA_GRID[:, None]
    return Syy + 2 * g * Syf[None, :] + g * g * Sff[None, :]   # (n_gamma, n_omega); model y = a - g f


def _rss(t, y, gamma, omega):
    f = t if omega < 1e-9 else -np.expm1(-omega * t) / omega
    r = y + gamma * f
    r = r - r.mean()
    return r @ r


def fit_m1(t, y):
    R = _rss_grid(t, y)
    i, j = np.unravel_index(np.argmin(R), R.shape)
    x0 = [max(GAMMA_GRID[i], 1e-6), np.log(OMEGA_GRID[j])]
    r = minimize(lambda x: _rss(t, y, x[0], np.exp(x[1])), x0, method="L-BFGS-B",
                 bounds=[(0.0, GAMMA_MAX), (np.log(1e-5), np.log(5.0))])
    return float(r.x[0]), float(np.exp(r.x[1])), float(r.fun)


def fit_m0(t, y):
    tc = t - t.mean(); yc = y - y.mean()
    k = -(tc @ yc) / (tc @ tc)
    return float(k), float(((yc + k * tc) ** 2).sum())


def aicc(rss, n, k):
    k = k + 1  # + residual variance
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)


def predict(t_tr, y_tr, t_te, model):
    if model == "M0":
        k, _ = fit_m0(t_tr, y_tr)
        a = np.mean(y_tr + k * t_tr)
        return a - k * t_te
    g, om, _ = fit_m1(t_tr, y_tr)
    f = lambda tt: -np.expm1(-om * tt) / om
    a = np.mean(y_tr + g * f(t_tr))
    return a - g * f(t_te)


def profile_intervals(t, y):
    n = len(t)
    R = np.maximum(_rss_grid(t, y), 1e-300)
    ll = -0.5 * n * np.log(R / n)
    thr = ll.max() - chi2.ppf(0.95, 1) / 2
    pg = ll.max(axis=1); po = ll.max(axis=0)
    ig = np.where(pg >= thr)[0]; io = np.where(po >= thr)[0]
    g_lo, g_hi = GAMMA_GRID[ig.min()], GAMMA_GRID[ig.max()]
    o_lo, o_hi = OMEGA_GRID[io.min()], OMEGA_GRID[io.max()]
    return dict(gamma_ci_low=g_lo, gamma_ci_high=g_hi, omega_ci_low=o_lo, omega_ci_high=o_hi,
                gamma_profile_boundary=bool(ig.min() == 0 or ig.max() == len(GAMMA_GRID) - 1),
                omega_profile_boundary=bool(io.min() == 0 or io.max() == len(OMEGA_GRID) - 1))


def analyse_cell(c):
    t, y = c.t, c.y
    n = len(t)
    g, om, rss1 = fit_m1(t, y)
    k0, rss0 = fit_m0(t, y)
    ntr = int(round(0.7 * n))
    out = dict(group=c.group, lane=c.lane, cell=c.cell, n_points=n, duration_h=float(t.max() - t.min()),
               max_jump=c.max_jump, M0_rate=k0, M1_gamma0=g, M1_omega=om,
               AICc_M0=aicc(rss0, n, 2), AICc_M1=aicc(rss1, n, 3))
    out["dAICc_M1_minus_M0"] = out["AICc_M1"] - out["AICc_M0"]
    if ntr >= 4 and n - ntr >= 1:
        e0 = y[ntr:] - predict(t[:ntr], y[:ntr], t[ntr:], "M0")
        e1 = y[ntr:] - predict(t[:ntr], y[:ntr], t[ntr:], "M1")
        out.update(holdout_RMSE_M0=float(np.sqrt(np.mean(e0 ** 2))), holdout_RMSE_M1=float(np.sqrt(np.mean(e1 ** 2))))
    out.update(profile_intervals(t, y))
    return out


# ---------- synthetic identifiability benchmark (cellwise, OU noise) ----------
def ou_noise(t, sigma, phi, rng):
    n = len(t)
    C = np.exp(-phi * np.abs(t[:, None] - t[None, :])) + 1e-10 * np.eye(n)
    return sigma * np.linalg.cholesky(C) @ rng.standard_normal(n)


def densify(t, mult):
    if mult == 1:
        return t.copy()
    return np.linspace(t.min(), t.max(), int(len(t) * mult))


def synthetic_task(args):
    group, schedules, sigma, phi, gammas, omegas, densities, reps, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    for dens in densities:
        for g0 in gammas:
            for om in omegas:
                for r in range(reps):
                    t = densify(schedules[r % len(schedules)], dens)
                    y = 10 - g0 * (-np.expm1(-om * t) / om) + ou_noise(t, sigma, phi, rng)
                    gh, oh, _ = fit_m1(t, y)
                    pi = profile_intervals(t, y)
                    rows.append(dict(group=group, density=dens, gamma_true=g0, omega_true=om, rep=r,
                                     n_points=len(t), gamma_hat=gh, omega_hat=oh,
                                     gamma_rel_err=abs(gh - g0) / g0, omega_rel_err=abs(oh - om) / om,
                                     omega_profile_boundary=pi["omega_profile_boundary"],
                                     gamma_profile_boundary=pi["gamma_profile_boundary"]))
    return rows
