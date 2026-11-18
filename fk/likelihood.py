"""Likelihoods for the hierarchical apparent-decay model.

Mean:      log F_i(t) = a_i - gamma_i * f(t; omega),  f = (1 - exp(-omega t)) / omega
           (omega -> 0 gives f = t, the constant-rate model M0)
Residuals: stationary OU, Cov = sigma^2 exp(-phi |t - t'|)
Intercept a_i is profiled by generalised least squares.

For fixed (omega, sigma, phi) the per-cell log-likelihood is exactly quadratic in gamma:
    ll(gamma) = c - (A + 2 gamma B + gamma^2 D) / (2 sigma^2)
so the OU covariance is factorised once per cell and per likelihood call.

Random effects are integrated by adaptive Gauss-Hermite quadrature (nodes centred at the
mode of the integrand and scaled by its curvature). Fixed-node quadrature is NOT used:
it produces spurious optima for these sharply peaked cell likelihoods (paper, Fig. S2).

Cell-level distributions for u = log gamma:
  'lognormal': u ~ N(mu, s^2)
  'gamma'    : gamma ~ Gamma(shape k = 1/cv^2, mean m = exp(mu))
Lane model:   u_ij = mu + l_j + v_ij, l_j ~ N(0, tau^2), v_ij ~ N(0, s^2)
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, gammaln
from numpy.polynomial.hermite import hermgauss

LOG2PI = np.log(2 * np.pi)


def f_mean(t, omega):
    if omega < 1e-9:
        return t.copy()
    return -np.expm1(-omega * t) / omega


def cell_quad(t, y, omega, sigma, phi):
    """(c, A, B, D) for one cell, or None if the covariance is not positive definite."""
    n = len(t)
    C = np.exp(-phi * np.abs(t[:, None] - t[None, :]))
    C.flat[:: n + 1] += 1e-8
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return None
    W = solve_triangular(L, np.column_stack([np.ones(n), y, f_mean(t, omega)]), lower=True)
    z, wy, wf = W[:, 0], W[:, 1], W[:, 2]
    zz = z @ z
    py = wy - z * (z @ wy) / zz
    pf = wf - z * (z @ wf) / zz
    c = -0.5 * (n * LOG2PI + 2 * n * np.log(sigma) + 2 * np.log(np.diag(L)).sum())
    return c, py @ py, py @ pf, pf @ pf


def gls_intercept(t, y, gamma, omega, phi):
    n = len(t)
    C = np.exp(-phi * np.abs(t[:, None] - t[None, :])) + 1e-8 * np.eye(n)
    L = np.linalg.cholesky(C)
    z = solve_triangular(L, np.ones(n), lower=True)
    r = solve_triangular(L, y + gamma * f_mean(t, omega), lower=True)
    return (z @ r) / (z @ z)


# ---------- log prior on u = log gamma, with first and second derivatives ----------
def _prior(u, mu, scale, dist):
    """Return (logp, d1, d2) elementwise. mu may be an array broadcastable with u.
    lognormal: scale = s (sd of log gamma). gamma: scale = cv."""
    if dist == "lognormal":
        z = (u - mu) / scale
        return (-0.5 * z * z - np.log(scale) - 0.5 * LOG2PI, -z / scale, np.full_like(u, -1.0 / scale ** 2))
    if dist == "gamma":
        k = 1.0 / scale ** 2
        theta = np.exp(mu) / k                 # mean exp(mu)
        e = np.exp(u)
        logp = k * u - e / theta - gammaln(k) - k * np.log(theta)
        return logp, k - e / theta, -e / theta
    raise ValueError(dist)


def marg_cells_vec(q, sigma, mus, scale, x, logw, dist="lognormal"):
    """log ∫ exp(ll(e^u)) p(u | mu) du for one cell and an array of prior locations `mus`.
    Adaptive Gauss-Hermite, vectorised over mus."""
    c, A, B, D = q
    s2 = sigma ** 2
    mus = np.atleast_1d(np.asarray(mus, float))
    spread = scale if dist == "lognormal" else min(2.0, scale)  # search width in log space

    def h(u, mu):
        g = np.exp(u)
        lp, _, _ = _prior(u, mu, scale, dist)
        return c - (A + 2 * g * B + g * g * D) / (2 * s2) + lp

    # coarse search for the mode
    grid = mus[:, None] + spread * np.linspace(-6, 6, 49)[None, :]
    if D > 0 and -B / D > 0:
        grid = np.concatenate([grid, np.full((len(mus), 1), np.log(-B / D))], axis=1)
    hv = h(grid, mus[:, None])
    u = grid[np.arange(len(mus)), np.argmax(hv, axis=1)]
    # Newton refinement
    for _ in range(40):
        g = np.exp(u)
        _, p1, p2 = _prior(u, mus, scale, dist)
        d1 = -(B + g * D) * g / s2 + p1
        d2 = -(B + g * D) * g / s2 - D * g * g / s2 + p2
        ok = d2 < 0
        step = np.where(ok, np.clip(-d1 / np.where(ok, d2, -1.0), -1.0, 1.0), 0.0)
        u = u + step
        if np.all(np.abs(step) < 1e-10):
            break
    g = np.exp(u)
    _, _, p2 = _prior(u, mus, scale, dist)
    d2 = -(B + g * D) * g / s2 - D * g * g / s2 + p2
    shat = np.where(d2 < 0, 1.0 / np.sqrt(np.maximum(-d2, 1e-300)), spread)
    uk = u[:, None] + np.sqrt(2) * shat[:, None] * x[None, :]
    return logsumexp(h(uk, mus[:, None]) + x[None, :] ** 2 + logw[None, :], axis=1) + np.log(np.sqrt(2) * shat)


class Model:
    """Negative log-likelihood for one group.

    theta (log scale where positive):
      [mu, log omega, log sigma, log phi, log scale, (log tau)]
      omega omitted if omega_fixed is given; scale omitted if common (no random effect);
      log tau present only for lane=True.
    """

    def __init__(self, cells, random=True, dist="lognormal", lane=False, omega_fixed=None,
                 nodes=15, lane_nodes=15, omega_bounds=(1e-6, 5.0)):
        self.t = [c.t for c in cells]
        self.y = [c.y for c in cells]
        self.random, self.dist, self.lane = random, dist, lane
        self.omega_fixed = omega_fixed
        self.x, w = hermgauss(nodes)
        self.logw = np.log(w)
        self.lx, lw = hermgauss(lane_nodes)
        self.llogw = np.log(lw)
        if lane:
            lanes = sorted(set(c.lane for c in cells))
            idx = {l: i for i, l in enumerate(lanes)}
            self.lane_of = np.array([idx[c.lane] for c in cells])
            self.n_lanes = len(lanes)
        lo_om, hi_om = np.log(omega_bounds[0]), np.log(omega_bounds[1])
        b = [(-8.0, -1.0)]
        if omega_fixed is None:
            b.append((lo_om, hi_om))
        b += [(-6.0, 0.5), (np.log(1e-3), np.log(200.0))]
        if random:
            b.append((-4.0, 1.5))
        if lane:
            b.append((-7.0, 1.0))
        self.bounds = b
        self.n_cells = len(cells)

    @property
    def k(self):
        return len(self.bounds)

    def unpack(self, th):
        i = 0
        out = {"mu": th[i]}; i += 1
        if self.omega_fixed is None:
            out["omega"] = float(np.exp(th[i])); i += 1
        else:
            out["omega"] = float(self.omega_fixed)
        out["sigma"] = float(np.exp(th[i])); out["phi"] = float(np.exp(th[i + 1])); i += 2
        out["scale"] = float(np.exp(th[i])) if self.random else 0.0
        if self.random:
            i += 1
        out["tau"] = float(np.exp(th[i])) if self.lane else 0.0
        return out

    def quads(self, p):
        return [cell_quad(t, y, p["omega"], p["sigma"], p["phi"]) for t, y in zip(self.t, self.y)]

    def per_cell(self, th):
        """Per-cell marginal log-likelihoods (single-level models only)."""
        p = self.unpack(th)
        qs = self.quads(p)
        out = np.empty(len(qs))
        for j, q in enumerate(qs):
            if q is None:
                out[j] = -1e100
            elif self.random:
                out[j] = marg_cells_vec(q, p["sigma"], [p["mu"]], p["scale"], self.x, self.logw, self.dist)[0]
            else:
                c, A, B, D = q
                g = np.exp(p["mu"])
                out[j] = c - (A + 2 * g * B + g * g * D) / (2 * p["sigma"] ** 2)
        return out

    def nll(self, th):
        if not self.lane:
            v = -self.per_cell(th).sum()
            return v if np.isfinite(v) else 1e100
        p = self.unpack(th)
        qs = self.quads(p)
        if any(q is None for q in qs):
            return 1e100
        total = 0.0
        tau = p["tau"]
        for j in range(self.n_lanes):
            members = [qs[i] for i in np.where(self.lane_of == j)[0]]

            def g_lane(l):  # log integrand over lane effect l (array)
                l = np.atleast_1d(l)
                val = -0.5 * (l / tau) ** 2 - np.log(tau) - 0.5 * LOG2PI
                for q in members:
                    val = val + marg_cells_vec(q, p["sigma"], p["mu"] + l, p["scale"], self.x, self.logw)
                return val
            grid = tau * np.linspace(-6, 6, 25)
            gv = g_lane(grid)
            l0 = grid[np.argmax(gv)]
            hstep = max(tau * 0.05, 1e-4)
            for _ in range(8):
                f0, fp, fm = g_lane(np.array([l0, l0 + hstep, l0 - hstep]))
                d1 = (fp - fm) / (2 * hstep); d2 = (fp - 2 * f0 + fm) / hstep ** 2
                if d2 >= 0:
                    break
                st = float(np.clip(-d1 / d2, -tau, tau))
                l0 += st
                if abs(st) < 1e-6:
                    break
            f0, fp, fm = g_lane(np.array([l0, l0 + hstep, l0 - hstep]))
            d2 = (fp - 2 * f0 + fm) / hstep ** 2
            sh = 1 / np.sqrt(-d2) if d2 < 0 else tau
            lk = l0 + np.sqrt(2) * sh * self.lx
            total += logsumexp(g_lane(lk) + self.lx ** 2 + self.llogw) + np.log(np.sqrt(2) * sh)
        return -total if np.isfinite(total) else 1e100

    def cv(self, p):
        if not self.random:
            return 0.0
        if self.dist == "gamma":
            return p["scale"]
        return float(np.sqrt(np.expm1(p["scale"] ** 2)))


def eb_log_gamma(cell, p, grid_n=4001):
    """Posterior mean and sd of log gamma for one cell under a lognormal single-level fit."""
    q = cell_quad(cell.t, cell.y, p["omega"], p["sigma"], p["phi"])
    c, A, B, D = q
    s = max(p["scale"], 1e-3)
    u = np.linspace(p["mu"] - 8 * s, p["mu"] + 8 * s, grid_n)
    g = np.exp(u)
    lp = c - (A + 2 * g * B + g * g * D) / (2 * p["sigma"] ** 2) - 0.5 * ((u - p["mu"]) / s) ** 2
    w = np.exp(lp - lp.max()); w /= w.sum()
    m = (w * u).sum()
    return m, float(np.sqrt((w * (u - m) ** 2).sum()))
