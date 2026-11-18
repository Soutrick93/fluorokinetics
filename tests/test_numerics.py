"""Numerical checks; no data files needed. Run: python -m pytest tests -q"""
import os, sys
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fk.data import Cell
from fk.likelihood import Model, marg_cells_vec, _prior
from fk.fit import fit
from fk.cellwise import ou_noise


def make_cells(n=40, mu=np.log(0.02), s=0.4, omega=0.04, sigma=0.05, phi=4.0, lanes=8, tau=0.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 55.2, 6.0)
    leff = rng.normal(0, tau, lanes) if tau > 0 else np.zeros(lanes)
    cells = []
    for i in range(n):
        lane = i % lanes
        g = np.exp(mu + leff[lane] + s * rng.standard_normal())
        y = 10 - g * (-np.expm1(-omega * t) / omega) + ou_noise(t, sigma, phi, rng)
        cells.append(Cell("sim", f"L{lane}", f"c{i}", t, y, 3.0, 0.0))
    return cells


def brute_cell(q, sigma, mu, scale, dist):
    c, A, B, D = q
    u = np.linspace(mu - 12, mu + 12, 200001)
    g = np.exp(u)
    lp, _, _ = _prior(u, mu, scale, dist)
    return logsumexp(c - (A + 2 * g * B + g * g * D) / (2 * sigma ** 2) + lp) + np.log(u[1] - u[0])


def test_adaptive_quadrature_lognormal_and_gamma():
    cells = make_cells(n=10)
    for dist, scale in [("lognormal", 0.4), ("gamma", 0.45)]:
        m = Model(cells, dist=dist, nodes=21)
        p = dict(mu=np.log(0.02), omega=0.04, sigma=0.05, phi=4.0, scale=scale)
        for q in m.quads(p):
            a = marg_cells_vec(q, p["sigma"], [p["mu"]], scale, m.x, m.logw, dist)[0]
            assert abs(a - brute_cell(q, p["sigma"], p["mu"], scale, dist)) < 5e-3


def test_lane_model_reduces_to_single_level():
    cells = make_cells(n=20)
    th = np.array([np.log(0.02), np.log(0.04), np.log(0.05), np.log(4.0), np.log(0.4)])
    single = Model(cells).nll(th)
    lane = Model(cells, lane=True).nll(np.r_[th, np.log(1e-3)])
    assert abs(single - lane) < 1e-2


def test_recovery_of_simulated_heterogeneity():
    cells = make_cells(n=120, s=0.47, seed=3)      # true CV = 0.5
    r = fit(Model(cells), starts=3, seed=0)
    assert r["starts_agree"] == r["starts"]
    assert abs(r["cv"] - 0.5) < 0.15
    assert abs(r["params"]["omega"] - 0.04) < 0.015
