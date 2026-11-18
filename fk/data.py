"""Data loading, trajectory selection and quality control.

A trajectory is one cell (keys: date, strain, condition, Fluo, gl_id, cell) in the
starvation phase (c_step == 'SPM.1') with at least `min_points` distinct positive
fluorescence measurements. With min_points = 8 this gives the 393 trajectories
analysed in the paper.
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["date", "strain", "condition", "Fluo", "gl_id", "cell"]
REQUIRED = KEYS + ["c_step", "time_sec", "m_start", "Ampl", "length_um"]


@dataclass
class Cell:
    group: str          # e.g. "20201217_GFP"
    lane: str
    cell: str
    t: np.ndarray       # hours since start of starvation phase (m_start)
    y: np.ndarray       # log total fluorescence
    length_um: float
    max_jump: float     # largest |diff(log F)| between consecutive frames
    extra: dict = field(default_factory=dict)


def read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")
    if path.suffix.lower() == ".feather":
        try:
            return pd.read_feather(path)
        except ImportError as e:
            raise ImportError("Reading .feather needs pyarrow: python -m pip install pyarrow") from e
    return pd.read_csv(path, compression="infer")


def load_cells(path, phase="SPM.1", min_points=8, time_origin="m_start"):
    d = read_table(path)
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Is this myframes_degradation?")
    d = d[d.c_step.astype(str).eq(phase)].copy()
    for c in ["Ampl", "time_sec", "m_start", "length_um"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Ampl", "time_sec"])
    d = d[d.Ampl > 0]
    cells = []
    for key, g in d.groupby(KEYS, dropna=False):
        g = g.sort_values("time_sec").drop_duplicates("time_sec")
        if len(g) < min_points:
            continue
        ref = float(g.m_start.iloc[0]) if time_origin == "m_start" else float(g.time_sec.iloc[0])
        t = (g.time_sec.to_numpy(float) - ref) / 3600.0
        y = np.log(g.Ampl.to_numpy(float))
        if np.unique(t).size < 3:
            continue
        cells.append(Cell(group=f"{key[0]}_{key[3]}", lane=str(key[4]), cell=str(key[5]), t=t, y=y,
                          length_um=float(np.nanmean(g.length_um)),
                          max_jump=float(np.abs(np.diff(y)).max())))
    if not cells:
        raise ValueError("No trajectories passed selection; check the input file and phase name.")
    return cells


def apply_qc(cells, jump):
    """Drop cells with any |Δlog F| > jump between consecutive frames (jump <= 0: keep all)."""
    if jump is None or jump <= 0:
        return list(cells), []
    keep = [c for c in cells if c.max_jump <= jump]
    drop = [c for c in cells if c.max_jump > jump]
    return keep, drop


def by_group(cells):
    out = {}
    for c in cells:
        out.setdefault(c.group, []).append(c)
    return dict(sorted(out.items()))
