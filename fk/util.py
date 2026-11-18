import json, time, os, csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd


def log(msg, logfile=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def pmap(fn, tasks, jobs=1, on_result=None):
    """Map fn over tasks, in parallel if jobs > 1. on_result(task_index, result) is called
    as results arrive (used for checkpointing). Returns results in task order."""
    results = [None] * len(tasks)
    if jobs <= 1 or len(tasks) <= 1:
        for i, t in enumerate(tasks):
            results[i] = fn(t)
            if on_result:
                on_result(i, results[i])
        return results
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(fn, t): i for i, t in enumerate(tasks)}
        for f in as_completed(futs):
            i = futs[f]
            results[i] = f.result()
            if on_result:
                on_result(i, results[i])
    return results


class Checkpoint:
    """Append-only CSV of finished rows keyed by 'key'; lets long runs resume after a crash."""

    def __init__(self, path):
        self.path = Path(path)
        self.done = set()
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                self.done = set(pd.read_csv(self.path)["key"].astype(str))
            except Exception:
                self.done = set()

    def add(self, rows):
        """Append rows. Rows may have different columns: if new columns appear, the file is
        rewritten with the union of columns (missing values left blank)."""
        rows = rows if isinstance(rows, list) else [rows]
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists:
            with open(self.path, newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), [])
            if all(k in header for k in keys):
                with open(self.path, "a", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=header, restval="")
                    for r in rows:
                        w.writerow(r)
            else:
                old = pd.read_csv(self.path)
                pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(self.path, index=False)
        else:
            pd.DataFrame(rows, columns=keys).to_csv(self.path, index=False)
        for r in rows:
            self.done.add(str(r["key"]))

    def frame(self):
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()


def write_json(path, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
