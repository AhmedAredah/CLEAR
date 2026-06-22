"""CLEAR — train/test splits with explicit leakage guarantees.

The default is recording-level (clean). The random split is available but is flagged LEAKY because
sibling windows of the same vehicle/recording land in both folds. leakage_check() verifies that a
clean split shares no recording (or vehicle) across folds and returns a PASS/WARN verdict for the
Eval Card.
"""
from __future__ import annotations
import numpy as np

KINDS = ("random", "recording", "location", "vehicle")


def make_split(ws, kind="recording", test_frac=0.2, seed=0):
    """Return (train_idx, test_idx, info). kind in {random, recording, location, vehicle}."""
    rng = np.random.default_rng(seed); N = len(ws)
    if kind == "random":
        perm = rng.permutation(N); cut = int((1 - test_frac) * N)
        tr, te = perm[:cut], perm[cut:]
    elif kind in ("recording", "location", "vehicle"):
        key = {"recording": ws.REC, "location": ws.LOC, "vehicle": ws.VID}[kind]
        groups = np.unique(key); rng.shuffle(groups)
        n_test = max(1, int(round(test_frac * len(groups)))); test_groups = set(groups[:n_test].tolist())
        mask = np.array([g in test_groups for g in key])
        te = np.where(mask)[0]; tr = np.where(~mask)[0]
    else:
        raise ValueError(f"unknown split {kind!r}; choose from {KINDS}")
    info = leakage_check(ws, tr, te, kind)
    return tr, te, info


def leakage_check(ws, tr, te, kind):
    """PASS if train/test share no recording AND no vehicle; WARN otherwise (e.g. random)."""
    rec_overlap = len(set(ws.REC[tr]) & set(ws.REC[te]))
    veh_overlap = len(set(ws.VID[tr]) & set(ws.VID[te]))
    leaky = (kind == "random") or rec_overlap > 0 or veh_overlap > 0
    return dict(kind=kind, n_train=int(len(tr)), n_test=int(len(te)),
                recordings_shared=rec_overlap, vehicles_shared=veh_overlap,
                verdict="WARN-LEAKY" if leaky else "PASS-CLEAN", leaky=bool(leaky))
