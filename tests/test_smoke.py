"""Data-free unit tests for CLEAR. Build a synthetic constant-acceleration WindowSet and check the
core invariants (CA recovers CA motion, CA beats CV, leakage check, Eval Card)."""
import numpy as np
from clear.data import WindowSet
from clear.baselines import CV, CA
from clear.splits import make_split
from clear.card import evaluate


def make_synth(n=400, fps=25.0, hor=25, pred_s=5.0, seed=0):
    rng = np.random.default_rng(seed)
    PRED_T = np.linspace(pred_s / hor, pred_s, hor)
    vlon = rng.uniform(20, 35, n); vlat = rng.uniform(-0.5, 0.5, n)
    alon = rng.uniform(-1.5, 1.5, n); alat = rng.uniform(-0.2, 0.2, n)
    VT = np.stack([vlon, vlat], 1).astype(np.float32); AT = np.stack([alon, alat], 1).astype(np.float32)
    t = PRED_T[None, :]
    gt_long = vlon[:, None] * t + 0.5 * alon[:, None] * t ** 2
    gt_lat = vlat[:, None] * t + 0.5 * alat[:, None] * t ** 2
    GT = np.stack([gt_lat, gt_long], -1).astype(np.float32)
    LEAD = np.full((n, hor), np.nan, np.float32)
    rec = rng.integers(0, 8, n); vid = rec * 1000 + rng.integers(0, 50, n)
    return WindowSet(VT, AT, GT, LEAD, rng.integers(0, 3, n), rng.integers(0, 2, n),
                     rec, np.zeros(n, int), vid, fps, hor, PRED_T, "synthetic")


def test_ca_recovers_and_beats_cv():
    ws = make_synth()
    ca = np.sqrt(((CA().predict(ws)[:, -1] - ws.GT[:, -1]) ** 2).sum(-1)).mean()
    cv = np.sqrt(((CV().predict(ws)[:, -1] - ws.GT[:, -1]) ** 2).sum(-1)).mean()
    assert ca < 1e-3, f"CA should reproduce CA motion exactly, got {ca:.4f}"
    assert ca < cv, "CA must beat CV on accelerating motion"


def test_leakage_checks():
    ws = make_synth()
    _, _, clean = make_split(ws, "recording")
    assert clean["verdict"] == "PASS-CLEAN" and clean["recordings_shared"] == 0
    _, _, rand = make_split(ws, "random")
    assert rand["leaky"], "random split must be flagged leaky"


def test_eval_card():
    ws = make_synth()
    card = evaluate(ws, split="recording")
    assert card.checklist["Constant-acceleration baseline reported"]
    assert card.ladder["CA"]["fde"]["5"]["rmse"] < card.ladder["CV"]["fde"]["5"]["rmse"]
    assert "all_pct" in card.admissibility["CA"]


if __name__ == "__main__":
    test_ca_recovers_and_beats_cv(); test_leakage_checks(); test_eval_card()
    print("OK - all smoke tests passed")
