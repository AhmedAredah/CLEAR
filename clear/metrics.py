"""CLEAR — metrics: horizon FDE/RMSE, bootstrap CIs, paired CA win-rate, stratified error,
best-of-k sweep. All operate on canonical-frame predictions (N,HOR,2) vs ground truth (N,HOR,2)."""
from __future__ import annotations
import numpy as np


def _fde(pred, gt, hz):                       # final displacement error at step index hz (1-based)
    return np.sqrt(((pred[:, hz - 1] - gt[:, hz - 1]) ** 2).sum(-1))


def horizon_errors(pred, gt, horizons):
    """Return {hz: dict(fde_mean, rmse, ci)} per horizon (RMSE = sqrt(mean sq FDE))."""
    out = {}
    for hz in horizons:
        e = _fde(pred, gt, hz)
        out[hz] = dict(fde_mean=round(float(e.mean()), 3), fde_median=round(float(np.median(e)), 3),
                       rmse=round(float(np.sqrt((e ** 2).mean())), 3), ci=bootstrap_ci(e))
    return out


def bootstrap_ci(err, n=1000, seed=0, stat="mean"):
    rng = np.random.default_rng(seed); N = len(err)
    if N == 0: return [None, None]
    idx = rng.integers(0, N, size=(n, N))
    s = err[idx].mean(1) if stat == "mean" else np.sqrt((err[idx] ** 2).mean(1))
    return [round(float(np.percentile(s, 2.5)), 3), round(float(np.percentile(s, 97.5)), 3)]


def paired_ca_winrate(pred, ca_pred, gt, hz):
    """Fraction of windows where CA is strictly closer to GT than the model, + mean signed gap CI."""
    em = _fde(pred, gt, hz); ec = _fde(ca_pred, gt, hz)
    ca_wins = float((ec < em).mean())
    diff = em - ec                              # >0 => CA better
    return dict(ca_win_pct=round(100 * ca_wins, 1), mean_gap=round(float(diff.mean()), 3),
                gap_ci=bootstrap_ci(diff))


def stratified(pred, gt, ws, hz):
    """RMSE@hz per stratum (cruise/mild/hard, lane-keep/lane-change) + window shares."""
    e2 = ((pred[:, hz - 1] - gt[:, hz - 1]) ** 2).sum(-1)
    N = len(ws); out = {}
    masks = {"cruise": ws.KIN == 0, "mild": ws.KIN == 1, "hard": ws.KIN == 2,
             "lane-keep": ws.LC == 0, "lane-change": ws.LC == 1}
    for k, m in masks.items():
        if m.sum() == 0: continue
        out[k] = dict(rmse=round(float(np.sqrt(e2[m].mean())), 3), share_pct=round(100 * float(m.mean()), 1))
    return out


def predicted_collision(pred, ws, veh_len=5.0):
    """Admissibility: fraction of windows whose predicted longitudinal path overtakes the observed
    leader's actual future position within a car length (i.e. the model forecasts a rear-end crash).
    Reported overall and on the safety-relevant hard-braking tail. No-leader windows count as safe."""
    plong = pred[:, :, 1]                                  # (N,HOR) predicted longitudinal
    gap = ws.LEADER - plong                                # (N,HOR), nan where no leader
    has = np.isfinite(ws.LEADER).any(1)
    finite_gap = np.where(np.isfinite(gap), gap, np.inf)
    min_gap = finite_gap.min(1)                            # closest approach per window
    coll = (min_gap < veh_len) & has
    hard = ws.KIN == 2
    return dict(all_pct=round(100 * float(coll.mean()), 2),
                hard_pct=round(100 * float(coll[hard].mean()), 2) if hard.any() else None,
                n_with_leader=int(has.sum()))


def best_of_k(preds, probs, gt, hz, ks=(1, 2, 3, 4, 5, 6)):
    """preds (N,K,HOR,2), probs (N,K) or None. Oracle min FDE over the top-k most-probable modes,
    for each k. Single-mode = k=1. Returns {k: rmse}."""
    N, K = preds.shape[0], preds.shape[1]
    err = np.sqrt(((preds[:, :, hz - 1] - gt[:, None, hz - 1]) ** 2).sum(-1))   # (N,K)
    order = np.argsort(-probs, 1) if probs is not None else np.tile(np.arange(K), (N, 1))
    out = {}
    for k in ks:
        if k > K: break
        topk = order[:, :k]
        bk = np.take_along_axis(err, topk, 1).min(1)
        out[k] = round(float(np.sqrt((bk ** 2).mean())), 3)
    return out
