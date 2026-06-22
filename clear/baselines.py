"""CLEAR — the honest zero/low-parameter reference baselines: CV, CA, IDM.

All operate in the per-window canonical frame; predictions are [lat, long]. CA is the central
reference the toolkit always reports and pairs every learned model against.
"""
from __future__ import annotations
import numpy as np
from .predictor import Predictor


class CV(Predictor):
    """Constant velocity: p(t) = v0 * t."""
    name = "CV"
    def predict(self, ws):
        t = ws.PRED_T[None, :, None]                       # (1,HOR,1)
        v = ws.VT[:, None, :]                               # (N,1,2) [vlon,vlat]
        lon = v[..., 0] * t[..., 0]; lat = v[..., 1] * t[..., 0]
        return np.stack([lat, lon], -1).astype(np.float32)  # (N,HOR,2) [lat,long]


class CA(Predictor):
    """Constant acceleration: p(t) = v0 t + 0.5 a0 t^2."""
    name = "CA"
    def predict(self, ws):
        t = ws.PRED_T[None, :]                              # (1,HOR)
        v = ws.VT[:, None, :]; a = ws.AT[:, None, :]
        lon = v[..., 0] * t + 0.5 * a[..., 0] * t ** 2
        lat = v[..., 1] * t + 0.5 * a[..., 1] * t ** 2
        return np.stack([lat, lon], -1).astype(np.float32)


class IDM(Predictor):
    """Intelligent Driver Model (observed leader) longitudinal + constant-velocity lateral.
    Uses ws.LEADER (N,HOR) leader longitudinal positions in the canonical frame (nan = no leader)."""
    name = "IDM"
    def __init__(self, a=1.5, b=2.0, T=1.2, s0=2.0, delta=4.0, veh_len=5.0, clip=(-8.0, 4.0)):
        self.a, self.b, self.T, self.s0, self.delta = a, b, T, s0, delta
        self.veh_len, self.clip = veh_len, clip

    def predict(self, ws):
        N, HOR = ws.GT.shape[0], ws.GT.shape[1]
        out = np.zeros((N, HOR, 2), np.float32)
        out[:, :, 0] = ws.VT[:, None, 1] * ws.PRED_T[None, :]    # lateral = CV
        dt = float(ws.PRED_T[0])                                  # per-step time on the downsampled grid (0.2 s)
        lead = ws.LEADER                                          # (N, HOR) canonical lon, nan if none
        for i in range(N):
            v = max(float(ws.VT[i, 0]), 0.0); vdes = max(v + 2.0, 10.0); p = 0.0
            has = np.isfinite(lead[i]).any()
            ll = lead[i]
            for k in range(HOR):
                free = self.a * (1.0 - (v / max(vdes, 1.0)) ** self.delta)
                if has and np.isfinite(ll[k]):
                    s = max(ll[k] - p - self.veh_len, 0.1)
                    lv = (ll[k] - ll[k - 1]) / dt if k > 0 and np.isfinite(ll[k - 1]) else v
                    sstar = self.s0 + max(v * self.T + v * (v - lv) / (2 * np.sqrt(self.a * self.b)), 0.0)
                    acc = free - self.a * (sstar / s) ** 2
                else:
                    acc = free
                acc = float(np.clip(acc, *self.clip)); v = max(v + acc * dt, 0.0); p += v * dt
                out[i, k, 1] = p
        return out


REGISTRY = {"cv": CV, "ca": CA, "idm": IDM}
