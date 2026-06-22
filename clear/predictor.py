"""CLEAR — the model interface every predictor implements.

A predictor maps a WindowSet (struct-of-arrays of N windows) to predictions. The single-mode path
returns (N, HOR, 2) in the canonical frame, ordered [lateral, longitudinal] to match ground truth.
A multimodal model overrides predict_multi() to return (N, K, HOR, 2) plus optional (N, K) probs;
the default lifts a single mode to K=1 so the rest of the toolkit is mode-agnostic.
"""
from __future__ import annotations
import numpy as np


class Predictor:
    name = "predictor"

    def predict(self, ws) -> np.ndarray:
        """Single-mode prediction. Return (N, HOR, 2) [lat, long] in the canonical frame."""
        raise NotImplementedError

    def predict_multi(self, ws):
        """Multimodal prediction. Return (preds (N,K,HOR,2), probs (N,K) or None).
        Default: a single deterministic mode (K=1)."""
        p = self.predict(ws)                       # (N,HOR,2)
        return p[:, None, :, :], None
