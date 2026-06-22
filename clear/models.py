"""CLEAR — adapters for plugging trained deep models into the Eval Card.

A model is anything implementing Predictor.predict(ws) -> (N, HOR, 2) [lat, long] in the per-window
canonical frame. Below is a thin reference adapter that wraps an arbitrary callable, plus notes on
the official CS-LSTM / STDAN repos.

CS-LSTM / STDAN: these consume a 13x3 convolutional social grid of neighbour histories that the
current WindowSet does not carry (it stores target kinematics + the in-lane leader, which is all the
CV/CA/IDM baselines and the admissibility metric need). The full leakage-free CS-LSTM ladder
(leaky-vs-clean, on highD and exiD) is implemented in scripts/reproduction_ladder.py and
scripts/exid_reproduction_ladder.py; folding the neighbour grid into WindowSet so those run *inside*
CLEAR is a v0.3 item. For now, wrap any model that can already produce per-window trajectories:
"""
from __future__ import annotations
import numpy as np
from .predictor import Predictor


class CallableModel(Predictor):
    """Wrap a function f(ws) -> (N, HOR, 2) as a CLEAR Predictor.
        model = CallableModel("MyNet", lambda ws: my_net_forward(ws))
        card  = evaluate(ws, models=[model])
    """
    def __init__(self, name, fn, fn_multi=None):
        self.name = name; self._fn = fn; self._fn_multi = fn_multi

    def predict(self, ws):
        return np.asarray(self._fn(ws), np.float32)

    def predict_multi(self, ws):
        if self._fn_multi is None:
            return super().predict_multi(ws)
        preds, probs = self._fn_multi(ws)
        return np.asarray(preds, np.float32), (None if probs is None else np.asarray(probs, np.float32))
