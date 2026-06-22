"""CLEAR — adapters that run the official deep models *inside* the Eval Card.

CS-LSTM (Deo & Trivedi, conv-social-pooling) and STDAN consume a GLON x GLAT convolutional social
grid of neighbour histories. CLEAR builds that grid when a dataset is loaded with social_grid=True
(WindowSet.NBR + .THIST); this adapter assembles the model's batched input from it and returns
predictions in CLEAR's canonical [lat, long] frame, so a trained official model is graded against
CA by the same Eval Card as any other predictor.

CLEAR does not vendor the third-party model code (licensing). Clone the official repo and point the
adapter at it:

    git clone https://github.com/nachiket92/conv-social-pooling external/conv-social-pooling
    from clear import load_levelx, evaluate
    from clear.deep import CSLSTM
    ws    = load_levelx("highD/data", social_grid=True)
    model = CSLSTM(repo="external/conv-social-pooling", weights="cslstm.tar")  # trained weights
    card  = evaluate(ws, models=[model], split="recording")
    print(card.beats_CA)   # {'CS-LSTM': False}  -- on clean highway data, CA wins

Requires torch (`pip install clear-eval[deep]`).
"""
from __future__ import annotations
import os, sys
import numpy as np
from .predictor import Predictor

SC = 30.0   # CS-LSTM normalisation (metres)


def _require_grid(ws):
    if ws.NBR is None or ws.THIST is None:
        raise ValueError("deep adapters need a social grid: load the dataset with "
                         "load_levelx(..., social_grid=True).")


class CSLSTM(Predictor):
    """Official conv-social-pooling highwayNet, run for inference inside CLEAR.

    repo    : path to a clone of the conv-social-pooling repository (provides model.highwayNet)
    weights : optional path to trained weights (.tar/.pt state_dict); random init if None
    """
    name = "CS-LSTM"

    def __init__(self, repo, weights=None, batch=512, device=None):
        import torch
        self.torch = torch
        self._repo = os.path.abspath(repo); sys.path.insert(0, self._repo)
        self._weights = weights; self.batch = batch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.glon, self.glat = 13, 3; self._net = None

    def _build_net(self, ws):
        from model import highwayNet                       # from the cloned repo on sys.path
        glon, glat = (ws.grid or (self.glon, self.glat))
        args = dict(use_cuda=(self.device.type == "cuda"), encoder_size=64, decoder_size=128,
                    in_length=ws.NHIST, out_length=ws.HOR, grid_size=(glon, glat),
                    soc_conv_depth=64, conv_3x1_depth=16, dyn_embedding_size=32, input_embedding_size=32,
                    num_lat_classes=3, num_lon_classes=2, use_maneuvers=False, train_flag=False)
        net = highwayNet(args).to(self.device)
        if self._weights and os.path.exists(self._weights):
            net.load_state_dict(self.torch.load(self._weights, map_location=self.device))
        net.eval(); net.train_flag = False
        return net

    def predict(self, ws):
        _require_grid(ws)
        torch = self.torch; glon, glat = ws.grid
        if self._net is None: self._net = self._build_net(ws)
        N = len(ws); out = np.zeros((N, ws.HOR, 2), np.float32)
        for i0 in range(0, N, self.batch):
            bi = range(i0, min(i0 + self.batch, N)); bs = len(bi)
            nbc = max(sum(len(ws.NBR[i]) for i in bi), 1)
            hb = torch.zeros(ws.NHIST, bs, 2); nb = torch.zeros(ws.NHIST, nbc, 2)
            mb = torch.zeros(bs, glat, glon, 64, dtype=torch.bool); c = 0
            le = torch.zeros(bs, 3); lo = torch.zeros(bs, 2)
            for s, i in enumerate(bi):
                hb[:, s, :] = torch.from_numpy(ws.THIST[i]) / SC
                for cell, h in ws.NBR[i].items():
                    nb[:, c, :] = torch.from_numpy(h) / SC; mb[s, cell // glon, cell % glon, :] = True; c += 1
            with torch.no_grad():
                fp = self._net(hb.to(self.device), nb.to(self.device), mb.to(self.device),
                               le.to(self.device), lo.to(self.device))[:, :, :2].cpu().numpy() * SC
            out[i0:i0 + bs] = np.transpose(fp, (1, 0, 2))
        return out
