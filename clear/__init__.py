"""CLEAR — Clean, Leakage-free Evaluation And Reporting for highway trajectory prediction.

Quick start:
    from clear import load_levelx, evaluate
    ws   = load_levelx("exiD-dataset-v2.1/data", nrec=5)     # or highD-dataset-v1.0/data
    card = evaluate(ws, split="recording", out_dir="card/")  # CV/CA/IDM + your models
    print(card.to_markdown()); print(card.beats_CA)

Plug in your own model by subclassing Predictor and passing it in models=[...]:
    class MyModel(Predictor):
        def predict(self, ws): return ...   # (N, HOR, 2) [lat, long] canonical frame
"""
from .predictor import Predictor
from .baselines import CV, CA, IDM
from .data import load_levelx, load_ngsim, WindowSet
from .splits import make_split, leakage_check
from .card import evaluate, EvalCard
from .models import CallableModel

__version__ = "0.2.0"
__all__ = ["Predictor", "CV", "CA", "IDM", "load_levelx", "load_ngsim", "WindowSet",
           "make_split", "leakage_check", "evaluate", "EvalCard", "CallableModel"]
