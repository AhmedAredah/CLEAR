"""CLEAR quickstart. Point --data at a levelX *_tracks.csv directory (highD/exiD) you obtained
under their licenses (CLEAR ships no data). Emits an Eval Card and grades a toy model against CA.

    python examples/quickstart.py --data /path/to/highD/data
"""
import argparse
import numpy as np
from clear import load_levelx, evaluate, Predictor, CA


class ToyModel(Predictor):
    """A deliberately weak learned-style model: CA plus a little lateral noise."""
    name = "ToyModel"
    def predict(self, ws):
        p = CA().predict(ws).copy()
        p[:, :, 0] += 0.3 * np.sin(np.linspace(0, 3, ws.HOR))[None, :]   # wobble in lateral
        return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="levelX *_tracks.csv directory")
    ap.add_argument("--split", default="recording")
    ap.add_argument("--out", default="card")
    a = ap.parse_args()

    ws = load_levelx(a.data)
    card = evaluate(ws, models=[ToyModel()], split=a.split, out_dir=a.out)
    print(card.to_markdown())
    print("\nDid the toy model beat CA?", card.beats_CA)
    try:
        card.to_figure(f"{a.out}/evalcard.pdf"); print(f"figure -> {a.out}/evalcard.pdf")
    except Exception as e:
        print(f"(figure needs matplotlib: {e})")
