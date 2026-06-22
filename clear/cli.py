"""CLEAR CLI.  Usage:
  python -m clear.cli run --data exiD-dataset-v2.1/data --split recording --out card/
  python -m clear.cli run --data highD-dataset-v1.0/data --split random --nrec 10 --out card_highd/
Loads a levelX dataset, runs the CV/CA/IDM baseline ladder under a leakage-free split, and writes
evalcard.json + evalcard.md.
"""
import argparse, sys
from .data import load_levelx
from .card import evaluate


def main(argv=None):
    try: sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    except Exception: pass
    ap = argparse.ArgumentParser(prog="clear", description="Leakage-free highway-prediction Eval Card")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="load a dataset and emit an Eval Card")
    r.add_argument("--data", required=True, help="path to a levelX *_tracks.csv directory")
    r.add_argument("--split", default="recording", choices=["random", "recording", "location", "vehicle"])
    r.add_argument("--cap", type=int, default=300000); r.add_argument("--nrec", type=int, default=0)
    r.add_argument("--fps", type=float, default=25.0); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=None, help="output directory for evalcard.json/.md")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(f"loading {a.data} (cap={a.cap}, nrec={a.nrec}) ...", flush=True)
        ws = load_levelx(a.data, fps=a.fps, cap=a.cap, nrec=a.nrec)
        print(f"  {len(ws):,} windows; running Eval Card under '{a.split}' split ...", flush=True)
        card = evaluate(ws, split=a.split, seed=a.seed, out_dir=a.out)
        print(card.to_markdown())
        if a.out: print(f"\nwrote {a.out}/evalcard.json and evalcard.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
