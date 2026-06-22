"""CLEAR — reproduce the paper's baseline evaluation on all three datasets with one command.

    uv run --with numpy --with matplotlib python -m clear.repro

Writes an Eval Card (json + md + pdf) per dataset under repro_cards/. The CV/CA/IDM ladders,
stratified error, and admissibility here are the clean, leakage-free numbers the paper reports;
deep models are added by passing models=[...] to clear.evaluate (see README).
"""
import os
from .data import load_levelx, load_ngsim
from .card import evaluate

DATASETS = [
    ("highD", lambda: load_levelx("highD-dataset-v1.0/data"), "recording"),
    ("exiD",  lambda: load_levelx("exiD-dataset-v2.1/data"),  "recording"),
    ("NGSIM", lambda: load_ngsim(
        "NGSIM/Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260620.csv"),
        "vehicle"),
]


def main():
    os.makedirs("repro_cards", exist_ok=True)
    for name, loader, split in DATASETS:
        try:
            print(f"[{name}] loading ...", flush=True); ws = loader()
            print(f"[{name}] {len(ws):,} windows; evaluating under '{split}' split ...", flush=True)
            card = evaluate(ws, split=split, out_dir=f"repro_cards/{name}")
            try: card.to_figure(f"repro_cards/{name}/evalcard.pdf")
            except Exception as e: print(f"  (figure skipped: {e})")
            L = card.ladder
            print(f"  RMSE@5s  CV {L['CV']['fde']['5']['rmse']} | CA {L['CA']['fde']['5']['rmse']} | "
                  f"IDM {L['IDM']['fde']['5']['rmse']}  | leakage {card.leakage['verdict']} | "
                  f"checklist {sum(card.checklist.values())}/{len(card.checklist)}", flush=True)
        except FileNotFoundError as e:
            print(f"[{name}] SKIPPED (data not found: {e})", flush=True)
    print("Wrote repro_cards/<dataset>/evalcard.{json,md,pdf}")


if __name__ == "__main__":
    main()
