"""Generate docs/clear_overview.{png,pdf}: a clean, professional one-glance schematic of CLEAR."""
import os
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# Okabe-Ito accents
BLUE, GREEN, ORANGE, VERM, GREY = "#0072B2", "#009E73", "#E69F00", "#D55E00", "#5A5A5A"
INK, FAINT, LINE = "#1A1A1A", "#FBFCFD", "#D9DEE3"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "savefig.dpi": 220, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(11.4, 5.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ---------- title ----------
ax.text(0.5, 0.985, "CLEAR", ha="center", va="top", fontsize=18, fontweight="bold", color=INK)
ax.text(0.5, 0.905, "Clean, Leakage-free Evaluation And Reporting", ha="center", va="top",
        fontsize=11, color=INK)
ax.text(0.5, 0.862, "an honest evaluation harness for highway trajectory prediction", ha="center",
        va="top", fontsize=9.3, color=GREY, style="italic")

# ---------- three stage cards (equal height, aligned) ----------
CARD_Y, CARD_H = 0.385, 0.42
cards = [
    (0.025, 0.285, BLUE, "1", "INPUTS", [
        ("A dataset", "highD · exiD · NGSIM"),
        ("", "(you provide it — CLEAR ships none)"),
        ("Your model", "subclass Predictor"),
        ("", "predict(ws) → (N, T, 2)"),
    ]),
    (0.357, 0.286, GREEN, "2", "CLEAR PIPELINE", [
        ("Canonical frame", "per-window, handles ramps"),
        ("Leakage-safe split", "+ PASS / WARN verdict"),
        ("CV · CA · IDM", "honest baselines"),
        ("Metrics", "FDE/RMSE · CIs · CA win-rate"),
        ("", "strata · best-of-k · collisions"),
    ]),
    (0.690, 0.285, ORANGE, "3", "EVAL CARD", [
        ("Protocol", "+ leakage verdict"),
        ("Accuracy ladder", "every model vs CA"),
        ("Stratified error", "cruise / mild / hard"),
        ("Admissibility", "predicted collisions"),
        ("5-point checklist", "JSON · Markdown · figure"),
    ]),
]


def card(x, w, accent, num, title, rows):
    ax.add_patch(FancyBboxPatch((x, CARD_Y), w, CARD_H, boxstyle="round,pad=0.006,rounding_size=0.018",
                                linewidth=1.1, edgecolor=LINE, facecolor=FAINT, zorder=2))
    cy = CARD_Y + CARD_H - 0.052
    ax.add_patch(Circle((x + 0.032, cy + 0.004), 0.0185, color=accent, zorder=4))
    ax.text(x + 0.032, cy + 0.004, num, ha="center", va="center", fontsize=10, fontweight="bold",
            color="white", zorder=5)
    ax.text(x + 0.064, cy, title, ha="left", va="center", fontsize=11.5, fontweight="bold", color=accent,
            zorder=4)
    ax.plot([x + 0.022, x + w - 0.022], [CARD_Y + CARD_H - 0.092] * 2, color=LINE, lw=1.0, zorder=3)
    ty = CARD_Y + CARD_H - 0.135; step = (CARD_H - 0.175) / max(len(rows), 1)
    for lead, body in rows:
        if lead:
            ax.text(x + 0.024, ty, "▪", ha="left", va="center", fontsize=7, color=accent, zorder=4)
            ax.text(x + 0.044, ty, lead, ha="left", va="center", fontsize=9.2, fontweight="bold",
                    color=INK, zorder=4)
            ax.text(x + 0.044, ty - 0.028, body, ha="left", va="center", fontsize=8.4, color="#444",
                    zorder=4)
            ty -= step * 1.42
        else:
            ax.text(x + 0.044, ty + 0.006, body, ha="left", va="center", fontsize=8.4, color="#444",
                    style="italic", zorder=4)
            ty -= step * 0.92


for x, w, accent, num, title, rows in cards:
    card(x, w, accent, num, title, rows)

# arrows between cards
for x0, x1 in [(0.310, 0.357), (0.643, 0.690)]:
    ax.add_patch(FancyArrowPatch((x0, CARD_Y + CARD_H / 2), (x1, CARD_Y + CARD_H / 2),
                                 arrowstyle="-|>", mutation_scale=18, linewidth=2.0, color=GREY, zorder=1))

# ---------- WHY band ----------
ax.add_patch(FancyBboxPatch((0.025, 0.045), 0.95, 0.265, boxstyle="round,pad=0.006,rounding_size=0.018",
                            linewidth=1.1, edgecolor=LINE, facecolor="#F4F6F8", zorder=1))
ax.text(0.05, 0.275, "WHY IT MATTERS", ha="left", va="center", fontsize=10.5, fontweight="bold",
        color=VERM)
whys = [
    ("Leakage inflates.", "Sliding windows share vehicles and recordings across", "folds; CLEAR forbids it by default."),
    ("CA is a hard baseline.", "A zero-parameter constant-acceleration model is", "unbeaten on clean highway data."),
    ("Averaging hides the tail.", "~77% of windows are trivial cruising; one RMSE", "buries the safety-critical cases."),
    ("Accuracy ≠ safety.", "The lowest-error model can forecast the most rear-end", "collisions — CLEAR measures both."),
]
xcol = [0.055, 0.530]; yrow = [0.205, 0.095]
for i, (lead, l1, l2) in enumerate(whys):
    x = xcol[i % 2]; y = yrow[i // 2]
    ax.add_patch(Circle((x + 0.006, y + 0.028), 0.007, color=[BLUE, GREEN, ORANGE, VERM][i], zorder=3))
    ax.text(x + 0.024, y + 0.028, lead, ha="left", va="center", fontsize=9.6, fontweight="bold", color=INK)
    ax.text(x + 0.024, y, l1, ha="left", va="center", fontsize=8.6, color="#3A3A3A")
    ax.text(x + 0.024, y - 0.027, l2, ha="left", va="center", fontsize=8.6, color="#3A3A3A")

ax.text(0.5, 0.012, "github.com/AhmedAredah/CLEAR  ·  BSD-3-Clause", ha="center", va="center",
        fontsize=8, color=GREY)

out = os.path.dirname(__file__)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"clear_overview.{ext}"))
print("wrote docs/clear_overview.png and .pdf")
