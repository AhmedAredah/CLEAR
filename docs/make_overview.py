"""Generate docs/clear_overview.{png,pdf}: a one-glance schematic of what/how/why CLEAR works."""
import os
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OK = dict(sky="#56B4E9", orange="#E69F00", green="#009E73", verm="#D55E00", grey="#5a5a5a",
          blue="#0072B2", pale="#EAF3FA", dark="#222222")
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})


def box(ax, x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h - 0.052, title, ha="center", va="top", fontsize=9.5, fontweight="bold",
            color=OK["dark"], zorder=3)
    ax.text(x + w / 2, y + h - 0.115, "\n".join(lines), ha="center", va="top", fontsize=7.6,
            color="#333333", zorder=3)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.6, color=OK["grey"], zorder=1))


fig, ax = plt.subplots(figsize=(10.2, 4.7)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.5, 0.975, "CLEAR — Clean, Leakage-free Evaluation And Reporting", ha="center",
        fontsize=13.5, fontweight="bold", color=OK["dark"])
ax.text(0.5, 0.93, "an honest, leakage-free evaluation harness for highway trajectory prediction",
        ha="center", fontsize=9, color=OK["grey"], style="italic")

# --- INPUTS ---
box(ax, 0.02, 0.46, 0.22, 0.30, "1 · INPUTS",
    ["Dataset (highD / exiD /", "NGSIM) — you provide,", "CLEAR ships none", "",
     "Your model:", "predict(ws) -> (N,T,2)"], OK["pale"], OK["blue"])
# --- PIPELINE ---
box(ax, 0.30, 0.40, 0.34, 0.42, "2 · CLEAR PIPELINE",
    ["• canonical per-window frame", "  (handles curved ramps)", "",
     "• leakage-safe split:", "  recording / vehicle-disjoint", "  -> PASS-CLEAN or WARN-LEAKY", "",
     "• honest baselines CV / CA / IDM", "• metrics: FDE/RMSE, CIs,", "  CA win-rate, strata, best-of-k,", "  predicted-collision"],
    "#F4FBF7", OK["green"])
# --- OUTPUT: Eval Card ---
box(ax, 0.70, 0.40, 0.28, 0.42, "3 · EVAL CARD",
    ["Protocol + leakage verdict", "Accuracy ladder vs CA", "Stratified error (cruise/hard)",
     "Admissibility (collisions)", "Best-of-k transparency", "", "5-point disclosure checklist",
     "JSON · Markdown · figure"], "#FEF6E9", OK["orange"])
arrow(ax, 0.245, 0.61, 0.298, 0.61)
arrow(ax, 0.645, 0.61, 0.698, 0.61)

# --- WHY strip ---
ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.27, boxstyle="round,pad=0.01,rounding_size=0.02",
                            linewidth=1.0, edgecolor=OK["grey"], facecolor="#F7F7F7", zorder=1))
ax.text(0.06, 0.275, "WHY", fontsize=10, fontweight="bold", color=OK["verm"], va="top")
whys = [
    ("Leakage inflates.", "Sliding windows share vehicles/recordings across folds; CLEAR forbids it by default."),
    ("CA is a hard baseline.", "A zero-parameter constant-acceleration model is unbeaten on clean highway data."),
    ("Averaging hides the tail.", "~77% of windows are trivial cruising; one RMSE buries the safety-critical cases."),
    ("Accuracy != safety.", "The lowest-error model can forecast the most rear-end collisions; CLEAR measures both."),
]
y0 = 0.225
for i, (h, b) in enumerate(whys):
    yy = y0 - i * 0.05
    ax.text(0.085, yy, "●", fontsize=7, color=OK["verm"], va="center")
    ax.text(0.10, yy, h, fontsize=8.4, fontweight="bold", color=OK["dark"], va="center")
    ax.text(0.32, yy, b, fontsize=8.0, color="#333333", va="center")

out = os.path.dirname(__file__)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"clear_overview.{ext}"))
print("wrote docs/clear_overview.png and .pdf")
