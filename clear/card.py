"""CLEAR — the Eval Card: a standardized, disclosure-forcing evaluation report.

evaluate() runs a set of predictors (CV and CA are always included as references) on a leakage-free
split and assembles: the protocol disclosure + leakage check, a baseline ladder (FDE/RMSE@1-5s with
bootstrap CIs and the paired CA win-rate), stratified error, a best-of-k sweep for multimodal
models, and a five-point reporting checklist. The checklist operationalizes the paper's reporting
recommendations: it is RED until each disclosure is satisfied.
"""
from __future__ import annotations
import os, json
from dataclasses import dataclass, field, asdict
import numpy as np
from .baselines import CV, CA, IDM
from .splits import make_split
from . import metrics as M


@dataclass
class EvalCard:
    dataset: str
    protocol: dict
    leakage: dict
    ladder: dict                      # model -> {horizon rmse/fde/ci, ca_winrate, beats_ca}
    strata: dict                      # model -> stratum -> rmse/share
    admissibility: dict               # model -> {all_pct, hard_pct} predicted-collision rate
    bestofk: dict                     # model -> {k: rmse} (multimodal only)
    checklist: dict
    horizons_s: list

    def to_json(self, path):
        json.dump(asdict(self), open(path, "w"), indent=2); return path

    @property
    def beats_CA(self):               # did any non-baseline model beat CA at 5 s?
        return {m: v["beats_ca_5s"] for m, v in self.ladder.items() if m not in ("CV", "CA", "IDM")}

    def to_figure(self, path, title=None):
        """Render the Eval Card as a publication-style figure: RMSE@5s ladder + predicted-collision
        + the leakage/checklist banner. Requires matplotlib."""
        import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
        OK = dict(sky="#56B4E9", orange="#E69F00", green="#009E73", verm="#D55E00", grey="#5a5a5a", blue="#0072B2")
        col = {"CV": OK["sky"], "CA": OK["orange"], "IDM": OK["green"]}
        mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
            "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42})
        models = list(self.ladder.keys()); h5 = str(self.horizons_s[-1])
        rmse = [self.ladder[m]["fde"][h5]["rmse"] for m in models]
        coll_all = [self.admissibility[m]["all_pct"] for m in models]
        coll_hard = [self.admissibility[m]["hard_pct"] or 0 for m in models]
        cols = [col.get(m, OK["blue"]) for m in models]
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
        x = range(len(models))
        ax[0].bar(x, rmse, color=cols, edgecolor="k", lw=0.5)
        for i, v in zip(x, rmse): ax[0].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=7)
        ax[0].set_xticks(list(x)); ax[0].set_xticklabels(models, fontsize=7)
        ax[0].set_ylabel("RMSE@5 s (m)"); ax[0].set_title("Accuracy ladder (CA = honest baseline)", fontsize=8)
        ax[0].yaxis.grid(True, ls=":", lw=0.5, color="#bcbcbc"); ax[0].set_axisbelow(True)
        w = 0.38
        ax[1].bar([i - w / 2 for i in x], coll_all, w, color=cols, edgecolor="k", lw=0.5, alpha=0.55)
        ax[1].bar([i + w / 2 for i in x], coll_hard, w, color=cols, edgecolor="k", lw=0.5)
        ax[1].set_xticks(list(x)); ax[1].set_xticklabels(models, fontsize=7)
        ax[1].set_ylabel("predicted-collision (%)")
        ax[1].set_title("Admissibility (light=all, solid=hard)", fontsize=8)
        ax[1].yaxis.grid(True, ls=":", lw=0.5, color="#bcbcbc"); ax[1].set_axisbelow(True)
        passed = sum(self.checklist.values())
        banner = (f"CLEAR Eval Card — {self.dataset}   |   split: {self.leakage['kind']} "
                  f"({self.leakage['verdict']})   |   checklist {passed}/{len(self.checklist)}")
        fig.suptitle(title or banner, fontsize=8.5, y=1.04)
        fig.tight_layout(w_pad=1.5); fig.savefig(path); plt.close(fig)
        return path

    def to_markdown(self, path=None):
        L = self.protocol; lk = self.leakage
        out = [f"# CLEAR Eval Card — {self.dataset}", ""]
        out += ["## 1. Protocol disclosure",
                f"- windows: **{L['n_windows']:,}** | observe {L['obs_s']}s -> predict {L['pred_s']}s | {L['fps']} Hz",
                f"- split: **{lk['kind']}** -> leakage check: **{lk['verdict']}** "
                f"(recordings shared {lk['recordings_shared']}, vehicles shared {lk['vehicles_shared']})",
                f"- scoring: **{L['scoring']}** | horizons: {self.horizons_s} s", ""]
        out += ["## 2. Baseline ladder — FDE @5 s (mean / median) and RMSE, CA win-rate", "",
                "| model | FDE@5 mean [95% CI] | median | RMSE@5 | skill/CV | CA wins % | beats CA? |",
                "|---|---|---|---|---|---|---|"]
        for m, v in self.ladder.items():
            f5 = v["fde"]["5"]; ci = f5["ci"]
            cw = v.get("ca_winrate", {}).get("ca_win_pct", "—")
            beats = "—" if m in ("CV", "CA", "IDM") else ("**YES**" if v["beats_ca_5s"] else "no")
            out.append(f"| {m} | {f5['fde_mean']} [{ci[0]},{ci[1]}] | {f5['fde_median']} | {f5['rmse']} | "
                       f"{v.get('skill_cv_5s','—')} | {cw} | {beats} |")
        out += ["", "## 3. Stratified error — RMSE@5 s (share of windows)", "",
                "| model | cruise | mild | hard | lane-keep | lane-change |", "|---|---|---|---|---|---|"]
        for m, st in self.strata.items():
            cell = lambda k: f"{st[k]['rmse']} ({st[k]['share_pct']}%)" if k in st else "—"
            out.append(f"| {m} | {cell('cruise')} | {cell('mild')} | {cell('hard')} | "
                       f"{cell('lane-keep')} | {cell('lane-change')} |")
        out += ["", "## 4. Admissibility — predicted rear-end-collision rate (%)", "",
                "| model | all windows | hard braking |", "|---|---|---|"]
        for m, ad in self.admissibility.items():
            out.append(f"| {m} | {ad['all_pct']} | {ad['hard_pct'] if ad['hard_pct'] is not None else '—'} |")
        out += ["", "## 5. Scoring transparency — best-of-k RMSE@5 s", ""]
        if self.bestofk:
            for m, sw in self.bestofk.items():
                out.append(f"- **{m}**: " + ", ".join(f"k={k}:{r}" for k, r in sw.items())
                           + f"  (CA single-mode = {self.ladder['CA']['fde']['5']['rmse']})")
        else:
            out.append("- single-mode only (no multimodal model evaluated)")
        out += ["", "## 6. Reporting checklist"]
        for k, ok in self.checklist.items():
            out.append(f"- [{'x' if ok else ' '}] {k}")
        md = "\n".join(out)
        if path: open(path, "w", encoding="utf-8").write(md)
        return md


def evaluate(ws, models=None, split="recording", scoring="single-mode", seed=0,
             horizons_s=(1, 2, 3, 4, 5), out_dir=None):
    """Run CV/CA(/IDM) + user models on a leakage-free split; return an EvalCard."""
    HZ = [int(round(h / max(horizons_s) * ws.HOR)) for h in horizons_s]   # step index per horizon-second
    tr, te, lk = make_split(ws, split, seed=seed)
    sub = _subset(ws, te)
    preds = {"CV": CV(), "CA": CA(), "IDM": IDM()}
    user = {}
    for m in (models or []):
        user[getattr(m, "name", m.__class__.__name__)] = m
    allm = {**preds, **user}
    ca_pred = CA().predict(sub)
    ladder, strata, admis, bestofk = {}, {}, {}, {}
    for name, mdl in allm.items():
        p = mdl.predict(sub)
        he = {str(h): M.horizon_errors(p, sub.GT, [hz])[hz] for h, hz in zip(horizons_s, HZ)}
        cw = M.paired_ca_winrate(p, ca_pred, sub.GT, HZ[-1]) if name != "CA" else {}
        cv_rmse = M.horizon_errors(CV().predict(sub), sub.GT, [HZ[-1]])[HZ[-1]]["rmse"]
        skill = round(1 - he[str(horizons_s[-1])]["rmse"] / cv_rmse, 3) if cv_rmse else None
        ladder[name] = dict(fde=he, ca_winrate=cw, skill_cv_5s=skill,
                            beats_ca_5s=bool(he[str(horizons_s[-1])]["rmse"] <
                                             ladder.get("CA", {}).get("fde", {}).get(str(horizons_s[-1]),
                                             {"rmse": 1e9})["rmse"]) if "CA" in ladder else False)
        strata[name] = M.stratified(p, sub.GT, sub, HZ[-1])
        admis[name] = M.predicted_collision(p, sub)
        pm, pr = mdl.predict_multi(sub)
        if pm.shape[1] > 1:
            bestofk[name] = M.best_of_k(pm, pr, sub.GT, HZ[-1])
    # fix beats_ca now that CA is present
    ca5 = ladder["CA"]["fde"][str(horizons_s[-1])]["rmse"]
    for name, v in ladder.items():
        v["beats_ca_5s"] = bool(name not in ("CV", "CA", "IDM") and v["fde"][str(horizons_s[-1])]["rmse"] < ca5)
    checklist = {
        "Constant-acceleration baseline reported": "CA" in ladder,
        "Split is recording-/vehicle-disjoint (no leakage)": not lk["leaky"],
        "Scoring convention stated (single-mode vs best-of-k)": True,
        "Difficulty-stratified error reported": all(len(s) > 0 for s in strata.values()),
        "Admissibility metric reported (predicted-collision)": len(admis) > 0,
    }
    proto = dict(n_windows=len(sub), obs_s=3.0, pred_s=float(max(horizons_s)), fps=ws.FPS, scoring=scoring)
    card = EvalCard(ws.name, proto, lk, ladder, strata, admis, bestofk, checklist, list(horizons_s))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        card.to_json(os.path.join(out_dir, "evalcard.json"))
        card.to_markdown(os.path.join(out_dir, "evalcard.md"))
    return card


def _subset(ws, idx):
    from .data import WindowSet
    nbr = [ws.NBR[i] for i in idx] if ws.NBR is not None else None
    thist = [ws.THIST[i] for i in idx] if ws.THIST is not None else None
    return WindowSet(ws.VT[idx], ws.AT[idx], ws.GT[idx], ws.LEADER[idx], ws.KIN[idx], ws.LC[idx],
                     ws.REC[idx], ws.LOC[idx], ws.VID[idx], ws.FPS, ws.HOR, ws.PRED_T, ws.name,
                     NBR=nbr, THIST=thist, grid=ws.grid, NHIST=ws.NHIST)
