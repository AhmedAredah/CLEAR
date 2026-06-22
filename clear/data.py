"""CLEAR — data adapters. Loads a levelX dataset (highD / exiD; the inD/rounD/uniD family share the
format) into a WindowSet: a struct-of-arrays of leakage-free 3 s -> 5 s windows in a per-window
CANONICAL frame (origin at the agent, x-axis along its heading at the prediction time). The same
frame handles straight highways (highD) and curved ramps (exiD).

NGSIM (US-101/I-80) uses a different CSV (feet, frame-interleaved); a thin adapter is a documented
extension point (see load_ngsim).
"""
from __future__ import annotations
import os, csv, glob, math, random, collections
from dataclasses import dataclass
import numpy as np


@dataclass
class WindowSet:
    VT: np.ndarray            # (N,2) canonical [vlon, vlat] at t0
    AT: np.ndarray            # (N,2) canonical [alon, alat] at t0
    GT: np.ndarray            # (N,HOR,2) canonical future [lat, long]
    LEADER: np.ndarray        # (N,HOR) leader longitudinal in canonical frame (nan = none)
    KIN: np.ndarray           # (N,) kinematic regime 0=cruise 1=mild 2=hard
    LC: np.ndarray            # (N,) lane-change flag
    REC: np.ndarray           # (N,) recording id
    LOC: np.ndarray           # (N,) location id
    VID: np.ndarray           # (N,) global vehicle id (for vehicle-level split)
    FPS: float
    HOR: int
    PRED_T: np.ndarray        # (HOR,) horizon times (s)
    name: str = "dataset"
    NBR: list = None          # optional CS-LSTM social grid: per-window dict {cell_index: (NHIST,2) hist}
    THIST: list = None        # optional target history per window: (NHIST,2) [lat,long] canonical
    grid: tuple = None        # (GLON, GLAT) of the social grid, if NBR is populated
    NHIST: int = None         # number of observed steps (for deep adapters)

    def __len__(self): return len(self.VT)


def _schema(header):
    h = set(header)
    return dict(
        tid="trackId" if "trackId" in h else "id",
        x="xCenter" if "xCenter" in h else "x",
        y="yCenter" if "yCenter" in h else "y",
        heading="heading" if "heading" in h else None,
        lead="leadId" if "leadId" in h else ("precedingId" if "precedingId" in h else None),
        lanechg="laneChange" if "laneChange" in h else None,
        lane="laneId" if "laneId" in h else None,
    )


def load_levelx(data_dir, fps=25.0, obs_s=3.0, pred_s=5.0, stride_s=1.0, ds=5,
                cruise=0.5, hard=1.5, cap=300000, nrec=0, seed=3, lane_w=3.5,
                social_grid=False, glon=13, glat=3, cell=15 * 0.3048):
    """Stream *_tracks.csv -> WindowSet. ds = frame downsample (25 Hz / 5 -> 5 Hz internal grid).
    social_grid=True additionally builds the CS-LSTM GLON x GLAT neighbour grid per window (for the
    deep adapters in clear.deep), at extra cost; the baselines/Eval Card do not need it."""
    random.seed(seed)
    HIST = int(round(obs_s * fps)); HOR = int(round(pred_s * fps)); STRIDE = int(round(stride_s * fps))
    obs_idx = np.arange(-HIST, 1, ds); fut_idx = np.arange(ds, HOR + 1, ds)
    NHIST = len(obs_idx); NFUT = len(fut_idx); PRED_T = fut_idx / fps
    files = sorted(glob.glob(os.path.join(data_dir, "*_tracks.csv")))
    if nrec > 0: files = files[:nrec]
    if not files: raise FileNotFoundError(f"no *_tracks.csv in {data_dir}")
    VT, AT, GT, LEAD, KIN, LC, REC, LOC, VID, NBR, THIST = [], [], [], [], [], [], [], [], [], [], []
    vid_base = 0
    def pf(v):  # NaN-safe float (lane cols can be empty off-lanelet)
        try: return float(v)
        except (ValueError, TypeError): return math.nan
    for tf in files:
        rec = int(os.path.basename(tf)[:2])
        mp = tf.replace("_tracks.csv", "_recordingMeta.csv")
        loc = int(next(csv.DictReader(open(mp)))["locationId"]) if os.path.exists(mp) else 0
        with open(tf) as f:
            rd = csv.reader(f); header = next(rd); ix = {c: i for i, c in enumerate(header)}; S = _schema(header)
            has_lc_col = S["lanechg"] is not None; has_lane_col = S["lane"] is not None
            rows_by = collections.defaultdict(list); frame_pos = collections.defaultdict(list)
            for row in rd:
                tid = int(row[ix[S["tid"]]]); fr = int(row[ix["frame"]])
                x = float(row[ix[S["x"]]]); y = float(row[ix[S["y"]]])
                vx = float(row[ix["xVelocity"]]); vy = float(row[ix["yVelocity"]])
                ax = float(row[ix["xAcceleration"]]); ay = float(row[ix["yAcceleration"]])
                hd = math.radians(float(row[ix[S["heading"]]])) if S["heading"] else math.atan2(vy, vx)
                lead = int(row[ix[S["lead"]]]) if S["lead"] else -1
                lc = pf(row[ix[S["lanechg"]]]) if S["lanechg"] else 0.0
                lane = int(row[ix[S["lane"]]]) if S["lane"] else 0
                rows_by[tid].append((fr, x, y, hd, vx, vy, ax, ay, lead, lc, lane))
                frame_pos[fr].append((tid, x, y))
        A = {}
        for tid, r in rows_by.items():
            r.sort(); A[tid] = (int(r[0][0]), np.array([row[:8] + (row[9], row[10]) for row in r], float),
                                [row[8] for row in r])   # (f0, num-array[fr,x,y,hd,vx,vy,ax,ay,lc,lane], leadids)
        for tid, (f0, a, leadids) in A.items():
            n = len(a)
            for ti in range(HIST, n - HOR, STRIDE):
                if a[ti + HOR, 0] - a[ti - HIST, 0] != HIST + HOR:   # contiguous block
                    continue
                xo, yo, h0 = a[ti, 1], a[ti, 2], a[ti, 3]; c, s = math.cos(h0), math.sin(h0)
                vx, vy, ax, ay = a[ti, 4], a[ti, 5], a[ti, 6], a[ti, 7]
                vlon = c * vx + s * vy; vlat = -s * vx + c * vy
                alon = c * ax + s * ay; alat = -s * ax + c * ay
                fsub = a[ti + fut_idx]; dx = fsub[:, 1] - xo; dy = fsub[:, 2] - yo
                gt = np.stack([-s * dx + c * dy, c * dx + s * dy], 1)        # [lat, long]
                # leader longitudinal in canonical frame
                lead_lon = np.full(NFUT, np.nan); lid = leadids[ti]
                if lid != -1 and lid in A:
                    lf0, la, _ = A[lid]; jj = int(a[ti, 0]) + fut_idx - lf0; ok = (jj >= 0) & (jj < len(la))
                    if ok.any():
                        lj = jj[ok]; lead_lon[ok] = c * (la[lj, 1] - xo) + s * (la[lj, 2] - yo)
                # strata: a_max over horizon (canonical long accel approx via |xAcc rotated|); lane change
                amax = float(np.max(np.abs(c * a[ti:ti + HOR, 6] + s * a[ti:ti + HOR, 7])))
                kin = 0 if amax < cruise else (1 if amax < hard else 2)
                if has_lc_col:                                  # exiD: explicit laneChange column
                    lcv = a[ti:ti + HOR + 1, 8]
                    is_lc = bool(np.any(np.abs(np.nan_to_num(lcv)) > 0.5))
                elif has_lane_col:                              # highD: detect laneId change
                    is_lc = len(set(a[ti:ti + HOR + 1, 9].astype(int))) > 1
                else:
                    is_lc = False
                if social_grid:                                  # CS-LSTM GLON x GLAT neighbour grid
                    fr_t = int(a[ti, 0]); grid = {}
                    aob = a[ti + obs_idx]; odx = aob[:, 1] - xo; ody = aob[:, 2] - yo
                    THIST.append(np.stack([-s * odx + c * ody, c * odx + s * ody], 1).astype(np.float32))
                    for (nid, nx, ny) in frame_pos[fr_t]:
                        if nid == tid: continue
                        rlon = c * (nx - xo) + s * (ny - yo); rlat = -s * (nx - xo) + c * (ny - yo)
                        li = int(round(rlon / cell)) + glon // 2; wj = int(round(rlat / lane_w)) + glat // 2
                        if 0 <= li < glon and 0 <= wj < glat and nid in A:
                            nf0, na, _ = A[nid]; nidx = fr_t + obs_idx - nf0
                            if nidx[0] >= 0 and nidx[-1] < len(na):
                                ndx = na[nidx, 1] - xo; ndy = na[nidx, 2] - yo
                                grid[wj * glon + li] = np.stack([-s * ndx + c * ndy, c * ndx + s * ndy], 1).astype(np.float32)
                    NBR.append(grid)
                VT.append([vlon, vlat]); AT.append([alon, alat]); GT.append(gt.astype(np.float32))
                LEAD.append(lead_lon.astype(np.float32)); KIN.append(kin); LC.append(int(is_lc))
                REC.append(rec); LOC.append(loc); VID.append(vid_base + tid)
        vid_base += max(A) + 1 if A else 0
        if len(VT) >= cap and nrec == 0: break
    N = len(VT)
    if N > cap:
        keep = random.sample(range(N), cap)
        sel = lambda L: [L[i] for i in keep]
        VT, AT, GT, LEAD, KIN, LC, REC, LOC, VID = map(sel, (VT, AT, GT, LEAD, KIN, LC, REC, LOC, VID))
        if social_grid: NBR = sel(NBR); THIST = sel(THIST)
    low = data_dir.lower()
    name = next((d for d in ("highd", "exid", "ind", "round", "unid") if d in low), None)
    name = {"highd": "highD", "exid": "exiD", "ind": "inD", "round": "rounD", "unid": "uniD"}.get(name) \
        or ("exiD" if "xCenter" in open(files[0]).readline() else "levelX")
    return WindowSet(np.array(VT, np.float32), np.array(AT, np.float32), np.stack(GT),
                     np.stack(LEAD), np.array(KIN), np.array(LC), np.array(REC), np.array(LOC),
                     np.array(VID), fps, NFUT, PRED_T, name=name,
                     NBR=(NBR if social_grid else None), THIST=(THIST if social_grid else None),
                     grid=((glon, glat) if social_grid else None), NHIST=NHIST)


def load_ngsim(csv_path, fps=10.0, obs_s=3.0, pred_s=5.0, stride_s=1.0, ds=2,
               cruise=0.5, hard=1.5, cap=120000, seed=3, smooth_w=7,
               keep=("us-101", "i-80")):
    """Stream the single NGSIM CSV -> WindowSet (feet->m, comma-stripped, MA-smoothed). NGSIM is a
    straight freeway so the road frame is the canonical frame (long=Local_Y, lat=Local_X). Use the
    'vehicle' split for NGSIM (no recording structure). Columns are detected from the header."""
    random.seed(seed)
    HIST = int(round(obs_s * fps)); HOR = int(round(pred_s * fps)); STRIDE = int(round(stride_s * fps))
    obs_idx = np.arange(-HIST, 1, ds); fut_idx = np.arange(ds, HOR + 1, ds)
    NFUT = len(fut_idx); PRED_T = fut_idx / fps; FT = 0.3048
    g = lambda s: float(s.replace(",", "")) if s not in ("", "NA") else math.nan
    def find(cands, header):
        low = [h.lower() for h in header]
        for c in cands:
            for i, h in enumerate(low):
                if c in h: return i
        return None
    veh = collections.defaultdict(list); loc_ids = {}
    with open(csv_path, newline="") as f:
        rd = csv.reader(f); header = next(rd)
        C = {k: find(v, header) for k, v in dict(
            vid=["vehicle_id"], fr=["frame_id", "frame"], x=["local_x"], y=["local_y"],
            vel=["v_vel", "velocity"], lane=["lane_id", "lane"], loc=["location"],
            prec=["preceding"]).items()}
        if None in (C["vid"], C["fr"], C["x"], C["y"]):
            raise ValueError(f"NGSIM header missing required columns; got {header[:8]}...")
        for row in rd:
            loc = row[C["loc"]].lower().strip() if C["loc"] is not None else "ngsim"
            if keep and loc not in keep: continue
            key = (loc, row[C["vid"]])
            prec = row[C["prec"]] if C["prec"] is not None else "0"
            veh[key].append((int(g(row[C["fr"]])), g(row[C["x"]]) * FT, g(row[C["y"]]) * FT,
                             int(g(row[C["lane"]])) if C["lane"] is not None else 0, prec))
    def smooth(a, w=smooth_w):
        if len(a) < w: return a
        return np.convolve(a, np.ones(w) / w, mode="same")
    # build smoothed per-vehicle segments + frame index (for nearest-ahead leader)
    VEH = {}; frame_pos = collections.defaultdict(list); loc_idx = {}; gid = 0
    for (loc, vid), rows in veh.items():
        rows.sort(); fr = np.array([r[0] for r in rows])
        for seg in np.split(np.arange(len(rows)), np.where(np.diff(fr) > 15)[0] + 1):
            if len(seg) < HIST + HOR + 1: continue
            a = [rows[i] for i in seg]
            lon = smooth(np.array([r[2] for r in a])); lat = smooth(np.array([r[1] for r in a]))
            vlon = np.clip(np.gradient(lon) * fps, -45, 45); vlat = np.clip(np.gradient(lat) * fps, -8, 8)
            alon = np.clip(np.gradient(vlon) * fps, -8, 8); alat = np.clip(np.gradient(vlat) * fps, -8, 8)
            frames = np.array([r[0] for r in a]); lane = np.array([r[3] for r in a])
            gid += 1; loc_idx.setdefault(loc, len(loc_idx))
            VEH[gid] = dict(f0=int(frames[0]), lon=lon, lat=lat, vlon=vlon, vlat=vlat, alon=alon,
                            alat=alat, lane=lane, frames=frames, loc=loc)
            for i, fr_ in enumerate(frames): frame_pos[(loc, fr_)].append((gid, lon[i], lat[i], lane[i]))
    VT, AT, GT, LEAD, KIN, LC, REC, LOC, VID = [], [], [], [], [], [], [], [], []
    for gid, v in VEH.items():
        n = len(v["lon"])
        for ti in range(HIST, n - HOR, STRIDE):
            lo0 = v["lon"][ti]; la0 = v["lat"][ti]; fr_t = int(v["frames"][ti]); L = v["lane"][ti]
            fi = ti + fut_idx
            gt = np.stack([v["lat"][fi] - la0, v["lon"][fi] - lo0], 1).astype(np.float32)
            # leader: nearest same-lane ahead at t0; project its future lon
            best = None; bgap = 1e9
            for (nid, nlon, nlat, nl) in frame_pos[(v["loc"], fr_t)]:
                if nid == gid or nl != L: continue
                gp = nlon - lo0
                if 0 < gp < bgap: bgap = gp; best = nid
            lead_lon = np.full(NFUT, np.nan)
            if best is not None:
                bf0 = VEH[best]["f0"]; jj = fr_t + fut_idx - bf0; ok = (jj >= 0) & (jj < len(VEH[best]["lon"]))
                if ok.any(): lead_lon[ok] = VEH[best]["lon"][jj[ok]] - lo0
            amax = float(np.abs(v["alon"][ti:ti + HOR]).max())
            kin = 0 if amax < cruise else (1 if amax < hard else 2)
            is_lc = len(set(v["lane"][ti:ti + HOR + 1].tolist())) > 1
            VT.append([v["vlon"][ti], v["vlat"][ti]]); AT.append([v["alon"][ti], v["alat"][ti]])
            GT.append(gt); LEAD.append(lead_lon.astype(np.float32)); KIN.append(kin); LC.append(int(is_lc))
            REC.append(loc_idx[v["loc"]]); LOC.append(loc_idx[v["loc"]]); VID.append(gid)
    N = len(VT)
    if N > cap:
        keepi = random.sample(range(N), cap); sel = lambda Q: [Q[i] for i in keepi]
        VT, AT, GT, LEAD, KIN, LC, REC, LOC, VID = map(sel, (VT, AT, GT, LEAD, KIN, LC, REC, LOC, VID))
    return WindowSet(np.array(VT, np.float32), np.array(AT, np.float32), np.stack(GT), np.stack(LEAD),
                     np.array(KIN), np.array(LC), np.array(REC), np.array(LOC), np.array(VID),
                     fps, NFUT, PRED_T, name="NGSIM")
