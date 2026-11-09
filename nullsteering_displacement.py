#!/usr/bin/env python3
"""
nullsteer_displacement.py
Micro-displacement heatmap of a wall, AZIMUTH ONLY (elevation fixed at 0 deg).

For every wall cell (range bin k, look angle theta):
  1) build beam weights that keep theta with gain 1 and put NULLS on other directions
     (strong still echoes, other movers, and fixed angles outside the +-30 deg zone)  -> LCMV
  2) beam output y(t) = w^H x(t)
  3) phase relative to the cell's still echo -> displacement d(t) = lambda * dphi / (4 pi)
  4) RMS displacement (micrometres) -> heatmap

Uses the loader and radar settings from vibration_azimuth.py (keep both files in the same folder).
Only the 8 azimuth channels (TX0 + TX2) are used; the raised TX1 is dropped, so elevation is
not steered: the look direction is always at 0 deg elevation (radar height).

Run:
  python3 nullsteer_displacement.py --folder OpenRadar/data --dist 1.1                  # slow movement
  python3 nullsteer_displacement.py --folder OpenRadar/data --dist 1.1 --signal vib --band 100 300
  python3 nullsteer_displacement.py --folder OpenRadar/data --dist 1.1 --compare        # LCMV vs Bartlett vs MVDR
"""
import argparse
import glob
import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
import vibration_azimuth as va                                  # loader + config (same folder)
from vibration_azimuth import dsp                               # OpenRadar dsp

LAM, DR = va.LAM, va.DR
LOOK_DEG = np.arange(-30.0, 30.0001, 1.0)                       # look directions (azimuth, deg)
FIXED_NULLS_DEG = [-64.0, -48.6, 48.6, 64.0]                    # outside the working zone
MIN_SEP_SIN = 0.25                                              # ~1 beam width: never null closer than this
MAX_NULLS = 4                                                   # 8 channels: keep at most 4 nulls
SEARCH_DEG = np.arange(-80.0, 80.0001, 0.5)                     # grid used to find echoes to null


# ---------------------------------------------------------------- beam weights
def steer(deg):
    return va.steer(deg)                                        # [n, 8], elevation = 0


def find_peaks_deg(spec, n_max, floor_db):
    """Local maxima of a spectrum on SEARCH_DEG, strongest first, within floor_db of the top."""
    p = spec / spec.max()
    idx = [i for i in range(1, len(p) - 1) if p[i] >= p[i - 1] and p[i] >= p[i + 1]
           and 10 * np.log10(p[i] + 1e-30) > -floor_db]
    idx.sort(key=lambda i: -p[i])
    return [SEARCH_DEG[i] for i in idx[:n_max]]


def null_candidates(static_vec, R_dyn, n_static=2, n_moving=1):
    """Directions worth nulling in one range bin, in priority order:
    other movers first, then strong still echoes, then the fixed out-of-zone angles."""
    A = steer(SEARCH_DEG) / np.sqrt(8)
    stat_spec = np.abs(A.conj() @ static_vec) ** 2
    dyn_spec = np.real(np.einsum("ai,ij,aj->a", A.conj(), R_dyn, A))
    cands = find_peaks_deg(dyn_spec, n_moving, 10) + find_peaks_deg(stat_spec, n_static, 15)
    return cands + FIXED_NULLS_DEG


def lcmv_weights(look_deg, cands):
    """Gain 1 toward look_deg, gain 0 toward up to MAX_NULLS candidate directions that are at
    least one beam width away. Returns W [n_look, 8] and the nulls used for each look."""
    W, used = [], []
    s_c = np.sin(np.radians(cands)) if len(cands) else np.array([])
    for th in look_deg:
        s0 = np.sin(np.radians(th))
        nulls = []
        for c, sc in zip(cands, s_c):
            if abs(sc - s0) >= MIN_SEP_SIN and all(abs(np.sin(np.radians(n)) - sc) > 0.05 for n in nulls):
                nulls.append(c)
            if len(nulls) == MAX_NULLS:
                break
        C = steer([th] + nulls).T                                # [8, 1 + n_nulls]
        f = np.zeros(C.shape[1])
        f[0] = 1.0
        w = C @ np.linalg.solve(C.conj().T @ C, f)
        W.append(w)
        used.append(nulls)
    return np.array(W), used


def bartlett_weights(look_deg):
    return steer(look_deg) / 8.0                                 # gain 1 toward look, no nulls


def mvdr_weights(look_deg, R_full, loading=0.01):
    """Adaptive: gain 1 toward look, everything else pushed down automatically.
    Uses the FULL covariance (still echoes included) so strong still echoes are suppressed too."""
    Rl = R_full + loading * np.real(np.trace(R_full)) / 8 * np.eye(8)
    A = steer(look_deg)
    Ri_A = np.linalg.solve(Rl, A.T)                              # [8, n]
    return (Ri_A / np.einsum("ai,ia->a", A.conj(), Ri_A)).T


# ---------------------------------------------------------------- displacement
def displacement(y, signal, band):
    """y: beam outputs. move: [frames, n_look]  vib: [frames, loops, n_look].
    Returns RMS displacement per look (m), still-echo amplitude, d(t), and moving energy per look.
    Moving energy |y - mean y|^2 is used to FIND the moving cell; the phase gives HOW MUCH it moves.
    (A cell next to the target has a weak still echo, so its phase swing looks too big:
    displacement alone must not be used to pick the location.)"""
    if signal == "move":
        ref = y.mean(axis=0)
        phi = np.unwrap(np.angle(y * ref.conj()), axis=0)
        d = LAM * phi / (4 * np.pi)
        d -= d.mean(axis=0)
        if band is not None:
            freqs = np.fft.rfftfreq(d.shape[0], va.FRAME_PERIOD)
            D = np.fft.rfft(d, axis=0)
            D[(freqs < band[0]) | (freqs > band[1])] = 0
            d = np.fft.irfft(D, n=d.shape[0], axis=0)
        e = np.mean(np.abs(y - ref) ** 2, axis=0)
        return np.sqrt((d ** 2).mean(axis=0)), np.abs(ref), d, e
    # vib: phase inside each frame (182 loops at the slow-time rate), per-frame mean removed
    ref = y.mean(axis=(0, 1))
    phi = np.unwrap(np.angle(y * ref.conj()), axis=1)
    d = LAM * phi / (4 * np.pi)
    d -= d.mean(axis=1, keepdims=True)
    if band is not None:
        freqs = np.fft.rfftfreq(d.shape[1], 1.0 / va.FS_SLOW)
        win = np.hanning(d.shape[1])
        D = np.fft.rfft(d * win[None, :, None], axis=1)
        D[:, (freqs < band[0]) | (freqs > band[1])] = 0
        d = np.fft.irfft(D, n=d.shape[1], axis=1) / np.sqrt((win ** 2).mean())   # undo window loss
    rms = np.sqrt((d ** 2).mean(axis=(0, 1)))
    e = np.mean(np.abs(y - y.mean(axis=1, keepdims=True)) ** 2, axis=(0, 1))
    return rms, np.abs(ref), d.reshape(-1, d.shape[-1]), e


def process_file(path, args, method):
    r_lo = args.rmin if args.rmin is not None else (args.dist - 0.4 if args.dist else 0.4)
    r_hi = args.rmax if args.rmax is not None else (args.dist + 0.5 if args.dist else 3.0)
    k_lo, k_hi = max(int(r_lo / DR), 1), int(np.ceil(r_hi / DR))
    X = va.load_cube(path, k_hi * DR + 0.05)                     # [frames, loops, 8, bins]
    stat0 = (np.abs(X.sum(axis=2)) ** 2).mean(axis=(0, 1))
    D_wall = args.dist if args.dist else (k_lo + int(np.argmax(stat0[k_lo:k_hi]))) * DR

    n_look = len(LOOK_DEG)
    rms = np.full((k_hi - k_lo, n_look), np.nan)
    amp = np.zeros_like(rms)
    energy = np.zeros_like(rms)
    series = {}
    nulls_log = {}
    for j, k in enumerate(range(k_lo, k_hi)):
        xk = X[..., k]                                           # [frames, loops, 8]
        static_vec = xk.mean(axis=(0, 1))
        if args.signal == "move":
            xt = xk.mean(axis=1)                                 # one sample per frame
        else:
            xt = xk
        flat = xt.reshape(-1, 8)
        dyn = flat - static_vec
        R_dyn = dyn.T @ dyn.conj() / dyn.shape[0]               # E[x x^H] of the moving part
        if method == "lcmv":
            W, used = lcmv_weights(LOOK_DEG, null_candidates(static_vec, R_dyn))
            nulls_log[k] = used
        elif method == "mvdr":
            R_full = flat.T @ flat.conj() / flat.shape[0]       # E[x x^H], still echoes included
            W = mvdr_weights(LOOK_DEG, R_full)
        else:
            W = bartlett_weights(LOOK_DEG)
        y = xt @ W.conj().T                                      # beam outputs, [..., n_look]
        r, a, d, e = displacement(y, args.signal, args.band)
        rms[j], amp[j], energy[j] = r, a, e
        series[k] = d
    # show displacement only where there is a usable still echo AND real movement
    amp_db = 20 * np.log10(amp / amp.max() + 1e-12)
    e_db = 10 * np.log10(energy / energy.max() + 1e-30)
    keep = (amp_db > args.min_db) & (e_db > -args.energy_db)
    rms_masked = np.where(keep, rms, np.nan)
    return dict(k_lo=k_lo, k_hi=k_hi, rms=rms_masked, amp_db=amp_db, e_db=e_db, series=series,
                D=D_wall, nulls=nulls_log)


# ---------------------------------------------------------------- plots
def top_view(ax, res, title, vmax):
    edges_r = np.arange(res["k_lo"], res["k_hi"] + 1) * DR
    th = np.radians(LOOK_DEG)
    dth = th[1] - th[0]
    edges_t = np.concatenate([th - dth / 2, [th[-1] + dth / 2]])
    RR, TT = np.meshgrid(edges_r, edges_t, indexing="ij")
    um = res["rms"] * 1e6
    ax.set_facecolor("0.85")
    pm = ax.pcolormesh(RR * np.sin(TT), RR * np.cos(TT), um, vmin=0, vmax=vmax, cmap="inferno", shading="flat")
    D = res["D"]
    ax.axhline(D, color="cyan", ls="--", lw=0.8)
    if True:
        j, i = np.unravel_index(np.argmax(res["e_db"]), um.shape)
        R, t = (res["k_lo"] + j + 0.5) * DR, np.radians(LOOK_DEG[i])
        ax.plot(R * np.sin(t), R * np.cos(t), "c+", ms=14, mew=2)
    ax.set_aspect("equal")
    ax.set_ylim(max(D - 0.6, edges_r[0] * np.cos(th[-1])), min(D + 0.4, edges_r[-1]))
    ax.set_xlim(-D * np.tan(th[-1]) - 0.1, D * np.tan(th[-1]) + 0.1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Across the wall, x (m)")
    ax.set_ylabel("Distance from radar (m)")
    return pm


def peak_info(res):
    """Location = cell with the most moving energy; displacement = RMS from the phase there."""
    um = res["rms"] * 1e6
    j, i = np.unravel_index(np.argmax(res["e_db"]), um.shape)
    k = res["k_lo"] + j
    R, th = (k + 0.5) * DR, LOOK_DEG[i]
    return k, i, R, th, um[j, i]


def main():
    ap = argparse.ArgumentParser(description="Null-steered micro-displacement heatmap (azimuth only)")
    ap.add_argument("--folder")
    ap.add_argument("--files", nargs="+")
    ap.add_argument("--dist", type=float, default=None, help="radar-to-wall distance (m), optional")
    ap.add_argument("--rmin", type=float)
    ap.add_argument("--rmax", type=float)
    ap.add_argument("--signal", choices=["move", "vib"], default="move",
                    help="move: one sample per frame (slow), vib: samples inside each frame (fast)")
    ap.add_argument("--band", type=float, nargs=2, default=None, help="keep only this band (Hz)")
    ap.add_argument("--method", choices=["lcmv", "bartlett", "mvdr"], default="lcmv",
                    help="lcmv = beam + chosen nulls (default), bartlett = no nulls, mvdr = automatic")
    ap.add_argument("--compare", action="store_true", help="run lcmv, bartlett and mvdr side by side")
    ap.add_argument("--min-db", type=float, default=-20.0,
                    help="hide cells whose still echo is weaker than this vs the strongest (dB)")
    ap.add_argument("--energy-db", type=float, default=15.0,
                    help="hide cells whose movement is more than this below the strongest mover (dB)")
    ap.add_argument("--out", default="results_disp")
    args = ap.parse_args()

    if not args.files:
        if not args.folder:
            ap.error("give --folder or --files")
        args.files = sorted(glob.glob(os.path.join(args.folder, "*.npy")) +
                            glob.glob(os.path.join(args.folder, "*.bin")))
    os.makedirs(args.out, exist_ok=True)
    methods = ["lcmv", "bartlett", "mvdr"] if args.compare else [args.method]
    labs = [os.path.splitext(os.path.basename(f))[0] for f in args.files]
    print(f"Wavelength {LAM * 1e3:.2f} mm (1 deg phase = {LAM / (4 * np.pi) * np.pi / 180 * 1e6:.1f} um) | "
          f"range bin {DR * 100:.2f} cm | signal: {args.signal} | methods: {', '.join(methods)}")

    rows = []
    results = {}
    for path, lab in zip(args.files, labs):
        for m in methods:
            res = process_file(path, args, m)
            results[(lab, m)] = res
            k, i, R, th, um = peak_info(res)
            x = R * np.sin(np.radians(th))
            rows.append((lab, m, R, th, x, um))
            extra = ""
            if m == "lcmv" and res["nulls"].get(k):
                extra = " | nulls at " + ", ".join(f"{n:+.0f}°" for n in res["nulls"][k][i])
            print(f"{lab:12s} {m:8s} moving cell at {th:+5.1f}°, {R:.2f} m -> displacement {um:7.1f} um RMS"
                  f" | x = {x:+.2f} m{extra}")

    # 1) heatmaps: rows = files, columns = methods
    nf, nm = len(labs), len(methods)
    fig, axes = plt.subplots(nf, nm, figsize=(4.6 * nm, 3.9 * nf), squeeze=False)
    for a, lab in enumerate(labs):
        vmax = np.nanmax([np.nanmax(results[(lab, m)]["rms"]) for m in methods]) * 1e6
        for b, m in enumerate(methods):
            pm = top_view(axes[a, b], results[(lab, m)], f"{lab} - {m.upper()}", vmax)
        fig.colorbar(pm, ax=axes[a, :].tolist(), label="RMS displacement (µm)", shrink=0.85)
    fig.suptitle("Displacement of every wall cell (grey = weak echo or no movement, dashed = wall, + = moving cell)")
    fig.savefig(os.path.join(args.out, "displacement_heatmaps.png"), dpi=130, bbox_inches="tight")

    # 2) displacement signal of the largest cell (first method)
    fig, axes = plt.subplots(nf, 2, figsize=(11, 2.4 * nf), squeeze=False)
    fs = 1.0 / va.FRAME_PERIOD if args.signal == "move" else va.FS_SLOW
    for a, lab in enumerate(labs):
        res = results[(lab, methods[0])]
        k, i, R, th, um = peak_info(res)
        d = res["series"][k][:, i] * 1e6
        t = np.arange(d.size) / fs
        n_show = d.size if args.signal == "move" else min(d.size, 3 * va.N_LOOPS)
        axes[a, 0].plot(t[:n_show], d[:n_show], lw=0.8)
        axes[a, 0].set_ylabel("µm")
        axes[a, 0].set_title(f"{lab}: displacement at {th:+.0f}°, {R:.2f} m ({methods[0].upper()})", fontsize=9)
        f = np.fft.rfftfreq(d.size, 1 / fs)
        spec = np.abs(np.fft.rfft(d * np.hanning(d.size))) * 2 / np.hanning(d.size).sum()
        axes[a, 1].plot(f[1:], spec[1:], lw=0.8)
        axes[a, 1].set_ylabel("µm")
        axes[a, 1].set_title("Spectrum", fontsize=9)
    axes[-1, 0].set_xlabel("Time (s)" + ("" if args.signal == "move" else ", first 3 frames joined"))
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "displacement_peak_signal.png"), dpi=130)

    with open(os.path.join(args.out, "displacement_peaks.csv"), "w") as fh:
        fh.write("file,method,range_m,angle_deg,x_m,rms_um\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.3f},{r[3]:.1f},{r[4]:.3f},{r[5]:.2f}\n")
    print(f"Saved displacement_heatmaps.png, displacement_peak_signal.png, displacement_peaks.csv in {args.out}/")


if __name__ == "__main__":
    main()