#!/usr/bin/env python3
"""
phone_azimuth_locate.py
Locate a MOVING object (phone) in front of a wall in AZIMUTH only, from IWR1843 + DCA1000 data
(.npy or raw .bin files).
Default --signal move: slow movement (swinging, sliding), tracked frame to frame.
Option  --signal vib : fast vibration (e.g. phone motor), filtered inside each frame.
Uses OpenRadar (mmwave) for reading frames, range FFT, clutter removal and forward-backward averaging.

Two steps:
  1) Range check: find the wall / moving-object range bins
       python phone_azimuth_locate.py range  --folder data4 --dist 1.5
  2) Locate: Bartlett, fixed null steering and MVDR over -30..+30 deg
       python phone_azimuth_locate.py locate --folder data4 --dist 1.5
     (--truth x1 x2 ... is optional; only if you measured the phone positions)
  3) Heatmap: where the movement is for EVERY file, worked out from the data only
       python vibration_azimuth.py heatmap --folder data4            (wall distance found automatically)
  (--files a.bin b.bin ... can be used instead of --folder to set the order yourself)

Outputs go to --out (default: results/).
"""
import argparse
import glob
import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=SyntaxWarning)
try:
    import mmwave  # noqa: F401
except ImportError:                                    # not pip-installed: use the OpenRadar folder next to this script
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "OpenRadar"))
from mmwave.dataloader import DCA1000                  # noqa: E402
from mmwave import dsp                                 # noqa: E402
from mmwave.dsp.utils import Window                    # noqa: E402

# ===================== CONFIG — matches your mmWave Studio Lua script =====================
N_ADC = 256            # ADC_SAMPLES
N_RX = 4               # 4 RX
TX_ORDER = [0, 2, 1]   # chirp 0 = TX0, chirp 1 = TX2, chirp 2 = TX1 (ptx[1..3] in the Lua script)
N_LOOPS = 182          # CHIRP_LOOPS (read from the .npy shape if it differs)
FS_ADC = 4.4e6         # SAMPLE_RATE = 4400 ksps
SLOPE = 60.012e12      # FREQ_SLOPE = 60.012 MHz/us
F_START = 77e9         # START_FREQ = 77 GHz
CHIRP_CYCLE = 72e-6    # IDLE_TIME 7 us + RAMP_END_TIME 65 us
FRAME_PERIOD = 0.100   # PERIODICITY = 100 ms  -> 10 frames per second
ANGLE_SIGN = +1        # flip to -1 if left and right come out swapped
# ====================================================================================

C = 3e8
N_TX = len(TX_ORDER)
BW = SLOPE * N_ADC / FS_ADC
LAM = C / (F_START + BW / 2)
DR = C * FS_ADC / (2 * SLOPE * N_ADC)                  # range bin size (complex ADC)
LOOP_PERIOD = N_TX * CHIRP_CYCLE                       # time between two chirps of the same TX
FS_SLOW = 1.0 / LOOP_PERIOD                            # slow-time rate inside a frame (Hz)
SCAN_DEG = np.arange(-30.0, 30.0001, 0.25)             # beamforming scan: only the working zone
NULL_DEG = [-64.0, -48.6, 48.6, 64.0]                  # fixed nulls, outside the zone
POS = np.arange(8)                                     # azimuth channel positions in lambda/2 units


# ---------------------------------------------------------------- loading
def _frames_from_bin(path):
    """Raw DCA1000 .bin -> generator of frames [loops, n_tx, rx, adc] complex."""
    raw = np.fromfile(path, dtype=np.int16)
    per_frame = N_LOOPS * N_TX * N_RX * N_ADC * 2
    n_frames = raw.size // per_frame
    if n_frames == 0:
        raise ValueError(f"{path}: smaller than one frame, check the config block")
    if raw.size % per_frame:
        print(f"  note: {path} has {raw.size % per_frame} extra int16 values at the end (ignored)")
    for f in range(n_frames):
        fr = raw[f * per_frame:(f + 1) * per_frame].astype(np.float32)
        adc = DCA1000.organize(fr, N_LOOPS * N_TX, N_RX, N_ADC)          # [chirps, rx, samples]
        yield adc.reshape(N_LOOPS, N_TX, N_RX, N_ADC)


def _frames_from_raw_npy(path):
    """.npy of raw DCA1000 16-bit words, e.g. (frames, words_per_frame) as saved by OpenRadar's
    DCA1000.read(). OpenRadar stores them as uint16, but the ADC values are SIGNED 16-bit,
    so they are re-read as int16 before unpacking (otherwise every negative sample is wrong)."""
    a = np.load(path, mmap_mode="r")
    per_loop = N_TX * N_RX * N_ADC * 2
    if a.ndim == 2:                                                         # one row per frame
        if a.shape[1] % per_loop:
            raise ValueError(f"{path}: {a.shape[1]} values per frame is not a multiple of "
                             f"{per_loop} (TX x RX x samples x 2); check N_ADC / TX_ORDER")
        loops = a.shape[1] // per_loop
        if loops != N_LOOPS:
            print(f"  note: data has {loops} loops per frame, using that instead of N_LOOPS={N_LOOPS}")
        rows = (a[f] for f in range(a.shape[0]))
    else:                                                                   # one long stream
        loops = N_LOOPS
        flat = a.reshape(-1)
        n_f = flat.size // (per_loop * loops)
        rows = (flat[f * per_loop * loops:(f + 1) * per_loop * loops] for f in range(n_f))
    for r in rows:
        r = np.ascontiguousarray(r)
        r = (r.view(np.int16) if r.dtype.itemsize == 2 else r.astype(np.int16)).astype(np.float32)
        adc = DCA1000.organize(r, loops * N_TX, N_RX, N_ADC)                 # [chirps, rx, samples]
        yield adc.reshape(loops, N_TX, N_RX, N_ADC)


def _npy_as_frames(path):
    """.npy -> array viewed as [frames, loops, n_tx, rx, adc] (complex).
    Accepts the usual layouts: raw int16 (like .bin), (chirps, rx, adc), (frames, chirps, rx, adc),
    (frames, loops, tx, rx, adc), with rx/adc axes found by their sizes, and real arrays with a
    last axis of 2 (I, Q)."""
    a = np.load(path, mmap_mode="r")
    print(f"  {os.path.basename(path)}: shape {a.shape}, dtype {a.dtype}")
    if np.issubdtype(a.dtype, np.integer):                                  # raw DCA1000 words
        return None
    if not np.iscomplexobj(a) and a.shape[-1] == 2:                         # (..., 2) = I, Q
        a = np.asarray(a[..., 0]) + 1j * np.asarray(a[..., 1])
    if not np.iscomplexobj(a):
        raise ValueError(f"{path}: real-valued data with shape {a.shape} is not understood")
    shp = list(a.shape)
    adc_ax = max(i for i, n in enumerate(shp) if n == N_ADC) if N_ADC in shp else None
    if adc_ax is None:
        raise ValueError(f"{path}: no axis of size N_ADC={N_ADC}; set N_ADC in the config block")
    rx_cand = [i for i, n in enumerate(shp) if n == N_RX and i != adc_ax]
    if not rx_cand:
        raise ValueError(f"{path}: no axis of size N_RX={N_RX}")
    rx_ax = rx_cand[-1]
    rest = [i for i in range(a.ndim) if i not in (adc_ax, rx_ax)]
    a = np.transpose(a, rest + [rx_ax, adc_ax])                             # [..., rx, adc]
    lead = a.shape[:-2]
    if len(lead) == 1:                                                      # (chirps, rx, adc)
        per = N_LOOPS * N_TX
        n_f = lead[0] // per
        a = a[:n_f * per].reshape(n_f, N_LOOPS, N_TX, N_RX, N_ADC)
    elif len(lead) == 2:                                                    # (frames, chirps, rx, adc)
        chirps = lead[1]
        if chirps % N_TX:
            raise ValueError(f"{path}: {chirps} chirps per frame is not a multiple of {N_TX} TX")
        if chirps != N_LOOPS * N_TX:
            print(f"  note: data has {chirps // N_TX} loops per frame, using that instead of N_LOOPS={N_LOOPS}")
        a = a.reshape(lead[0], chirps // N_TX, N_TX, N_RX, N_ADC)
    elif len(lead) == 3:                                                    # (frames, loops, tx, rx, adc)
        if lead[2] != N_TX:
            raise ValueError(f"{path}: TX axis has {lead[2]} entries but TX_ORDER has {N_TX}")
    else:
        raise ValueError(f"{path}: shape {a.shape} is not understood")
    return a


def load_cube(path, max_range):
    """Read a .bin or .npy file -> range FFT -> 8 azimuth channels.
    Returns X [frames, loops, 8, n_bins] complex64 (bins 0 .. max_range)."""
    if 0 not in TX_ORDER or 2 not in TX_ORDER:
        raise ValueError("TX_ORDER must contain TX0 and TX2 (the azimuth transmitters)")
    i0, i2 = TX_ORDER.index(0), TX_ORDER.index(2)
    n_bins = min(int(max_range / DR) + 1, N_ADC)
    frames = None
    if path.lower().endswith(".npy"):
        arr = _npy_as_frames(path)
        if arr is None:
            frames = _frames_from_raw_npy(path)
        else:
            frames = (arr[f] for f in range(arr.shape[0]))
    if frames is None:
        frames = _frames_from_bin(path)
    out = []
    for adc in frames:
        adc = np.asarray(adc, dtype=np.complex64)
        adc = np.concatenate([adc[:, i0], adc[:, i2]], axis=1)             # TX0 rx0-3, TX2 rx0-3
        adc = adc - adc.mean(axis=-1, keepdims=True)
        rf = dsp.range_processing(adc, window_type_1d=Window.HANNING)      # [loops, 8, N_ADC]
        out.append(rf[..., :n_bins].astype(np.complex64))
    return np.stack(out)


def moving_part_slow(X, band=None):
    """Slow movement (swinging, sliding): one sample per frame.
    1) average the loops inside each frame (movement is ~constant within one frame),
    2) remove the still part = mean over all frames (OpenRadar clutter_removal along frames),
    3) optional: keep only movement frequencies in band (Hz), using FRAME_PERIOD.
    Returns Y [frames, 1, 8, n_bins]."""
    Z = X.mean(axis=1)                                                     # [frames, 8, bins]
    Z = dsp.clutter_removal(Z, axis=0)
    if band is not None:
        freqs = np.fft.fftfreq(Z.shape[0], FRAME_PERIOD)
        keep = (np.abs(freqs) >= band[0]) & (np.abs(freqs) <= band[1])
        if not keep.any():
            raise ValueError("No frame-rate frequencies in --band; check FRAME_PERIOD and --band")
        Z = np.fft.fft(Z * np.hanning(Z.shape[0])[:, None, None], axis=0)[keep]
    return Z[:, None].astype(np.complex64)


def moving_part(X, f_lo, f_hi):
    """Keep only what moves at f_lo..f_hi Hz (the vibration), per frame.
    1) remove the still part (OpenRadar clutter_removal along the loop axis),
    2) slow-time FFT inside each frame (Doppler), keep only bins with f_lo <= |f| <= f_hi,
    3) undo the TDM time shift between TX0 and TX2 chirps for each kept frequency.
    Slow swinging and people moving slowly fall below f_lo and are dropped.
    Returns Y [frames, kept_freqs, 8, n_bins]."""
    n_f, n_l = X.shape[:2]
    if n_l < 16:
        print("  WARNING: fewer than 16 loops per frame, vibration band cannot be separated well")
    freqs = np.fft.fftfreq(n_l, LOOP_PERIOD)
    keep = (np.abs(freqs) >= f_lo) & (np.abs(freqs) <= f_hi)
    if not keep.any():
        raise ValueError(f"No Doppler bins in {f_lo}-{f_hi} Hz (slow-time rate {FS_SLOW:.0f} Hz, "
                         f"resolution {FS_SLOW / n_l:.1f} Hz). Change --band or the config.")
    win = np.hanning(n_l)[:, None, None]
    dt = (TX_ORDER.index(2) - TX_ORDER.index(0)) * CHIRP_CYCLE           # TX2 fires this much after TX0
    comp = np.ones((keep.sum(), 8), complex)
    comp[:, 4:] = np.exp(-2j * np.pi * freqs[keep] * dt)[:, None]
    Y = np.empty((n_f, keep.sum(), 8, X.shape[-1]), np.complex64)
    for f in range(n_f):
        x = dsp.clutter_removal(X[f], axis=0)
        D = np.fft.fft(x * win, axis=0)[keep]
        Y[f] = D * comp[:, :, None]
    return Y


# ---------------------------------------------------------------- beamforming
def steer(deg):
    s = np.sin(np.radians(np.atleast_1d(deg)))
    return np.exp(1j * ANGLE_SIGN * np.pi * np.outer(s, POS))             # [n_angles, 8]


def null_weights(deg_scan, null_deg):
    """Steering vectors projected away from the null directions, scaled to unit length."""
    B = steer(null_deg).T                                                  # [8, n_nulls]
    P = np.eye(8) - B @ np.linalg.pinv(B)
    W = (P @ steer(deg_scan).T).T
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def dyn_covariance(Y, bins):
    """8x8 covariance of the moving part, summed over the chosen range bins."""
    R = np.zeros((8, 8), complex)
    for k in bins:
        x = Y[..., k].reshape(-1, 8).T                                     # [8, snapshots]
        R += x @ x.conj().T / x.shape[1]
    return R


def zone_power(Y, k_lo, k_hi):
    """Moving power per range bin, counted only inside the -30..+30 deg zone
    (null-steered beams), so movers outside the zone do not pick the bin."""
    W = null_weights(SCAN_DEG, NULL_DEG)
    out = np.zeros(Y.shape[-1])
    for k in range(k_lo, min(k_hi, Y.shape[-1])):
        R = dyn_covariance(Y, [k])
        out[k] = np.real(np.einsum("ai,ij,aj->a", W.conj(), R, W)).max()
    return out


def spectra(R, loading=0.01):
    A = steer(SCAN_DEG) / np.sqrt(8)
    bart = np.real(np.einsum("ai,ij,aj->a", A.conj(), R, A))
    W = null_weights(SCAN_DEG, NULL_DEG)
    nul = np.real(np.einsum("ai,ij,aj->a", W.conj(), R, W))
    Rfb = np.asarray(dsp.forward_backward_avg(R))
    Rl = Rfb + loading * np.real(np.trace(Rfb)) / 8 * np.eye(8)
    Ri = np.linalg.inv(Rl)
    mvdr = 1.0 / np.real(np.einsum("ai,ij,aj->a", A.conj(), Ri, A))
    return {"Bartlett": bart, "Null steering": nul, "MVDR": mvdr}


def peak_deg(p):
    """Peak angle with parabolic refinement."""
    i = int(np.argmax(p))
    if 0 < i < len(p) - 1:
        y0, y1, y2 = np.log(p[i - 1:i + 2] + 1e-30)
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return SCAN_DEG[i] + off * (SCAN_DEG[1] - SCAN_DEG[0])
    return SCAN_DEG[i]


# ---------------------------------------------------------------- steps
def window_bins(dist, rmin, rmax):
    lo = rmin if rmin is not None else dist - 0.25
    hi = rmax if rmax is not None else dist + 0.35
    return max(int(np.floor(lo / DR)), 1), int(np.ceil(hi / DR))


def run_range(args):
    max_r = (args.rmax or args.dist + 0.35) + 0.5
    k_lo, k_hi = window_bins(args.dist, args.rmin, args.rmax)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    print(f"Range bin = {DR * 100:.2f} cm | search window {k_lo * DR:.2f}-{k_hi * DR:.2f} m")
    print(f"{'file':30s} {'wall bin (static)':>20s} {'moving bin (in ±30°)':>24s}")
    for path, lab in zip(args.files, labels(args)):
        X = load_cube(path, max_r)
        Xd = get_moving(X, args)
        stat = 10 * np.log10((np.abs(X) ** 2).mean(axis=(0, 1, 2)) + 1e-12)
        dyn = 10 * np.log10((np.abs(Xd) ** 2).mean(axis=(0, 1, 2)) + 1e-12)
        r = np.arange(X.shape[-1]) * DR
        ax1.plot(r, stat, label=lab)
        ax2.plot(r, dyn, label=lab)
        kw = k_lo + int(np.argmax(stat[k_lo:k_hi]))
        kv = k_lo + int(np.argmax(zone_power(Xd, k_lo, k_hi)[k_lo:k_hi]))
        print(f"{lab:30s} {kw:>8d} ({kw * DR:.3f} m)     {kv:>10d} ({kv * DR:.3f} m)")
    for ax in (ax1, ax2):
        ax.axvspan(k_lo * DR, k_hi * DR, color="tab:green", alpha=0.08)
        ax.axvline(args.dist, color="k", ls="--", lw=0.8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    ax1.set_ylabel("Static power (dB)")
    ax1.set_title("Range profile: everything (wall, floor, door)")
    ax2.set_ylabel("Moving power (dB)")
    ax2.set_title("Range profile: moving part only")
    ax2.set_xlabel("Range (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "range_profiles.png"), dpi=150)
    print(f"Saved {args.out}/range_profiles.png")


def run_locate(args):
    k_lo, k_hi = window_bins(args.dist, args.rmin, args.rmax)
    max_r = k_hi * DR + 0.1
    labs = labels(args)
    rows, all_spec = [], []
    for path, lab in zip(args.files, labs):
        X = load_cube(path, max_r)
        Xd = get_moving(X, args)
        if args.bins:
            bins = args.bins
        else:
            kv = k_lo + int(np.argmax(zone_power(Xd, k_lo, k_hi)[k_lo:k_hi]))
            half = 2 if args.signal == "move" else 1                     # movement can spread over bins
            bins = [k for k in range(kv - half, kv + half + 1) if 0 <= k < X.shape[-1]]
        R = dyn_covariance(Xd, bins)
        sp = spectra(R)
        all_spec.append(sp)
        res = {m: peak_deg(p) for m, p in sp.items()}
        rows.append((lab, bins, res))
        print(f"{lab}: bins {bins} | " + " | ".join(
            f"{m} {d:+.1f}° (x={args.dist * np.tan(np.radians(d)):+.2f} m)" for m, d in res.items()))

    # spectra per file
    n = len(args.files)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, sp, (lab, _, res), i in zip(axes, all_spec, rows, range(n)):
        for m, p in sp.items():
            ax.plot(SCAN_DEG, 10 * np.log10(p / p.max() + 1e-12), label=m)
        if args.truth:
            ax.axvline(np.degrees(np.arctan(args.truth[i] / args.dist)), color="k", ls="--", lw=1)
        ax.set_ylim(-25, 1)
        ax.set_ylabel("dB")
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[-1].set_xlabel("Azimuth (deg)   dashed = true phone angle")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "locate_spectra.png"), dpi=150)

    # summary + csv
    with open(os.path.join(args.out, "locate_results.csv"), "w") as fh:
        fh.write("file,bins,method,angle_deg,x_m,true_x_m,error_m\n")
        for i, (lab, bins, res) in enumerate(rows):
            for m, d in res.items():
                x = args.dist * np.tan(np.radians(d))
                tx = args.truth[i] if args.truth else np.nan
                fh.write(f"{lab},{' '.join(map(str, bins))},{m},{d:.2f},{x:.3f},{tx:.3f},{x - tx:.3f}\n")
    methods = list(all_spec[0].keys())
    if args.truth:
        fig, ax = plt.subplots(figsize=(6, 5))
        lim = max(abs(np.array(args.truth))) * 1.3 + 0.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="perfect")
        for m, mk in zip(methods, "osd"):
            est = [args.dist * np.tan(np.radians(r[2][m])) for r in rows]
            err = np.sqrt(np.mean((np.array(est) - np.array(args.truth)) ** 2))
            ax.plot(args.truth, est, mk, ms=7, label=f"{m} (RMSE {err * 100:.1f} cm)")
        ax.set_xlabel("True phone position x (m)")
        ax.set_ylabel("Estimated x (m)")
    else:
        # no ground truth: show where each file's peak is, and how sure each method is
        fig, ax = plt.subplots(figsize=(7, 0.6 * n + 2))
        for j, (m, mk) in enumerate(zip(methods, "osd")):
            xs = [args.dist * np.tan(np.radians(r[2][m])) for r in rows]
            ax.plot(xs, np.arange(n) + (j - 1) * 0.15, mk, ms=7, label=m)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(labs)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Estimated phone position on the wall, x (m)   (- = left, + = right)")
        print("\nHow sure is each result (peak height above the spectrum's median, dB) "
              "and do the methods agree:")
        for (lab, _, res), sp in zip(rows, all_spec):
            xs = {m: args.dist * np.tan(np.radians(d)) for m, d in res.items()}
            spread = max(xs.values()) - min(xs.values())
            conf = " | ".join(f"{m} {10 * np.log10(p.max() / np.median(p)):.1f} dB" for m, p in sp.items())
            print(f"  {lab}: x = {xs['MVDR']:+.2f} m (MVDR) | methods differ by {spread * 100:.1f} cm | {conf}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "locate_summary.png"), dpi=150)
    print(f"Saved results to {args.out}/")


def get_moving(X, args):
    if args.signal == "move":
        return moving_part_slow(X, args.band)
    return moving_part(X, *band(args))


def bin_spectrum(R, method, loading=0.01):
    """Movement power vs SCAN_DEG for one 8x8 covariance."""
    if method == "bartlett":
        A = steer(SCAN_DEG) / np.sqrt(8)
        return np.real(np.einsum("ai,ij,aj->a", A.conj(), R, A))
    if method == "null":
        W = null_weights(SCAN_DEG, NULL_DEG)
        return np.real(np.einsum("ai,ij,aj->a", W.conj(), R, W))
    A = steer(SCAN_DEG) / np.sqrt(8)
    Rfb = np.asarray(dsp.forward_backward_avg(R))
    Ri = np.linalg.inv(Rfb + loading * np.real(np.trace(Rfb)) / 8 * np.eye(8) + 1e-20 * np.eye(8))
    return 1.0 / np.real(np.einsum("ai,ij,aj->a", A.conj(), Ri, A))


def run_heatmap(args):
    """For every file, independently: movement power for every (range bin, angle) cell in front of
    the radar, shown as a top-view heatmap, plus where on the wall the movement is strongest.
    Nothing is taken from file names or file order."""
    r_lo = args.rmin if args.rmin is not None else (args.dist - 0.4 if args.dist else 0.4)
    r_hi = args.rmax if args.rmax is not None else (args.dist + 0.5 if args.dist else 3.0)
    k_lo, k_hi = max(int(r_lo / DR), 1), int(np.ceil(r_hi / DR))
    th = np.radians(SCAN_DEG)
    n = len(args.files)
    labs = labels(args)
    maps, rows = [], []
    for path, lab in zip(args.files, labs):
        X = load_cube(path, k_hi * DR + 0.05)
        Y = get_moving(X, args)
        # wall distance: given, or the strongest still echo straight ahead (0 deg beam)
        stat = (np.abs(X.sum(axis=2)) ** 2).mean(axis=(0, 1))
        k_wall = k_lo + int(np.argmax(stat[k_lo:k_hi]))
        D = args.dist if args.dist else k_wall * DR
        P = np.zeros((k_hi - k_lo, len(SCAN_DEG)))
        for j, k in enumerate(range(k_lo, k_hi)):
            P[j] = bin_spectrum(dyn_covariance(Y, [k]), args.method)
        j, i = np.unravel_index(np.argmax(P), P.shape)
        R_pk, th_pk = (k_lo + j) * DR, SCAN_DEG[i]
        x_pk, y_pk = R_pk * np.sin(np.radians(th_pk)), R_pk * np.cos(np.radians(th_pk))
        contrast = 10 * np.log10(P.max() / np.median(P))
        wall_profile = P.max(axis=0)                                   # strongest bin for each angle
        maps.append((P, wall_profile, D))
        rows.append((lab, R_pk, th_pk, x_pk, y_pk, D, contrast))
        print(f"{lab:14s} most movement at {th_pk:+6.1f}°, range {R_pk:.2f} m  ->  "
              f"x = {x_pk:+.2f} m across, {y_pk:.2f} m out | peak {contrast:.1f} dB above median "
              f"| wall at {D:.2f} m")

    # 1) top-view heatmaps, one per file
    ranges = (np.arange(k_lo, k_hi) + 0.5) * DR
    edges_r = np.arange(k_lo, k_hi + 1) * DR
    dth = np.radians(SCAN_DEG[1] - SCAN_DEG[0])
    edges_t = np.concatenate([th - dth / 2, [th[-1] + dth / 2]])
    RR, TT = np.meshgrid(edges_r, edges_t, indexing="ij")
    cols = min(n, 3)
    rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(5 * cols, 4.2 * rws), squeeze=False)
    for ax, (P, _, D), r in zip(axes.ravel(), maps, rows):
        db = 10 * np.log10(P / P.max() + 1e-12)
        pm = ax.pcolormesh(RR * np.sin(TT), RR * np.cos(TT), db, vmin=-30, vmax=0,
                           cmap="inferno", shading="flat")
        ax.axhline(D, color="cyan", ls="--", lw=0.8)
        ax.plot(r[3], r[4], "c+", ms=14, mew=2)
        ax.set_title(f"{r[0]}: peak x = {r[3]:+.2f} m ({r[2]:+.1f}°)", fontsize=10)
        ax.set_xlabel("Across the wall, x (m)")
        ax.set_ylabel("Distance from radar (m)")
        ax.set_aspect("equal")
        ax.set_ylim(max(D - 0.6, edges_r[0] * np.cos(th[-1])), min(D + 0.4, edges_r[-1]))
        ax.set_xlim(-D * np.tan(th[-1]) - 0.1, D * np.tan(th[-1]) + 0.1)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(pm, ax=axes.ravel().tolist(), label="Movement power (dB, per file)", shrink=0.8)
    fig.suptitle(f"Where the movement is (top view, {args.method.upper()}, dashed = wall)")
    fig.savefig(os.path.join(args.out, "movement_heatmaps.png"), dpi=150, bbox_inches="tight")

    # 2) one strip per file: movement along the wall
    D0 = np.median([m[2] for m in maps])
    xw = D0 * np.tan(th)
    strip = np.array([10 * np.log10(m[1] / m[1].max() + 1e-12) for m in maps])
    fig, ax = plt.subplots(figsize=(9, 0.55 * n + 1.6))
    im = ax.pcolormesh(xw, np.arange(n), strip, vmin=-20, vmax=0, cmap="inferno", shading="nearest")
    for k, r in enumerate(rows):
        ax.plot(D0 * np.tan(np.radians(r[2])), k, "c+", ms=12, mew=2)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(f"Position along the wall, x (m)   (wall at {D0:.2f} m, - = left, + = right)")
    fig.colorbar(im, ax=ax, label="Movement (dB, per file)")
    ax.set_title("Movement along the wall for each file (+ = strongest)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "movement_wall_strips.png"), dpi=150)

    with open(os.path.join(args.out, "movement_peaks.csv"), "w") as fh:
        fh.write("file,range_m,angle_deg,x_across_m,y_out_m,wall_m,peak_db\n")
        for r in rows:
            fh.write(",".join([r[0]] + [f"{v:.3f}" for v in r[1:]]) + "\n")
    print(f"Saved movement_heatmaps.png, movement_wall_strips.png, movement_peaks.csv in {args.out}/")


def band(args):
    lo, hi = args.band if args.band else (100.0, 300.0)
    nyq = 0.95 * FS_SLOW / 2
    return lo, (min(hi, nyq) if hi > 0 else nyq)


def labels(args):
    return args.labels if args.labels else [os.path.splitext(os.path.basename(f))[0] for f in args.files]


def main():
    ap = argparse.ArgumentParser(description="Locate a vibrating phone in azimuth (IWR1843 + DCA1000)")
    ap.add_argument("mode", choices=["range", "locate", "heatmap"])
    ap.add_argument("--folder", help="folder with the .bin files (sorted by name)")
    ap.add_argument("--files", nargs="+", help="the .bin files, in the order you want")
    ap.add_argument("--signal", choices=["move", "vib"], default="move",
                    help="move = slow movement frame to frame (default), vib = fast vibration")
    ap.add_argument("--labels", nargs="+", help="short names for the files")
    ap.add_argument("--dist", type=float, default=None,
                    help="radar-to-wall distance (m); optional for heatmap (found from the data)")
    ap.add_argument("--method", choices=["mvdr", "bartlett", "null"], default="mvdr",
                    help="beamformer for the heatmap (default mvdr)")
    ap.add_argument("--rmin", type=float, help="start of range search window (m)")
    ap.add_argument("--rmax", type=float, help="end of range search window (m)")
    ap.add_argument("--bins", type=int, nargs="+", help="force these range bins in locate mode")
    ap.add_argument("--truth", type=float, nargs="+", help="true phone x (m), same order as files")
    ap.add_argument("--band", type=float, nargs=2, default=None,
                    help="move: optional movement band in Hz, e.g. 0.2 5 (needs FRAME_PERIOD). "
                         "vib: vibration band, default 100 300; high 0 = up to Nyquist")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if not args.files:
        if not args.folder:
            ap.error("give --folder or --files")
        fs = sorted(glob.glob(os.path.join(args.folder, "*.npy")) + glob.glob(os.path.join(args.folder, "*.bin")))
        if any("_Raw_" not in f for f in fs):
            fs = [f for f in fs if "_Raw_" not in f]                       # prefer mmWave Studio's processed files
        if not fs:
            ap.error(f"no .npy or .bin files in {args.folder}")
        args.files = fs
    print("Files (in this order):")
    for f in args.files:
        print("   ", f)
    if args.truth and len(args.truth) != len(args.files):
        ap.error("--truth must have one value per file, in the same order")
    os.makedirs(args.out, exist_ok=True)
    info = (f"movement, one sample per frame" if args.signal == "move"
            else f"vibration {band(args)[0]:.0f}-{band(args)[1]:.0f} Hz inside frames")
    print(f"Wavelength {LAM * 1e3:.2f} mm | range bin {DR * 100:.2f} cm | TX order {TX_ORDER} | {info}")
    if args.mode != "heatmap" and args.dist is None:
        ap.error("--dist is needed for range and locate modes")
    {"range": run_range, "locate": run_locate, "heatmap": run_heatmap}[args.mode](args)


if __name__ == "__main__":
    main()