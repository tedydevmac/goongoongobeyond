"""
features.py — shared feature extraction for rail corrugation classification.

Imported by both train.py and predict.py so train/inference feature logic can
never drift apart.

Physics recap (see Rail_Corrugation_Info_Kit.md):
- Column 1 is a 0/1 toothed-wheel square wave (90 teeth/rev, 0.85 m wheel).
  Counting transitions gives train speed.
- Columns 2-129 are 64 axle boxes x (vibration, shock). Positions 1,3,5,7 of
  every car -> Side I rail; positions 2,4,6,8 -> Side II rail. Sides are
  judged independently.
- Corrugation is a fixed *spatial* wavelength (2-40 cm) defect, so its
  temporal frequency shifts with speed. Order-tracking (freq -> wavelength
  via speed) is the physically stable feature; raw frequency/amplitude is not.
"""

import re
import os
import numpy as np
import pandas as pd
import pywt

FS = 10_000
WHEEL_DIAMETER = 0.85
TEETH = 90
WHEEL_CIRCUM = np.pi * WHEEL_DIAMETER
WAVELEN_MIN, WAVELEN_MAX = 0.02, 0.40

SIDE_I_POSITIONS = (1, 3, 5, 7)
SIDE_II_POSITIONS = (2, 4, 6, 8)
STATIONARY_TRANSITIONS_THRESHOLD = 10  # below this, treat as a near-stationary
                                        # recording (see is_stationary feature
                                        # and audit_files below)


# --------------------------------------------------------------------------- #
# Column parsing / speed decoding
# --------------------------------------------------------------------------- #

def parse_columns(columns):
    """Map each vibration/shock column name to (car, position, side, kind)."""
    meta = []
    pattern = re.compile(r"(Vibration|Shock) of bearing in position (\d+) of car (\d+)")
    for col in columns[1:]:  # skip 'Rotating speed'
        m = pattern.match(col)
        if not m:
            continue
        kind, pos, car = m.group(1), int(m.group(2)), int(m.group(3))
        side = "Side I" if pos in SIDE_I_POSITIONS else "Side II"
        meta.append({"col": col, "car": car, "position": pos, "side": side, "kind": kind})
    return pd.DataFrame(meta)


def estimate_speed(speed_signal, fs=FS):
    """
    Decode train speed (m/s) from the 0/1 toothed-wheel square wave.
    2 transitions per tooth (rising + falling), TEETH teeth per revolution.
    Returns (speed, n_transitions). A low transition count signals an
    unreliable estimate — callers should treat that as "no wavelength"
    rather than dividing by a near-zero number.
    """
    sig = np.asarray(speed_signal).astype(int)
    transitions = int(np.sum(np.abs(np.diff(sig)) > 0))
    duration = len(sig) / fs
    rev_per_sec = transitions / (2 * TEETH) / duration
    speed = rev_per_sec * WHEEL_CIRCUM
    return speed, transitions


# --------------------------------------------------------------------------- #
# Per-channel signal features
# --------------------------------------------------------------------------- #

def _band_mask(freqs, speed, wmin=WAVELEN_MIN, wmax=WAVELEN_MAX):
    if speed <= 0.05:
        return np.zeros_like(freqs, dtype=bool)
    f_lo, f_hi = speed / wmax, speed / wmin
    return (freqs >= f_lo) & (freqs <= f_hi) & (freqs > 0)


def _cwt_energy_variance(x, fs=FS, n_subwindows=10, scales=None):
    """
    CWT-based transient/impact signature. Corrugation produces periodic
    *impacts*, which show up as bursts of energy in the wavelet domain rather
    than a steady tone. We compute total wavelet power in each of
    n_subwindows equal slices of the 1s recording, then return the
    coefficient of variation across slices: a high value means energy is
    concentrated in short bursts (impact-like / corrugation-like), a low
    value means it's smoothly spread out (broadband ballast noise).
    """
    if scales is None:
        scales = np.arange(2, 40)  # roughly 50-1000 Hz at this rate, morlet
    coeffs, _ = pywt.cwt(x, scales, "morl", sampling_period=1 / fs)
    power = np.sum(coeffs**2, axis=0)  # total wavelet power per time sample
    edges = np.linspace(0, len(power), n_subwindows + 1).astype(int)
    window_energy = np.array([power[edges[i]:edges[i + 1]].sum() for i in range(n_subwindows)])
    mean_e = window_energy.mean()
    if mean_e < 1e-12:
        return 0.0
    return float(window_energy.std() / mean_e)


def signal_features(x, speed, fs=FS, use_cwt=True):
    """Time + frequency + wavelet domain features for one channel."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    crest = peak / rms if rms > 1e-9 else 0.0
    mu, sig = np.mean(x), np.std(x)
    kurt = np.mean((x - mu) ** 4) / (sig**4) if sig > 1e-9 else 0.0

    # speed-normalized variants — amplitude often scales with speed/contact
    # force, so we keep both raw and normalized and let feature importance
    # decide which is more useful, rather than committing to one.
    norm = speed if speed > 0.5 else np.nan
    rms_norm = rms / norm if not np.isnan(norm) else np.nan
    kurt_norm = kurt / norm if not np.isnan(norm) else np.nan

    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mag = np.abs(np.fft.rfft(x - mu))
    mask = _band_mask(freqs, speed)
    total_energy = np.sum(mag**2) + 1e-12
    band_energy_ratio = np.sum(mag[mask] ** 2) / total_energy if mask.any() else 0.0

    if mask.any():
        band_freqs, band_mag = freqs[mask], mag[mask]
        dom_idx = np.argmax(band_mag)
        dom_freq = band_freqs[dom_idx]
        wavelength = speed / dom_freq if dom_freq > 0 else np.nan
        dom_amp = float(band_mag[dom_idx])
    else:
        wavelength, dom_amp = np.nan, 0.0

    feats = {
        "rms": rms, "rms_norm": rms_norm, "peak": peak, "crest": crest,
        "kurt": kurt, "kurt_norm": kurt_norm,
        "band_energy_ratio": band_energy_ratio, "wavelength": wavelength,
        "dom_amp": dom_amp,
    }
    if use_cwt:
        feats["cwt_energy_cv"] = _cwt_energy_variance(x, fs=fs)
    return feats


# --------------------------------------------------------------------------- #
# Per-file feature extraction
# --------------------------------------------------------------------------- #

STAT_NAMES = ["rms", "rms_norm", "peak", "crest", "kurt", "kurt_norm",
              "band_energy_ratio", "dom_amp", "cwt_energy_cv"]


def extract_file_features(csv_path_or_df, use_cwt=True):
    """
    Full feature vector for one recording. Accepts either a path to a CSV
    or an already-loaded DataFrame (the latter is used by the augmentation
    pipeline, which perturbs the raw signal in memory before extracting
    features — it never writes augmented CSVs to disk).
    """
    df = pd.read_csv(csv_path_or_df) if isinstance(csv_path_or_df, str) else csv_path_or_df
    meta = parse_columns(df.columns)
    speed, transitions = estimate_speed(df.iloc[:, 0].values)

    out = {"speed": speed, "transitions": transitions,
           "is_stationary": int(transitions < STATIONARY_TRANSITIONS_THRESHOLD)}
    per_side_stats = {}
    for side, prefix in (("Side I", "s1"), ("Side II", "s2")):
        side_cols = meta[meta.side == side]
        for kind, kprefix in (("Vibration", "vibr"), ("Shock", "shoc")):
            sub_cols = side_cols[side_cols.kind == kind]
            rows = [signal_features(df[c].values, speed, use_cwt=use_cwt) for c in sub_cols.col]
            pc = pd.DataFrame(rows)
            for stat in STAT_NAMES:
                vals = pc[stat].values
                vals_clean = vals[~np.isnan(vals)]
                mean_v = np.mean(vals_clean) if len(vals_clean) else np.nan
                max_v = np.max(vals_clean) if len(vals_clean) else np.nan
                std_v = np.std(vals_clean) if len(vals_clean) else np.nan
                out[f"{prefix}_{kprefix}_{stat}_mean"] = mean_v
                out[f"{prefix}_{kprefix}_{stat}_max"] = max_v
                out[f"{prefix}_{kprefix}_{stat}_std"] = std_v
                per_side_stats[(side, kprefix, stat)] = mean_v

    # Side-differential features (Side I vs Side II), both diff and ratio —
    # ratios partly cancel out overall signal-intensity/speed effects that
    # raw differences don't.
    for kprefix in ("vibr", "shoc"):
        for stat in STAT_NAMES:
            v1 = per_side_stats.get(("Side I", kprefix, stat), np.nan)
            v2 = per_side_stats.get(("Side II", kprefix, stat), np.nan)
            out[f"diff_{kprefix}_{stat}"] = v1 - v2
            out[f"ratio_{kprefix}_{stat}"] = (
                v1 / v2 if (v2 is not None and not np.isnan(v2) and abs(v2) > 1e-9) else np.nan
            )

    return out


def batch_extract_features(file_paths, use_cwt=True, n_jobs=1, show_progress=True):
    """
    Extract features for a list of CSV paths. Returns a DataFrame indexed by
    filename. n_jobs>1 uses a process pool (safe on Colab's multi-core CPU).
    """
    filenames = [p.split("/")[-1] for p in file_paths]

    if n_jobs and n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(extract_file_features, file_paths))
    else:
        iterator = file_paths
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(file_paths, desc="extracting features")
        results = [extract_file_features(p) for p in iterator]

    feat_df = pd.DataFrame(results)
    feat_df.insert(0, "filename", filenames)
    return feat_df


# --------------------------------------------------------------------------- #
# Augmentation — applied to RAW signals in memory, before feature extraction.
# Only time-shift / crop / additive noise. No synthetic speed alteration:
# that would desynchronize the frequency<->speed relationship the wavelength
# feature depends on.
# --------------------------------------------------------------------------- #

def augment_dataframe(df, rng, max_shift_frac=0.15, noise_std_frac=0.02):
    """
    Return an augmented copy of a raw recording DataFrame (same shape).
    - Circular time-shift by a random offset (keeps 1s duration, preserves
      periodicity of the corrugation signature).
    - Additive Gaussian noise scaled to each vibration/shock column's own
      std (proportional, not a fixed magnitude across very different-scale
      channels).
    Column 1 (speed sensor) is shifted identically but never noised — it's a
    clean digital signal, not a physical measurement to perturb.
    """
    n = len(df)
    shift = rng.integers(-int(n * max_shift_frac), int(n * max_shift_frac) + 1)
    out = df.copy()
    for col in df.columns:
        shifted = np.roll(df[col].values, shift)
        if col == df.columns[0]:
            out[col] = shifted
        else:
            std = np.std(shifted)
            noise = rng.normal(0, noise_std_frac * std, size=n) if std > 0 else 0
            out[col] = shifted + noise
    return out


def crop_and_pad(df, rng, crop_frac=0.8):
    """
    Randomly crop a crop_frac-length window and tile it back to the original
    length (keeps downstream feature code, which assumes a fixed sample
    count, unchanged).
    """
    n = len(df)
    crop_len = int(n * crop_frac)
    start = rng.integers(0, n - crop_len + 1)
    cropped = df.iloc[start:start + crop_len].reset_index(drop=True)
    reps = int(np.ceil(n / crop_len))
    tiled = pd.concat([cropped] * reps, ignore_index=True).iloc[:n].reset_index(drop=True)
    return tiled


# --------------------------------------------------------------------------- #
# Data audit — shared so predict.py can warn about the same issues train.py
# checks for (e.g. near-stationary / degenerate-speed recordings, which zero
# out the wavelength/band-energy features and can systematically skew
# predictions for those files if not accounted for).
# --------------------------------------------------------------------------- #

def audit_files(file_paths, min_transitions=STATIONARY_TRANSITIONS_THRESHOLD):
    print(f"\n[audit] checking {len(file_paths)} files...")
    issues = []
    for p in file_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            issues.append((p, f"failed to read: {e}"))
            continue
        if df.shape != (10000, 129):
            issues.append((p, f"unexpected shape {df.shape}"))
        if df.isna().any().any():
            issues.append((p, "contains NaNs"))
        _, transitions = estimate_speed(df.iloc[:, 0].values)
        if transitions < min_transitions:
            issues.append((p, f"degenerate speed signal ({transitions} transitions) — "
                              f"likely a near-stationary recording"))
    if issues:
        print(f"[audit] {len(issues)} issue(s) found:")
        for p, msg in issues[:20]:
            print(f"   - {os.path.basename(p)}: {msg}")
        if len(issues) > 20:
            print(f"   ... and {len(issues) - 20} more")
    else:
        print("[audit] no issues found.")
    return issues
