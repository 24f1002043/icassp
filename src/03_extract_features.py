"""Extract two CPU-only acoustic representations for every CREMA-D clip.

  mfcc    - 13 MFCC means and standard deviations (26-d); the classic
            low-capacity baseline.
  funcs   - an eGeMAPS-style functional set built from prosodic, spectral and
            cepstral low-level descriptors (~380-d).

Both are written to data/features/<name>.npz keyed by clip name.
"""
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import FEAT, ROOT

warnings.filterwarnings("ignore")
AUDIO = ROOT / "data" / "cremad_audio_repo" / "AudioWAV"
SR = 16000


def _functionals(x, prefix, with_slope=True):
    """Summarise a 1-D (time,) or 2-D (dim, time) descriptor."""
    x = np.atleast_2d(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.arange(x.shape[1], dtype=float)
    tc = t - t.mean() if x.shape[1] > 1 else t
    denom = (tc ** 2).sum() or 1.0
    feats, names = [], []
    stats = {
        "mean": x.mean(1), "std": x.std(1),
        "p1": np.percentile(x, 1, axis=1), "p99": np.percentile(x, 99, axis=1),
        "median": np.median(x, axis=1),
        "iqr": np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1),
    }
    stats["range"] = stats["p99"] - stats["p1"]
    if with_slope:
        stats["slope"] = (x - x.mean(1, keepdims=True)) @ tc / denom
    for k, v in stats.items():
        feats.append(v)
        names += [f"{prefix}_{k}_{i}" for i in range(x.shape[0])]
    return np.concatenate(feats), names


def extract(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    if y.size < SR // 10:                       # guard against near-empty clips
        y = np.pad(y, (0, SR // 10 - y.size))
    y = y / (np.abs(y).max() + 1e-9)

    n_fft, hop = 400, 160                       # 25 ms window, 10 ms hop
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=40)
    logmel = librosa.power_to_db(mel + 1e-10)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=20)

    # ---- the small baseline representation ---------------------------------
    m13 = mfcc[:13]
    mfcc_vec = np.concatenate([m13.mean(1), m13.std(1)])

    # ---- the functional representation -------------------------------------
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)[0]
    loud = 20 * np.log10(rms + 1e-8)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft,
                                             hop_length=hop)[0]
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    bw = librosa.feature.spectral_bandwidth(S=S, sr=SR)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85)[0]
    flat = librosa.feature.spectral_flatness(S=S)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR)

    f0 = librosa.yin(y, fmin=55, fmax=450, sr=SR, frame_length=1024,
                     hop_length=hop)
    logf0 = np.log(np.clip(f0, 55, 450))
    voiced = (rms > np.percentile(rms, 40)).astype(float)
    vf = voiced.mean()

    parts, names = [], []
    for arr, pre in [(mfcc, "mfcc"), (d1, "dmfcc"), (d2, "ddmfcc"),
                     (logmel[::4], "logmel")]:
        f, n = _functionals(arr, pre)
        parts.append(f); names += n
    for arr, pre in [(loud, "loud"), (logf0, "logf0"), (zcr, "zcr"),
                     (cent, "cent"), (bw, "bw"), (roll, "roll"),
                     (flat, "flat")]:
        f, n = _functionals(arr, pre)
        parts.append(f); names += n
    f, n = _functionals(contrast, "contrast")
    parts.append(f); names += n

    extra = np.array([vf, len(y) / SR, np.log(len(y) / SR),
                      float(np.mean(np.abs(np.diff(logf0)))),
                      float(np.mean(np.abs(np.diff(loud))))])
    parts.append(extra)
    names += ["voiced_frac", "dur", "log_dur", "f0_jitter", "loud_shimmer"]

    return mfcc_vec, np.nan_to_num(np.concatenate(parts)), names


def _job(name):
    try:
        a, b, _ = extract(str(AUDIO / name))
        return name, a.astype(np.float32), b.astype(np.float32)
    except Exception as exc:                     # keep the pool alive
        return name, None, repr(exc)


def main():
    clips = sorted(p.name for p in AUDIO.glob("*.wav"))
    if "--bench" in sys.argv:
        clips = clips[:24]
    print(f"extracting from {len(clips)} clips using 14 workers")
    t0 = time.time()
    res, bad = {}, []
    with ProcessPoolExecutor(max_workers=14) as ex:
        for i, (name, a, b) in enumerate(ex.map(_job, clips, chunksize=8), 1):
            if a is None:
                bad.append((name, b)); continue
            res[name] = (a, b)
            if i % 500 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(clips)}  {el:.0f}s  "
                      f"eta {el/i*(len(clips)-i):.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s; {len(bad)} failures")
    for n, e in bad[:5]:
        print("  FAIL", n, e)

    names = sorted(res)
    keys = [n[:-4] for n in names]               # strip .wav
    mfcc = np.stack([res[n][0] for n in names])
    funcs = np.stack([res[n][1] for n in names])
    _, _, fnames = extract(str(AUDIO / names[0]))
    print("mfcc", mfcc.shape, "funcs", funcs.shape)
    if "--bench" not in sys.argv:
        np.savez_compressed(FEAT / "mfcc.npz", X=mfcc, clips=np.array(keys))
        np.savez_compressed(FEAT / "funcs.npz", X=funcs, clips=np.array(keys),
                            names=np.array(fnames))
        print("wrote", FEAT / "mfcc.npz", "and", FEAT / "funcs.npz")


if __name__ == "__main__":
    main()
