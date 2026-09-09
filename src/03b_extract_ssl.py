"""Mean-pooled WavLM representations for every clip (frozen, CPU only).

Hidden states are averaged over the valid frames of each utterance and then over
transformer layers 1-12; no layer is selected on task performance, so nothing
here is tuned against either label set.

Results are written in shards so that the run can be resumed: only the layer
average is retained, which keeps the resident set small enough to sit alongside
the rest of the pipeline.
"""
import sys
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModel

from common import FEAT, ROOT

AUDIO = ROOT / "data" / "cremad_audio_repo" / "AudioWAV"
SHARDS = FEAT / "ssl_shards"
MODEL = "microsoft/wavlm-base-plus"
SR = 16000
BATCH = 8
SHARD = 400          # clips per shard file


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) \
        if "--limit" in sys.argv else None
    torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1])
                          if "--threads" in sys.argv else 8)
    SHARDS.mkdir(parents=True, exist_ok=True)

    clips = sorted(p.name for p in AUDIO.glob("*.wav"))
    if limit:
        clips = clips[:limit]
    groups = [clips[i:i + SHARD] for i in range(0, len(clips), SHARD)]
    todo = [g for g in enumerate(groups) if not (SHARDS / f"s{g[0]:04d}.npy").exists()]
    print(f"{MODEL}: {len(clips)} clips in {len(groups)} shards, "
          f"{len(todo)} still to do", flush=True)
    if not todo:
        print("all shards present")
    else:
        fe = AutoFeatureExtractor.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL, output_hidden_states=True).eval()
        t0 = time.time()
        for done, (gi, group) in enumerate(todo, 1):
            vecs = []
            for i in range(0, len(group), BATCH):
                chunk = group[i:i + BATCH]
                waves = []
                for c in chunk:
                    y, sr = sf.read(str(AUDIO / c), dtype="float32")
                    if y.ndim > 1:
                        y = y.mean(1)
                    assert sr == SR, sr
                    waves.append(y)
                inp = fe(waves, sampling_rate=SR, return_tensors="pt",
                         padding=True, return_attention_mask=True)
                with torch.no_grad():
                    res = model(input_values=inp.input_values,
                                attention_mask=inp.attention_mask)
                    lens = model._get_feat_extract_output_lengths(
                        inp.attention_mask.sum(-1)).to(torch.long)
                    T = res.hidden_states[0].shape[1]
                    mask = (torch.arange(T)[None, :] < lens[:, None]).float()
                    denom = mask.sum(1)[:, None]
                    acc = None
                    for h in res.hidden_states[1:]:          # layers 1-12
                        pooled = (h * mask[:, :, None]).sum(1) / denom
                        acc = pooled if acc is None else acc + pooled
                    vecs.append((acc / (len(res.hidden_states) - 1)).numpy())
                del res
            np.save(SHARDS / f"s{gi:04d}.npy",
                    np.concatenate(vecs).astype(np.float32))
            el = time.time() - t0
            print(f"  shard {gi} ({done}/{len(todo)})  {el:.0f}s  "
                  f"eta {el/done*(len(todo)-done):.0f}s", flush=True)

    if limit:
        return
    X = np.concatenate([np.load(SHARDS / f"s{i:04d}.npy")
                        for i in range(len(groups))])
    assert len(X) == len(clips), (len(X), len(clips))
    np.savez_compressed(FEAT / "ssl.npz", X=X,
                        clips=np.array([c[:-4] for c in clips]))
    print("wrote", FEAT / "ssl.npz", X.shape)


if __name__ == "__main__":
    main()
