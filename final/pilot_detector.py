"""Local detector pilots (lead's scratch experiment, NOT part of the Colab notebook).
Decides which detector-v2 recipe the Colab notebook should train. Uses the YOLO dataset that the
harness built in final/_localwork/litter_local/yolo_det (same split as the notebook).

  A  = yolo11s @ 960 px                    (resolution: small litter)
  B  = yolo11s @ 960 px + repeat-factor oversampling of images with rare classes (imbalance)
"""
import math, sys, time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
YD = ROOT / "_localwork/pilot_data/yolo_det"   # private copy: every harness run rebuilds litter_local
RUNS = ROOT / "_localwork/pilot_runs"


def rfs_train_list(t=0.3):
    # LVIS-style repeat factor sampling: r_c = max(1, sqrt(t / f_c)), image repeat = round(max r_c over its classes)
    labels = sorted((YD / "labels/train").glob("*.txt"))
    img_cls = {}
    for f in labels:
        cs = {int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()}
        img_cls[f.stem] = cs
    n = len(img_cls)
    freq = Counter(c for cs in img_cls.values() for c in cs)
    r = {c: max(1.0, math.sqrt(t / (freq[c] / n))) for c in freq}
    lines, reps = [], Counter()
    for stem, cs in img_cls.items():
        k = max(1, round(max((r[c] for c in cs), default=1.0)))
        reps[k] += 1
        img = YD / "images/train" / f"{stem}.jpg"
        lines += [str(img)] * k
    out = YD / "train_rfs.txt"
    out.write_text("\n".join(lines))
    print("image freq per class:", {c: round(freq[c] / n, 3) for c in sorted(freq)})
    print("repeat factor per class:", {c: round(r[c], 2) for c in sorted(r)}, "| images by repeat:", dict(reps),
          "| train list:", len(lines), "entries for", n, "images")
    yaml = (YD / "data.yaml").read_text().replace("train: images/train", f"train: {out.as_posix()}")
    y2 = YD / "data_rfs.yaml"
    y2.write_text(yaml)
    return y2


if __name__ == "__main__":
    from ultralytics import YOLO
    which = sys.argv[1:] or ["A", "B"]
    for w in which:
        data = YD / "data.yaml" if w == "A" else rfs_train_list()
        t0 = time.time()
        run = RUNS / f"pilot_{w}"
        for attempt in range(6):                       # shared GPU: resume after transient CUDA/cuDNN failures
            try:
                if (run / "weights/last.pt").exists():
                    YOLO(str(run / "weights/last.pt")).train(resume=True)
                else:
                    YOLO("yolo11s.pt").train(data=str(data), epochs=80, imgsz=960, batch=8, patience=20, seed=42,
                                             project=str(RUNS), name=f"pilot_{w}", exist_ok=True, cache=True,
                                             plots=True, device=0, workers=2, verbose=False)
                break
            except Exception as e:
                msg = str(e)
                if "nothing to resume" in msg or "is finished" in msg:
                    break
                print(f"PILOT {w} attempt {attempt} failed: {msg[:200]} → resuming", flush=True)
                import gc, torch; gc.collect(); torch.cuda.empty_cache(); time.sleep(20)
        print(f"PILOT {w} done in {(time.time() - t0) / 60:.1f} min", flush=True)
