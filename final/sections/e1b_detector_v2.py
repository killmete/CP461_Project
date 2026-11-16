# %% [markdown]
# ### 4.3 Detector v2: fix the confusion matrix (≈60–75 min on T4)
# At the progress report the detector was the bottleneck: it found only 31% of the litter, and almost no **Glass** (recall 0.04) or **Other Waste** (0.06).
# The confusion matrix showed that most errors were *missed objects* (true class → background), not class swaps. v2 targets the two causes:
# * **Small objects** (43% of test objects are smaller than 32×32 px): train and predict at **`DET_V2_IMG_SIZE` = 960 px** instead of 640.
# * **Class imbalance** (Plastic 47% of objects vs Glass 5%): **repeat-factor sampling** (LVIS, Gupta et al. 2019). Training images that contain a rare class are shown more often in each epoch: class *c* gets repeat factor r_c = max(1, √(t / f_c)), where f_c is the share of training images containing *c*.
#
# v1 is kept unchanged as the "before" model, so every table and figure below compares **v1 (progress) vs v2 (final)** on the same test split.

# %%
def rfs_yaml(data_yaml, t=RFS_T):
    # Repeat-factor sampling: write a train list in which images with rare classes appear several times
    root = Path(data_yaml).parent
    img_cls = {}
    for f in sorted((root / "labels/train").glob("*.txt")):
        img_cls[f.stem] = {int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()}
    n = len(img_cls)
    freq = Counter(c for cs in img_cls.values() for c in cs)
    r = {c: max(1.0, math.sqrt(t / (freq[c] / n))) for c in freq}
    lines, reps = [], Counter()
    for stem, cs in img_cls.items():
        k = max(1, round(max((r[c] for c in cs), default=1.0)))
        reps[k] += 1
        lines += [str(root / "images/train" / f"{stem}.jpg")] * k
    (root / "train_rfs.txt").write_text("\n".join(lines))
    cfg = yaml.safe_load(Path(data_yaml).read_text()); cfg["train"] = str(root / "train_rfs.txt")
    out = root / "data_rfs.yaml"; out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    rfs_table = pd.DataFrame({"images with class (train)": [freq[k] for k in range(len(CLASSES))],
                              "share of images": [freq[k] / n for k in range(len(CLASSES))],
                              "repeat factor": [r[k] for k in range(len(CLASSES))]}, index=CLASSES).round(3)
    display(rfs_table); rfs_table.to_csv(OUT / "TABLE_E1b_repeat_factors.csv")
    print(f"train list: {len(lines)} entries for {n} images · images by repeat count: {dict(sorted(reps.items()))}")
    return out

DET_V1_BEST = DET_BEST                                   # progress-report detector (640 px, no rebalancing)
DET_V2_BEST = None
if TRAIN_DET_V2:
    v2_yaml = rfs_yaml(DET_YAML) if DET_V2_RFS else DET_YAML
    t0 = time.time()
    DET_V2_BEST = train_yolo(DET_V2_MODEL, v2_yaml, "yolo_det_v2", DET_V2_EPOCHS, imgsz=DET_V2_IMG_SIZE, batch=DET_V2_BATCH)
    print(f"detector v2 ready: {DET_V2_BEST}  ({(time.time() - t0) / 60:.1f} min)")
    for f in ["results.png", "BoxPR_curve.png", "confusion_matrix_normalized.png"]:
        if (RUNS / "yolo_det_v2" / f).exists():
            shutil.copy(RUNS / "yolo_det_v2" / f, FIG / f"det_v2_train_{f}")
else:
    print("TRAIN_DET_V2 = False → the final results use detector v1")
