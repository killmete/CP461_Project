# %% [markdown]
# ## 5 · E1: Detector evaluation on the **test** split (v1 = progress report, v2 = final)
# Ultralytics metrics (P, R and mAP per class) for both detectors. The final pipeline below uses **v2**; section 5.1 adds imbalance-aware metrics, ROC/PR curves and before/after confusion matrices.

# %%
def yolo_test_table(weights, imgsz, name):
    mdl = YOLO(str(weights))
    m = mdl.val(data=str(DET_YAML), split="test", imgsz=imgsz, batch=BATCH, device=YOLO_DEVICE,
                project=str(RUNS), name=name, exist_ok=True, plots=True, verbose=False)
    rows = []
    for j, c in enumerate(m.ap_class_index):
        p, r = float(m.box.p[j]), float(m.box.r[j])
        rows.append([CLASSES[c], int(stats.loc["test", CLASSES[c]]), p, r, 2 * p * r / (p + r + 1e-9), m.box.ap50[j], m.box.ap[j]])
    mp, mr = float(m.box.mp), float(m.box.mr)
    rows.append(["all", int(stats.loc["test", "objects"]), mp, mr, 2 * mp * mr / (mp + mr + 1e-9), m.box.map50, m.box.map])
    t = pd.DataFrame(rows, columns=["class", "GT objects", "Precision", "Recall", "F1", "mAP50", "mAP50-95"]).set_index("class")
    t[["Precision", "Recall", "F1", "mAP50", "mAP50-95"]] = t[["Precision", "Recall", "F1", "mAP50", "mAP50-95"]].astype(float).round(3)
    for f in (RUNS / name).glob("*.png"):
        shutil.copy(f, FIG / f"{name}_{f.name}")
    del mdl; free_gpu()
    return t

det_table_v1 = yolo_test_table(DET_V1_BEST, IMG_SIZE, "det_test")              # v1: same numbers as the progress report
det_table_v1.to_csv(OUT / "E1_detection_metrics.csv")
print("v1 (progress report detector):"); display(det_table_v1)
if DET_V2_BEST is not None:
    det_table_v2 = yolo_test_table(DET_V2_BEST, DET_V2_IMG_SIZE, "det_test_v2")
    det_table_v2.to_csv(OUT / "E1_detection_metrics_v2.csv")
    print("v2 (final detector):"); display(det_table_v2)
    e1_compare = pd.concat({"v1": det_table_v1, "v2": det_table_v2}, axis=1).swaplevel(axis=1)[["Precision", "Recall", "F1", "mAP50", "mAP50-95"]]
    e1_compare.to_csv(OUT / "TABLE_E1_v1_vs_v2.csv"); display(e1_compare)
    DET_BEST, DET_IMGSZ, det_table = DET_V2_BEST, DET_V2_IMG_SIZE, det_table_v2
else:
    DET_BEST, DET_IMGSZ, det_table = DET_V1_BEST, IMG_SIZE, det_table_v1
det = YOLO(str(DET_BEST))                                # the detector used by everything below (E3–E7, prototype)
print(f"→ pipeline detector: {DET_BEST}  @ {DET_IMGSZ}px")
