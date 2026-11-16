# %% [markdown]
# ### 5.2 Before / after: detector v1 (progress report) vs v2 (final)
# Same test split, same matching rule (box IoU ≥ 0.5). For each detector, two operating points are reported: the global threshold `CONF_THRES` and per-class thresholds tuned on the **val** split (method 2 for the confusion matrix). ROC/PR curves are threshold-free.

# %%
t0_e7 = time.time()
gt_val, val_ids = load_gt("val", max_images=MAX_TEST_IMAGES)

# ---- v1: re-predict with the progress-report detector (640 px) ----
det_v1 = YOLO(str(DET_V1_BEST))
preds_test_v1 = det_preds if DET_V2_BEST is None else predict_ids(det_v1, test_ids, imgsz=IMG_SIZE, desc="detector v1 on test")
preds_val_v1 = predict_ids(det_v1, val_ids, imgsz=IMG_SIZE, desc="detector v1 on val")
del det_v1; free_gpu()
rep = {f"v1 · conf {CONF_THRES}": detection_report(preds_test_v1, f"v1 · conf {CONF_THRES}", conf_thr=CONF_THRES, detector="v1")}
thr_v1, thr_table_v1 = tune_class_thresholds(preds_val_v1, gt_val, val_ids)
rep["v1 · per-class thr (val)"] = detection_report(preds_test_v1, "v1 · per-class thr (val)", conf_thr=thr_v1, detector="v1")
preds_by_det = {"v1": preds_test_v1}

# ---- v2: the pipeline detector (det_preds from section 5 were made with it) ----
if DET_V2_BEST is not None:
    preds_val_v2 = predict_ids(det, val_ids, imgsz=DET_IMGSZ, desc="detector v2 on val")
    rep[f"v2 · conf {CONF_THRES}"] = detection_report(det_preds, f"v2 · conf {CONF_THRES}", conf_thr=CONF_THRES, detector="v2")
    thr_v2, thr_table_v2 = tune_class_thresholds(preds_val_v2, gt_val, val_ids)
    rep["v2 · per-class thr (val)"] = detection_report(det_preds, "v2 · per-class thr (val)", conf_thr=thr_v2, detector="v2")
    preds_by_det["v2"] = det_preds
    chk = rep[f"v2 · conf {CONF_THRES}"]["per_class"][["TP", "FP", "FN"]].values == thr_table.loc[CLASSES, ["TP", "FP", "FN"]].values
    print("sanity check vs section 5 table:", "OK ✓" if chk.all() else "MISMATCH ✗")

auc_e7 = roc_pr_report(preds_by_det)
last = list(rep)[-2] if DET_V2_BEST is not None else list(rep)[0]          # v2 at the global threshold (or v1 alone)
first = f"v1 · conf {CONF_THRES}"
cmp_pair = compare_detectors({first: rep[first], last: rep[last]} if last != first else {first: rep[first]}, auc_e7)
cmp_e7 = compare_detectors(rep, auc_e7, fig_suffix="_all")                 # all 4 settings → TABLE_E7_detector_comparison.csv

# ---- findings (computed, not typed) ----
a, b = rep[first]["summary"], rep[last]["summary"]
pa, pb_ = rep[first]["per_class"], rep[last]["per_class"]
print(f"\nE7 findings ({int(a['GT objects'])} GT objects in {len(test_ids)} test images):")
print(f"• 'Accuracy of localised objects' (progress headline) hides the missed objects: v1 {a['Accuracy (localised objects)']:.3f} "
      f"but only {a['Accuracy (all GT objects)']:.3f} over all GT objects (balanced accuracy {a['Balanced accuracy']:.3f}).")
if last != first:
    print(f"• v1 → v2 at conf {CONF_THRES}: macro-F1 {a['F1 (macro)']:.3f} → {b['F1 (macro)']:.3f} · balanced accuracy "
          f"{a['Balanced accuracy']:.3f} → {b['Balanced accuracy']:.3f} · MCC {a['MCC']:.3f} → {b['MCC']:.3f} · "
          f"macro ROC-AUC {auc_e7.loc[('v1', 'macro'), 'ROC-AUC']:.3f} → {auc_e7.loc[('v2', 'macro'), 'ROC-AUC']:.3f}")
    print("• Recall per class v1 → v2: " + " · ".join(f"{c} {pa.loc[c, 'Recall']:.2f}→{pb_.loc[c, 'Recall']:.2f}" for c in CLASSES))
    t2 = rep["v2 · per-class thr (val)"]["summary"]
    print(f"• Per-class thresholds on v2: macro-F1 {b['F1 (macro)']:.3f} → {t2['F1 (macro)']:.3f}, balanced accuracy "
          f"{b['Balanced accuracy']:.3f} → {t2['Balanced accuracy']:.3f} (thresholds trade recall for precision; they cannot find missed objects).")
print(f"E7 done in {time.time() - t0_e7:.0f}s")
