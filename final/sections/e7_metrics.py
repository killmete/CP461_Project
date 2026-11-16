# %% [markdown]
# ## E7 · Imbalance-aware metrics for the detector (answers "accuracy alone is not enough")
# The test split is **imbalanced**: Plastic has about 7× more objects than Glass. A detector that is good only on Plastic
# can still get a high accuracy. So here we report metrics that give **every class the same weight**, plus ROC / PR curves.
# The code is a small reusable API, so the same functions evaluate **any** detector (v1 now, v2 later → before / after).
#
# | Function | What it does |
# |---|---|
# | `load_gt(split)` | ground-truth boxes + classes for `"val"` or `"test"` |
# | `predict_ids(model, ids)` | run a YOLO detector on these images (conf ≥ 0.001, one image per call) |
# | `detection_report(preds, name, conf_thr=...)` | confusion matrix + per-class table + summary metrics (one global or 5 per-class thresholds) |
# | `roc_pr_report({name: preds})` | object-level ROC-AUC and PR-AUC per class, with figures |
# | `tune_class_thresholds(preds_val, gt_val, ids_val)` | one confidence threshold per class, chosen on **val** (never on test) |
# | `compare_detectors({name: report})` | comparison table + before/after confusion matrices + F1 per class |

# %%
from sklearn.metrics import (cohen_kappa_score, matthews_corrcoef, roc_auc_score, roc_curve,
                             average_precision_score, precision_recall_curve)
import re

NC = len(CLASSES)
E7_LINE_COLORS = ["#898781", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # 1st detector grey = "before"


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")


def load_gt(split, max_images=None):
    """GT dict {img_id: dict(anns, boxes xyxy, cls, rles, area)} + sorted ids for 'val' or 'test' (same format as GT)."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(COCO_JSON[split]))
    ids = sorted(coco.getImgIds())
    if max_images:
        ids = ids[:max_images]
    gt = {}
    for i in ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=i))
        boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                          for a in anns], np.float32).reshape(-1, 4)
        gt[i] = dict(anns=anns, boxes=boxes, cls=np.array([a["category_id"] - 1 for a in anns], int),
                     rles=[coco.annToRLE(a) for a in anns], area=np.array([a["area"] for a in anns], float))
    return gt, ids


def predict_ids(model, ids, imgsz=IMG_SIZE, desc="detector", **kw):
    """{img_id: dict(boxes, scores, cls)} like det_preds. Extra kwargs go to model.predict (e.g. augment=True)."""
    args = dict(conf=0.001, iou=0.7, max_det=100, imgsz=imgsz, device=YOLO_DEVICE, verbose=False)
    args.update(kw)
    preds = {}
    for i in tqdm(ids, desc=desc):             # one image per call (a list of paths = one big batch → GPU memory)
        r = model.predict(str(FAST_IMG / images_all[i]["file_name"]), **args)[0]
        preds[i] = dict(boxes=r.boxes.xyxy.cpu().numpy().astype(np.float32), scores=r.boxes.conf.cpu().numpy(),
                        cls=r.boxes.cls.cpu().numpy().astype(int))
    return preds


def _thr_vec(conf_thr):
    t = np.asarray(conf_thr, float)
    return np.full(NC, float(t)) if t.ndim == 0 else t.reshape(NC)


def _match_all(preds, gt, ids, conf_thr):
    """Greedy class-agnostic matching (cell 24 `match`) → 6×6 CM (last = background) + per-GT-object labels."""
    thr = _thr_vec(conf_thr)
    cm_ = np.zeros((NC + 1, NC + 1), int)
    y_true, y_pred = [], []                        # one entry per GT object; y_pred = NC means "missed"
    for i in ids:
        p, g = preds[i], gt[i]
        k = p["scores"] >= thr[p["cls"]] if len(p["cls"]) else np.zeros(0, bool)
        pb, ps, pc = p["boxes"][k], p["scores"][k], p["cls"][k]
        pairs, ump, umg = match(pb, ps, g["boxes"])
        lab = np.full(len(g["cls"]), NC)
        for pi, gi in pairs:
            cm_[g["cls"][gi], pc[pi]] += 1; lab[gi] = pc[pi]
        for pi in ump:
            cm_[NC, pc[pi]] += 1
        for gi in umg:
            cm_[g["cls"][gi], NC] += 1
        y_true.extend(g["cls"].tolist()); y_pred.extend(lab.tolist())
    return cm_, np.array(y_true, int), np.array(y_pred, int)


def _per_class(cm_):
    tp = np.diag(cm_)[:NC].astype(float)
    fp = cm_[:, :NC].sum(0) - tp
    support = cm_[:NC, :].sum(1).astype(float)     # every GT object of the class (matched or missed)
    fn = support - tp
    P = tp / np.maximum(tp + fp, 1); R = tp / np.maximum(support, 1)
    F = 2 * P * R / np.maximum(P + R, 1e-9)
    return tp, fp, fn, support, P, R, F


def detection_report(preds, name, gt=None, ids=None, conf_thr=CONF_THRES, detector=None, save=True, show=True):
    """conf_thr: float, or 5 per-class thresholds. Returns dict(name, detector, conf_thr, cm, per_class, summary, y_true, y_pred)."""
    gt = GT if gt is None else gt
    ids = test_ids if ids is None else ids
    thr = _thr_vec(conf_thr)
    cm_, y_true, y_pred = _match_all(preds, gt, ids, thr)
    tp, fp, fn, support, P, R, F = _per_class(cm_)
    per_class = pd.DataFrame({"conf_thr": thr, "support": support.astype(int), "TP": tp.astype(int), "FP": fp.astype(int),
                              "FN": fn.astype(int), "Precision": P, "Recall": R, "F1": F}, index=CLASSES)
    per_class.index.name = "class"
    has = support > 0                               # classes absent from the split (tiny dry runs) are left out of means
    mP, mR = tp.sum() / max(tp.sum() + fp.sum(), 1), tp.sum() / max(support.sum(), 1)
    matched = cm_[:NC, :NC].sum()
    summary = pd.Series({
        "Precision (micro)": mP, "Recall (micro)": mR, "F1 (micro)": 2 * mP * mR / max(mP + mR, 1e-9),
        "Precision (macro)": P[has].mean(), "Recall (macro)": R[has].mean(), "F1 (macro)": F[has].mean(),
        "F1 (weighted)": (F * support).sum() / max(support.sum(), 1),
        "Balanced accuracy": R[has].mean(),
        "Accuracy (all GT objects)": tp.sum() / max(support.sum(), 1),
        "Accuracy (localised objects)": np.trace(cm_[:NC, :NC]) / max(matched, 1),
        "Cohen kappa": cohen_kappa_score(y_true, y_pred, labels=list(range(NC + 1))) if len(y_true) else np.nan,
        "MCC": matthews_corrcoef(y_true, y_pred) if len(y_true) else np.nan,
        "GT objects": int(support.sum()), "Localised (IoU≥%.1f)" % MATCH_IOU: int(matched),
        "FP on background": int(cm_[NC, :NC].sum()),
    }, name=name)
    if save:
        per_class.round(3).to_csv(OUT / f"TABLE_E7_{_slug(name)}_per_class.csv")
        summary.round(4).to_frame().to_csv(OUT / f"TABLE_E7_{_slug(name)}_summary.csv")
    if show:
        print(f"── {name}  (conf_thr = {np.round(thr, 2).tolist()})")
        display(per_class.round(3)); display(summary.round(3).to_frame())
    return dict(name=name, detector=detector or name, conf_thr=thr, cm=cm_, per_class=per_class, summary=summary,
                y_true=y_true, y_pred=y_pred)

# %% [markdown]
# ### What each summary metric means
# We match predicted boxes to GT boxes (box IoU ≥ `MATCH_IOU`). **TP** = right box **and** right class. A GT object with no box, or with a box of the wrong class, is a **FN** for its class.
# - **Precision (per class)** = TP / all boxes predicted as that class. "When the model says Glass, how often is it right?"
# - **Recall (per class)** = TP / GT objects of that class. "How many of the Glass objects did we find with the right class?"
# - **F1** = harmonic mean of precision and recall (high only if both are high).
# - **micro** = add up TP/FP/FN of all classes first → dominated by the big class (Plastic).
# - **macro** = compute per class, then take the simple mean → every class counts the same. **This is the main number for imbalanced data.**
# - **weighted F1** = per-class F1 weighted by the number of GT objects (between micro and macro).
# - **Balanced accuracy** = mean of per-class recall over **all** GT objects (a missed object counts as wrong). Same as macro recall.
# - **Accuracy (all GT objects)** = correct objects / all GT objects (missed = wrong). Not balanced.
# - **Accuracy (localised objects)** = the old v1 number: correct class among objects that **were found**. It hides the missed objects, so it looks too good.
# - **Cohen's kappa** = agreement between true and predicted labels (predicted = matched class or "missed"), corrected for chance. 0 = chance, 1 = perfect.
# - **MCC** (Matthews correlation coefficient) = correlation between true and predicted labels; uses the whole confusion table, robust to imbalance. 0 = chance, 1 = perfect.
#
# Kappa and MCC are computed over GT objects, so false boxes on background are not in them (they are in Precision and "FP on background").

# %% [markdown]
# ### Object-level ROC and PR curves (AUC)
# A detector has no single "score per sample", so we build one per **GT object**: for each class *c*, `s_c` = the highest confidence of a class-*c* box whose IoU with the object is ≥ `MATCH_IOU` (0 if there is none). Ultralytics' NMS is per class, so overlapping boxes of different classes are kept → each object gets a 5-score vector.
# Then, for each class, one-vs-rest **ROC-AUC** (sklearn `roc_auc_score`) and **PR-AUC** (average precision). **Macro** = mean over classes.
# Objects that the detector **missed** get score 0 for every class, so low detector recall lowers the AUC (it is not hidden).
# The straight line at the end of each curve = these missed objects (all tied at score 0).
# False boxes on background have no GT object, so they do not enter the ROC; they appear in Precision instead.

# %%
def object_scores(preds, gt=None, ids=None):
    """(N_gt_objects × 5) score matrix and the true class of every GT object."""
    gt = GT if gt is None else gt
    ids = test_ids if ids is None else ids
    S, y = [], []
    for i in ids:
        p, g = preds[i], gt[i]
        s = np.zeros((len(g["cls"]), NC))
        if len(g["cls"]) and len(p["cls"]):
            hit = np.where(box_iou(g["boxes"], p["boxes"]) >= MATCH_IOU, p["scores"][None, :], 0.0)   # G × P
            for c in range(NC):
                k = p["cls"] == c
                if k.any():
                    s[:, c] = hit[:, k].max(1)
        S.append(s); y.append(g["cls"])
    return np.concatenate(S) if S else np.zeros((0, NC)), np.concatenate(y) if y else np.zeros(0, int)


def roc_pr_report(preds_by_name, gt=None, ids=None):
    """preds_by_name: {detector name: preds}. Saves TABLE_E7_auc.csv, fig_E7_roc_curves.png, fig_E7_pr_curves.png."""
    rows, curves = [], {}
    for name, preds in preds_by_name.items():
        S, y = object_scores(preds, gt, ids)
        curves[name] = (S, y)
        for c in range(NC):
            pos = y == c
            ok = 0 < pos.sum() < len(y)
            rows.append([name, CLASSES[c], int(pos.sum()), roc_auc_score(pos, S[:, c]) if ok else np.nan,
                         average_precision_score(pos, S[:, c]) if ok else np.nan, pos.mean() if len(y) else np.nan])
        sub = pd.DataFrame(rows[-NC:])
        rows.append([name, "macro", int(len(y)), sub[3].mean(), sub[4].mean(), np.nan])
    auc = pd.DataFrame(rows, columns=["detector", "class", "GT objects", "ROC-AUC", "PR-AUC (AP)", "PR baseline (prevalence)"])
    auc = auc.set_index(["detector", "class"])
    auc.round(3).to_csv(OUT / "TABLE_E7_auc.csv")
    display(auc.round(3))

    n = len(curves)
    for kind in ("roc", "pr"):
        fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 6.2), squeeze=False)
        for ax, (name, (S, y)) in zip(axes[0], curves.items()):
            for c in range(NC):
                pos = y == c
                if not 0 < pos.sum() < len(y):
                    continue
                a = auc.loc[(name, CLASSES[c])]
                if kind == "roc":
                    fpr, tpr, _ = roc_curve(pos, S[:, c])
                    ax.plot(fpr, tpr, lw=2, color=CLASS_COLORS[c], label=f"{CLASSES[c]}  AUC {a['ROC-AUC']:.3f}")
                else:
                    pr, rc, _ = precision_recall_curve(pos, S[:, c])
                    ax.plot(rc, pr, lw=2, color=CLASS_COLORS[c], label=f"{CLASSES[c]}  AP {a['PR-AUC (AP)']:.3f}")
                    ax.axhline(pos.mean(), color=CLASS_COLORS[c], lw=0.8, ls=":")
            m_ = auc.loc[(name, "macro")]
            if kind == "roc":
                ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
                ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
                ax.set_title(f"{name}\nmacro ROC-AUC {m_['ROC-AUC']:.3f}", loc="left")
            else:
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
                ax.set_title(f"{name}\nmacro AP {m_['PR-AUC (AP)']:.3f}  (dotted = chance level)", loc="left")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal"); ax.grid(axis="x")
            ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, columnspacing=1.2)
        fig.suptitle("Object-level " + ("ROC" if kind == "roc" else "PR") + " curves, one-vs-rest (test)",
                     x=0.01, ha="left", fontsize=11, color=INK2)
        plt.tight_layout(rect=(0, 0, 1, 0.96)); savefig(f"fig_E7_{kind}_curves.png")
    return auc

# %% [markdown]
# ### Per-class confidence thresholds (tuned on **val**)
# One global threshold may be unfair to some classes: the detector's confidence is not calibrated the same way for every class (e.g. it may be less confident on rare classes). Here every class gets its own threshold, chosen to **maximise that class's F1 on the validation split**. The test split is only used for the final report.
# Search: (1) the same threshold for all classes over the grid → best value per class; (2) one round of coordinate search (change one class's threshold, keep the others fixed), because the class-agnostic matching makes classes interact a little.

# %%
def tune_class_thresholds(preds_val, gt_val, ids_val, grid=np.arange(0.05, 0.91, 0.05)):
    """→ (5 per-class thresholds, table). Each class's threshold maximises its own F1 on val."""
    grid = np.round(np.asarray(grid, float), 3)
    f1_of = lambda t: _per_class(_match_all(preds_val, gt_val, ids_val, t)[0])[6]
    F_uni = np.stack([f1_of(t) for t in grid])                 # len(grid) × 5
    thr = grid[F_uni.argmax(0)].copy()
    for c in range(NC):                                          # 1 round of coordinate search
        best = (-1, thr[c])
        for t in grid:
            v = thr.copy(); v[c] = t
            f = f1_of(v)[c]
            if f > best[0] + 1e-12:
                best = (f, t)
        thr[c] = best[1]
    base = _per_class(_match_all(preds_val, gt_val, ids_val, CONF_THRES)[0])
    tuned = _per_class(_match_all(preds_val, gt_val, ids_val, thr)[0])
    table = pd.DataFrame({"val GT objects": base[3].astype(int), "threshold (uniform search)": grid[F_uni.argmax(0)],
                          "threshold (final)": thr, f"val F1 @ {CONF_THRES}": base[6], "val F1 @ tuned": tuned[6],
                          f"val Recall @ {CONF_THRES}": base[5], "val Recall @ tuned": tuned[5]}, index=CLASSES)
    table.index.name = "class"
    table.round(3).to_csv(OUT / "TABLE_E7_class_thresholds.csv")
    display(table.round(3))
    return thr, table

# %% [markdown]
# ### Comparison table + before / after figures

# %%
def compare_detectors(reports, auc=None, fig_suffix=""):
    """reports: {label: detection_report(...)}; auc (optional) = roc_pr_report table → adds macro ROC-AUC / AP by detector."""
    keys = ["F1 (macro)", "Balanced accuracy", "Precision (macro)", "Recall (macro)", "F1 (weighted)", "F1 (micro)",
            "Precision (micro)", "Recall (micro)", "Cohen kappa", "MCC", "Accuracy (all GT objects)",
            "Accuracy (localised objects)", "FP on background"]
    rows = {}
    for lab, r in reports.items():
        row = r["summary"][keys].copy()
        if auc is not None and (r["detector"], "macro") in auc.index:
            row["ROC-AUC (macro)"] = auc.loc[(r["detector"], "macro"), "ROC-AUC"]
            row["PR-AUC (macro)"] = auc.loc[(r["detector"], "macro"), "PR-AUC (AP)"]
        for c in CLASSES:
            row[f"F1 {c}"] = r["per_class"].loc[c, "F1"]
        for c in CLASSES:
            row[f"Recall {c}"] = r["per_class"].loc[c, "Recall"]
        rows[lab] = row
    table = pd.DataFrame(rows).T
    table.index.name = "detector / setting"
    table.round(3).to_csv(OUT / "TABLE_E7_detector_comparison.csv")
    display(table.round(3).T)

    # --- row-normalised confusion matrices side by side (same style as fig_confusion_matrix.png) ---
    labels = CLASSES + ["Background"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    n = len(reports)
    fig, axes = plt.subplots(1, n, figsize=(5.9 * n + 0.6, 6.2), squeeze=False)
    for j, (ax, (lab, r)) in enumerate(zip(axes[0], reports.items())):
        cm_ = r["cm"]; norm = cm_ / np.maximum(cm_.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1); ax.grid(False)
        for a_ in range(NC + 1):
            for b_ in range(NC + 1):
                if a_ == NC and b_ == NC:
                    continue
                ax.text(b_, a_, f"{norm[a_, b_]:.2f}\n({cm_[a_, b_]})", ha="center", va="center", fontsize=8,
                        color="white" if norm[a_, b_] > 0.55 else INK)
        ax.set_xticks(range(NC + 1), labels, rotation=30, ha="right")
        ax.set_yticks(range(NC + 1), labels if j == 0 else [""] * (NC + 1))
        ax.set_xlabel("Predicted")
        if j == 0:
            ax.set_ylabel("True")
        s = r["summary"]
        ax.set_title(f"{lab}\nmacro-F1 {s['F1 (macro)']:.3f} · balanced acc {s['Balanced accuracy']:.3f}", loc="left", fontsize=11)
    fig.suptitle(f"Detector confusion matrices (test, row-normalised, box IoU ≥ {MATCH_IOU})", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95)); savefig(f"fig_E7_confusion_before_after{fig_suffix}.png")

    # --- F1 per class, grouped bars, class support under the name ---
    support = next(iter(reports.values()))["per_class"]["support"]
    x = np.arange(NC); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(9, 1.9 * NC + 1.2 * n), 4.4))
    for j, (lab, r) in enumerate(reports.items()):
        f = r["per_class"]["F1"].values
        b = ax.bar(x - 0.4 + w * (j + 0.5), f, w * 0.92, color=E7_LINE_COLORS[j % len(E7_LINE_COLORS)],
                   label=f"{lab}  (macro-F1 {r['summary']['F1 (macro)']:.3f})")
        for xi, v in zip(x - 0.4 + w * (j + 0.5), f):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [f"{c}\n(n = {int(support[c])})" for c in CLASSES])
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 (right box and right class)")
    ax.set_title("Detector F1 per class on the test split (n = GT objects)", loc="left")
    ax.legend(fontsize=8.5, loc="upper right")
    plt.tight_layout(); savefig(f"fig_E7_f1_by_class{fig_suffix}.png")
    return table

# %% [markdown]
# ### E7 demo: detector v1 — global threshold vs per-class thresholds tuned on val
# For v2, run the same lines with the v2 predictions, e.g.
# `rep_v2 = detection_report(predict_ids(det_v2, test_ids), "v2 · tuned", conf_thr=thr_v2, detector="v2")`, then
# `roc_pr_report({"v1": det_preds, "v2": preds_v2})` and `compare_detectors({..., "v2 · tuned": rep_v2}, auc)`.

# %%
t0_e7 = time.time()
# 1) v1 at the global threshold + sanity check against cell 24
rep_v1 = detection_report(det_preds, f"v1 · conf {CONF_THRES}", conf_thr=CONF_THRES, detector="v1")
chk = rep_v1["per_class"][["TP", "FP", "FN"]].join(thr_table.loc[CLASSES, ["TP", "FP", "FN"]].add_suffix(" (cell 24)"))
print("sanity check vs thr_table (cell 24):", "OK ✓" if (chk[["TP", "FP", "FN"]].values ==
      chk[["TP (cell 24)", "FP (cell 24)", "FN (cell 24)"]].values).all() else "MISMATCH ✗")
display(chk)

# 2) per-class thresholds tuned on val (≈1–2 min of GPU for the val predictions)
gt_val, val_ids = load_gt("val", max_images=MAX_TEST_IMAGES)
t1 = time.time()
preds_val_v1 = predict_ids(det, val_ids, desc="detector v1 on val images")
print(f"val predictions: {len(val_ids)} images in {time.time() - t1:.0f}s")
thr_v1, thr_table_v1 = tune_class_thresholds(preds_val_v1, gt_val, val_ids)
rep_v1_tuned = detection_report(det_preds, "v1 · per-class thr (val)", conf_thr=thr_v1, detector="v1")

# 3) threshold-free: object-level ROC / PR
auc_e7 = roc_pr_report({"v1": det_preds})

# 4) comparison (v2 rows are added here later)
reports_e7 = {f"v1 · conf {CONF_THRES}": rep_v1, "v1 · per-class thr (val)": rep_v1_tuned}
cmp_e7 = compare_detectors(reports_e7, auc_e7)

# 5) findings
a, b = rep_v1["summary"], rep_v1_tuned["summary"]
pa, pb_ = rep_v1["per_class"], rep_v1_tuned["per_class"]
m_auc = auc_e7.loc[("v1", "macro")]
cls_auc = auc_e7.loc["v1"].drop("macro")["ROC-AUC"].dropna()
gain = (pb_["F1"] - pa["F1"]).sort_values(ascending=False)
print(f"\nE7 findings (test split, {int(a['GT objects'])} GT objects, {len(test_ids)} images):")
print(f"• The old headline 'accuracy of localised objects' = {a['Accuracy (localised objects)']:.3f}, but it ignores the "
      f"{int(a['GT objects'] - a[f'Localised (IoU≥{MATCH_IOU:.1f})'])} missed objects; accuracy over all GT objects = "
      f"{a['Accuracy (all GT objects)']:.3f} and balanced accuracy = {a['Balanced accuracy']:.3f}.")
print(f"• At conf {CONF_THRES}: macro-F1 {a['F1 (macro)']:.3f} vs micro-F1 {a['F1 (micro)']:.3f} → the gap is the class imbalance; "
      f"kappa {a['Cohen kappa']:.3f}, MCC {a['MCC']:.3f}.")
print(f"• Weakest classes at conf {CONF_THRES}: " + ", ".join(f"{c} recall {pa.loc[c, 'Recall']:.2f}"
                                                         for c in pa["Recall"].nsmallest(2).index) + ".")
print(f"• Per-class thresholds tuned on val ({', '.join(f'{c} {t:.2f}' for c, t in zip(CLASSES, thr_v1))}): "
      f"macro-F1 {a['F1 (macro)']:.3f} → {b['F1 (macro)']:.3f}, balanced accuracy {a['Balanced accuracy']:.3f} → "
      f"{b['Balanced accuracy']:.3f}, MCC {a['MCC']:.3f} → {b['MCC']:.3f}.")
print(f"• Largest F1 change: {gain.index[0]} {pa.loc[gain.index[0], 'F1']:.3f} → {pb_.loc[gain.index[0], 'F1']:.3f}; "
      f"smallest: {gain.index[-1]} {pa.loc[gain.index[-1], 'F1']:.3f} → {pb_.loc[gain.index[-1], 'F1']:.3f}.")
print(f"• Object-level ROC-AUC (threshold-free): macro {m_auc['ROC-AUC']:.3f}, best {cls_auc.idxmax()} {cls_auc.max():.3f}, "
      f"worst {cls_auc.idxmin()} {cls_auc.min():.3f}; macro PR-AUC {m_auc['PR-AUC (AP)']:.3f}.")
print(f"E7 done in {time.time() - t0_e7:.0f}s")
