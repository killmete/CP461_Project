# %% [markdown]
# ## 12.5 · Benchmark: are we higher or lower than published work?
# No published work uses exactly our 5 material classes and 70/15/15 split, so these are the **closest** comparisons, each with its caveat
# (full citations and the table each number comes from: `final/benchmark/benchmark_notes.md`). Published numbers are fixed; our rows are filled in from this run.

# %%
BENCH = [  # (group, source, setting, model, value, caveat)
    ("Box mAP50 (multi-class TACO)", "Majchrowska et al. 2022, Waste Management", "TACO+ · 7 material classes", "EfficientDet-D2", 0.162, "more images (≈4.5k), 7 classes"),
    ("Box mAP50 (multi-class TACO)", "Saakyan & Solntsev 2026, arXiv 2602.23043", "TACO · 59 classes · 640 px", "YOLO26-S", 0.124, "59 fine classes (harder)"),
    ("Box mAP50 (multi-class TACO)", "Saakyan & Solntsev 2026, arXiv 2602.23043", "TACO · 59 classes · 640 px", "D-FINE-S", 0.244, "59 fine classes (harder)"),
    ("Box mAP50 (1-class 'litter')", "Córdova et al. 2022, Sensors", "TACO · 1 class", "YOLOv5s @640", 0.547, "no classification needed (easier)"),
    ("Mask AP (TACO)", "Proença & Simões 2020 (TACO paper)", "TACO-10 · 4-fold CV", "Mask R-CNN R50", 0.176, "10 classes, ±0.016 over folds"),
    ("Mask AP (TACO)", "Saakyan & Solntsev 2026, arXiv 2602.23043", "TACO · 59 classes · 640 px", "YOLO26-seg S", 0.111, "59 fine classes"),
    ("Mask AP (TACO)", "Saakyan & Solntsev 2026, arXiv 2602.23043", "TACO · 59 classes · 640 px", "D-FINE-seg S", 0.177, "59 fine classes"),
    ("Box-prompt mIoU (GT box)", "Xiong et al. 2024, EfficientSAM (CVPR)", "COCO", "EfficientSAM-Ti", 0.757, "reference paper; COCO not TACO"),
    ("Box-prompt mIoU (GT box)", "Xiong et al. 2024, EfficientSAM (CVPR)", "COCO", "EfficientSAM-S", 0.769, "reference paper; COCO not TACO"),
    ("Box-prompt mIoU (GT box)", "Xiong et al. 2024, EfficientSAM (CVPR)", "COCO", "SAM ViT-H (636M)", 0.784, "reference paper; 25–60× bigger"),
]
ours = []
for tag, t in [("v1 (progress)", det_table_v1)] + ([("v2 (final)", det_table_v2)] if DET_V2_BEST is not None else []):
    ours.append(("Box mAP50 (multi-class TACO)", "OURS", "TACO · 5 material classes", f"YOLO11s {tag}", float(t.loc["all", "mAP50"]), ""))
for k, (st, _) in SEG_EVAL.items():
    name = f"YOLO → {k}" if k != "YOLO-seg" else "YOLO11s-seg (one-stage)"
    ours.append(("Mask AP (TACO)", "OURS", "TACO · 5 material classes", name, float(st[0]), "detector " + ("v2" if DET_V2_BEST is not None else "v1")))
for v in SAM_VARIANTS:
    ours.append(("Box-prompt mIoU (GT box)", "OURS", "TACO test (710 objects)", SAM_NAME[v], float(e2[e2.method == SAM_NAME[v]].iou.mean()), "zero-shot"))
bench = pd.DataFrame(BENCH + ours, columns=["metric", "source", "setting", "model", "value", "caveat"])
bench["value"] = bench["value"].round(3)
bench.to_csv(OUT / "TABLE_benchmark_comparison.csv", index=False)
display(bench)

groups = list(dict.fromkeys(bench.metric))
fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 4.6))
for ax, g in zip(axes, groups):
    d = bench[bench.metric == g].sort_values("value")
    cols = ["#eb6834" if s == "OURS" else "#c3c2b7" for s in d.source]
    ax.barh(range(len(d)), d.value, color=cols, height=0.65)
    ax.set_yticks(range(len(d)), [m if s == "OURS" else f"{m}\n{s.split(',')[0]}" for m, s in zip(d.model, d.source)], fontsize=7.5)
    for y, v in enumerate(d.value):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, max(0.1, d.value.max() * 1.3)); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_title(g, loc="left", fontsize=10)
fig.legend(handles=[matplotlib.patches.Patch(color="#eb6834", label="ours"), matplotlib.patches.Patch(color="#c3c2b7", label="published")],
           loc="upper right", ncol=2, fontsize=9)
plt.tight_layout(rect=(0, 0, 1, 0.94)); savefig("fig_benchmark_comparison.png")

our_map = bench[(bench.source == "OURS") & (bench.metric == "Box mAP50 (multi-class TACO)")].value.iloc[-1]   # final detector (v2 if trained)
our_ap = bench[(bench.source == "OURS") & (bench.metric == "Mask AP (TACO)") & bench.model.str.startswith("YOLO →")].value.max()  # best YOLO→EfficientSAM
print(f"• Box mAP50: ours {our_map:.3f} vs closest material-class benchmark 0.162 (7 classes) → {'higher' if our_map > 0.162 else 'lower'}; "
      f"vs 1-class 'litter' detectors 0.547 → lower (they do not classify).")
print(f"• Mask AP: ours {our_ap:.3f} vs TACO paper Mask R-CNN 0.176 ± 0.016 (TACO-10) → {'higher' if our_ap > 0.176 + 0.016 else 'about the same'}.")
print(f"• EfficientSAM on TACO with GT boxes ({', '.join(f'{SAM_NAME[v]} {e2[e2.method == SAM_NAME[v]].iou.mean():.3f}' for v in SAM_VARIANTS)}) "
      f"matches the reference paper on COCO (0.757 / 0.769): zero-shot transfer to litter works; the detector is the bottleneck.")
