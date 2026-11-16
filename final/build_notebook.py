"""Builds the final Colab notebook from the v1 notebook + section files.
   python final/build_notebook.py   →   Litter_EfficientSAM_TACO_FINAL.ipynb (project root)
Every patch asserts that its anchor text exists, so a silent no-op is impossible."""
import copy, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEC = ROOT / "final/sections"
v1 = json.loads((ROOT / "Litter_EfficientSAM_TACO.ipynb").read_text(encoding="utf-8"))
cells = v1["cells"]


def src(c):
    return "".join(c["source"])


def mk(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def section(path, stop_at=None):
    """'# %% [markdown]' / '# %%' separated .py → cells (markdown lines lose their '# ' prefix)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if stop_at is not None:
        lines = lines[:stop_at]
    out, kind, buf = [], None, []
    def flush():
        body = "\n".join(buf).strip("\n")
        if kind and body:
            if kind == "markdown":
                body = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
            out.append(mk(kind, body))
    for l in lines:
        if l.startswith("# %%"):
            flush(); kind, buf = ("markdown" if "[markdown]" in l else "code"), []
        else:
            buf.append(l)
    flush()
    return out


def patch(i, old, new, count=1):
    s = src(cells[i])
    assert s.count(old) >= 1, f"anchor not found in cell {i}: {old[:60]!r}"
    cells[i]["source"] = s.replace(old, new, count if count else s.count(old)).splitlines(keepends=True)


def strip_outputs(c):
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
    return c


# ---------------------------------------------------------------- patches on v1 cells (indices = v1 notebook)
INTRO = r"""
# 🗑️ Automatic Litter Detection & Instance Segmentation using EfficientSAM — FINAL
**ทีม อุดมไปด้วยธาตุเหล็ก · CP461 Computer Vision · Final project**

**Pipeline:** TACO photos → **YOLO11 detector (trained here, 5 classes)** → bounding boxes as prompts → **EfficientSAM-Ti / -S (pretrained, zero-shot)** → instance masks + class + count

| | |
|---|---|
| Paper | *EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything*, Xiong et al., CVPR 2024 |
| Code | [yformer/EfficientSAM](https://github.com/yformer/EfficientSAM) · [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| Data | [TACO](https://github.com/pedropro/TACO): 1,500 images, 4,784 objects, 60 categories regrouped into 5 classes |

### ▶️ How to run (Colab)
1. **Runtime ▸ Change runtime type ▸ T4 GPU**.
2. Keep `DRIVE_FOLDER` = the **same Drive folder as the progress run** (default `CP461Project`). Images, the v1 detector and the YOLO-seg baseline are reused from it, nothing is downloaded or retrained twice.
3. **Runtime ▸ Run all** and keep the tab open. Time on a T4 (with the progress run in Drive): **≈ 1.5–2 h**: detector v2 training ≈ 60–75 min, evaluation + E6/E7 ≈ 20–25 min. From an empty Drive folder add ≈ 1.5 h (download + v1 + YOLO-seg).
4. If Colab disconnects: **Run all** again — training resumes from the last checkpoint.
5. All figures and tables are saved in `outputs/` in your Drive folder and downloaded as **`outputs.zip`** at the end.

### What changed since the progress report (answers to the instructor's comments)
| Comment | Answer in this notebook |
|---|---|
| "Confusion matrix still not good" | **E1b** detector v2: 960 px + repeat-factor sampling for rare classes; v1 vs v2 confusion matrices (5.2) |
| "Class imbalance → accuracy alone is not enough" | **E7** macro-F1, balanced accuracy, Cohen's κ, MCC, per-class P/R/F1, object-level ROC-AUC / PR-AUC, per-class thresholds |
| "What does *studying the effect of the prompt* mean? Report it" | **E6** prompt study: box vs point vs box+points vs noisy boxes, by class, size and object shape |
| "What is the benchmark? Higher or lower?" | **12.5** benchmark table vs TACO paper, other TACO detectors and the EfficientSAM paper |

### Experiments
| # | Experiment | Question it answers | Metrics |
|---|---|---|---|
| E1 / E1b | YOLO detector v1 (progress) vs v2 (final) | How well do we find and classify litter? Did v2 fix the confusion matrix? | P, R, F1, mAP50, mAP50-95, confusion matrix |
| E2 | **GT boxes** → EfficientSAM-Ti vs -S | How good are masks with a perfect box? | IoU, Dice (per class, size) |
| E3 | **YOLO boxes** → EfficientSAM-Ti vs -S | Full pipeline quality | Mask AP, AP50, AP75, IoU, Dice |
| E4 | YOLO-seg baseline (one-stage) | Is Detector + EfficientSAM better than an end-to-end segmenter? | Same as E3 |
| E5 | Speed | What does each option cost? | ms / image, parameters |
| E6 | **Prompt study** | Which prompt gives the best masks, and why do thin objects fail? | mIoU, IoU≥0.5/0.75 rate by class, size, fill ratio, box noise |
| E7 | **Imbalance-aware evaluation** | Is the detector good on *every* class, not only Plastic? | macro/weighted F1, balanced acc., κ, MCC, ROC-AUC, PR-AUC |
"""
cells[0] = mk("markdown", INTRO)

patch(2, 'LAUNCH_DEMO        = True             # Gradio web demo (prototype) at the end',
      'LAUNCH_DEMO        = True             # Gradio web demo (prototype) at the end\n\n'
      '# ---- final: detector v2 (answers "the confusion matrix is still not good") ----\n'
      'TRAIN_DET_V2       = True             # False = keep the progress-report detector (v1) for everything\n'
      'DET_V2_MODEL       = "yolo11s.pt"     # same size as v1, so the gain comes from the recipe, not a bigger model\n'
      'DET_V2_IMG_SIZE    = 960              # larger input for small litter (v1 used 640)\n'
      'DET_V2_RFS         = True             # repeat-factor sampling: show images with rare classes more often\n'
      'RFS_T              = 0.3              # RFS threshold t (classes in < 30% of training images get repeated)\n'
      'DET_V2_EPOCHS      = 80\n'
      'DET_V2_BATCH       = 16               # T4 16 GB is fine at 960 px; use 8 if you get CUDA out-of-memory')
patch(17, "def train_yolo(weights, data_yaml, name, epochs):", "def train_yolo(weights, data_yaml, name, epochs, imgsz=None, batch=None):")
patch(17, "imgsz=IMG_SIZE, batch=BATCH,", "imgsz=imgsz or IMG_SIZE, batch=batch or BATCH,")
for i in (24, 32, 43):                                   # detector inference at the detector's own size (v2 = 960)
    s = src(cells[i]); n = 0
    lines = []
    for l in s.splitlines(keepends=True):
        if "det.predict(" in l and "imgsz=IMG_SIZE" in l:
            l = l.replace("imgsz=IMG_SIZE", "imgsz=DET_IMGSZ"); n += 1
        lines.append(l)
    assert n == 1, f"cell {i}: expected 1 det.predict line, got {n}"
    cells[i]["source"] = lines
patch(24, 'ax.set_title(f"Detector confusion matrix (test, conf ≥ {CONF_THRES}, row-normalised)", loc="left")',
      'ax.set_title(f"Detector {\'v2\' if DET_V2_BEST is not None else \'v1\'} confusion matrix (test, conf ≥ {CONF_THRES}, row-normalised)", loc="left")')
patch(41, 'detector `{DET_MODEL}` at {IMG_SIZE}px, {EPOCHS} epochs max',
      'detector `{DET_V2_MODEL if DET_V2_BEST is not None else DET_MODEL}` at {DET_IMGSZ}px ({"v2: RFS + 960 px" if DET_V2_BEST is not None else "v1"})')
patch(41, '(OUT / "results_summary.md").write_text("\\n".join(lines))',
      '# final-only tables (E1 v1 vs v2, E6 prompts, E7 imbalance metrics, benchmark)\n'
      'for title, f in [("E1 detector v1 vs v2 (Ultralytics, test)", "TABLE_E1_v1_vs_v2.csv"),\n'
      '                 ("E7 imbalance-aware detector comparison", "TABLE_E7_detector_comparison.csv"),\n'
      '                 ("E7 object-level ROC-AUC / PR-AUC", "TABLE_E7_auc.csv"),\n'
      '                 ("E6 prompt study (overall)", "TABLE_E6_prompt_overall.csv"),\n'
      '                 ("E6 prompt study by fill ratio (thin objects)", "TABLE_E6_prompt_by_fill.csv"),\n'
      '                 ("Benchmark comparison", "TABLE_benchmark_comparison.csv")]:\n'
      '    if (OUT / f).exists():\n'
      '        t = pd.read_csv(OUT / f, header=[0, 1], index_col=0) if f == "TABLE_E1_v1_vs_v2.csv" else pd.read_csv(OUT / f)\n'
      '        if isinstance(t.columns, pd.MultiIndex):\n'
      '            t.columns = [f"{m} {d}" for m, d in t.columns]\n'
      '        lines += [f"## {title}", "", md_table(t), ""]\n'
      '(OUT / "results_summary.md").write_text("\\n".join(lines))')
patch(41, 'shutil.copy(DET_BEST, OUT / "weights/yolo_det_best.pt")',
      'shutil.copy(DET_BEST, OUT / "weights/yolo_det_best.pt")          # the pipeline detector (v2)\n'
      'shutil.copy(DET_V1_BEST, OUT / "weights/yolo_det_v1.pt")')
patch(43, '                    agnostic_nms=True, verbose=False)[0]\n'
          '    boxes, cls, sc = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()\n',
      '                    agnostic_nms=True, iou=0.5, verbose=False)[0]\n'
      '    boxes, cls, sc = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()\n'
      '    keep = [i for i in np.argsort(-sc)]                # demo only: drop a box that lies ≥ 80% inside a higher-scoring box\n'
      '    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]); kept = []\n'
      '    for i in keep:\n'
      '        iw = np.clip(np.minimum(boxes[i, 2], boxes[kept, 2]) - np.maximum(boxes[i, 0], boxes[kept, 0]), 0, None)\n'
      '        ih = np.clip(np.minimum(boxes[i, 3], boxes[kept, 3]) - np.maximum(boxes[i, 1], boxes[kept, 1]), 0, None)\n'
      '        if not len(kept) or not (iw * ih >= 0.8 * np.minimum(area[i], area[kept])).any():\n'
      '            kept.append(i)\n'
      '    kept = sorted(kept); boxes, cls, sc = boxes[kept], cls[kept], sc[kept]\n')
patch(43, 'title="Automatic Litter Detection & Segmentation · YOLO + EfficientSAM"',
      'title="Automatic Litter Detection & Segmentation · YOLO + EfficientSAM (final)"')

# ---------------------------------------------------------------- assemble
def retitle(cs, old, new):
    s0 = src(cs[0]); assert s0.startswith(old), f"heading {old!r} not found: {s0[:60]!r}"
    cs[0]["source"] = s0.replace(old, new, 1).splitlines(keepends=True)
    return cs


e7_lines = (SEC / "e7_metrics.py").read_text(encoding="utf-8").splitlines()
demo_at = next(k for k, l in enumerate(e7_lines) if l.startswith("# %%") and "E7 demo" in e7_lines[k + 1])
INSERT_AFTER = {
    21: section(SEC / "e1b_detector_v2.py"),
    24: retitle(section(SEC / "e7_metrics.py", stop_at=demo_at), "## E7 ·", "### 5.1 · E7 ·") + section(SEC / "e7_run_final.py"),
    28: retitle(section(SEC / "e6_prompt_study.py"), "## 13 · E6", "## 7.1 · E6"),
    39: section(SEC / "bench_section.py"),
}
REPLACE = {22: [], 23: section(SEC / "e1_eval_v1_v2.py")}  # v1's single-detector E1 → v1 vs v2

final = []
for i, c in enumerate(cells):
    if i in REPLACE:
        final += REPLACE[i]
    else:
        final.append(strip_outputs(copy.deepcopy(c)))
    final += INSERT_AFTER.get(i, [])
nb = copy.deepcopy(v1); nb["cells"] = final
nb.setdefault("metadata", {})["accelerator"] = "GPU"
out = ROOT / "Litter_EfficientSAM_TACO_FINAL.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out} · {len(final)} cells ({sum(c['cell_type'] == 'code' for c in final)} code)")
for k, c in enumerate(final):
    first = src(c).strip().splitlines()[0] if src(c).strip() else ""
    print(f"[{k:2d}] {c['cell_type'][:4]} {first[:90]}")
