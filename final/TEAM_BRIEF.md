# CP461 final — team brief (shared by every role)

Project: *Automatic Litter Detection and Instance Segmentation Using EfficientSAM* (team อุดมไปด้วยธาตุเหล็ก).
TACO 1,500 images → 5 classes (Plastic, Paper & Cardboard, Metal, Glass, Other Waste). Split 1,050 / 225 / 225.
Pipeline: YOLO11 detector → boxes → EfficientSAM-Ti / -S (pretrained, zero-shot) → masks + count. Baseline: YOLO11s-seg.

## Instructor feedback on the progress report (must all be answered in the final)
1. "ศึกษาผลของ prompt คืออะไร" + "objective 3 says *study the effect of the prompt*, so report the results of that study" → a real prompt ablation (E6).
2. "benchmark of this task? is ours higher or lower?" → published benchmark comparison.
3. "confusion matrix still not good, keep improving" → better detector (E1b), before/after confusion matrices.
4. "class imbalance, so accuracy alone is not enough" → imbalance-aware metrics (E7).

Final rubric also needs: comparison tables (Acc, F1, P, R, AUC…), confusion matrix + ROC curve, correct/incorrect examples, prototype, conclusions (does the model answer the problem, better than the reference paper?, strengths/weaknesses, future work).

## Hard constraint: the final notebook runs on **Google Colab (T4 GPU, 16 GB)**
- Code must be Colab-compatible: no Windows paths, no local-only packages beyond what the notebook `%pip install`s
  (ultralytics==8.4.175, pycocotools, tqdm; sklearn/pandas/matplotlib/cv2 are preinstalled on Colab).
- New GPU work must stay small: inference-only sections should run in **≤ 15 min on a T4** for the full test split.
- Everything is saved under `OUT` (tables, CSV) and `FIG` (PNG via `savefig(name)`), so it lands in `outputs.zip`.

## How the notebook is structured (reference: `final/notebook_v1_code.py`, cell numbers = v1 notebook)
Globals available after **cell 24** (detector predictions):
`CLASSES, CLASS_COLORS, METHOD_COLORS, INK, INK2, MUTED, OUT, FIG, savefig(name), hex2rgb, DEVICE, YOLO_DEVICE,
CONF_THRES (0.25), MATCH_IOU (0.5), IMG_SIZE, SEED, images_all, load_rgb(img_id), overlay(img, mask, color),
mask_utils, COCO, COCOeval, test_coco, test_ids, GT[img_id] = dict(anns, boxes (N×4 xyxy px), cls (N,), rles, area),
det (YOLO model), det_preds[img_id] = dict(boxes, scores, cls) at conf=0.001, box_iou(a,b), match(pb, ps, gb, thr) → (pairs, unmatched_pred, unmatched_gt),
cm (6×6 confusion matrix, last row/col = background), per_image, thr_table, cls_acc, stats, splits, COCO_JSON, FAST_IMG`.

Globals available after **cell 26** (EfficientSAM): `ESAM = {"ti": model, "s": model}, SAM_NAME, SAM_VARIANTS, esam_embed(model, img) → emb,
esam_decode(model, emb, boxes, H, W) → N×H×W bool (box prompt = 2 points with labels 2/3, best of 3 masks by predicted IoU),
to_rle(mask), rle_iou(rle_a, rle_b), sync(), free_gpu()`.
EfficientSAM point labels: 1 = positive point, 0 = negative point, 2 = box top-left, 3 = box bottom-right, -1 = padding.
`model.predict_masks(emb, pts[B,N,K,2], labels[B,N,K], multimask_output, input_h, input_w, output_h=-1, output_w=-1)` → (logits[B,N,3,256,256], iou[B,N,3]).

Globals after **cell 28** (E2/E3): `e2` (per GT object, GT box prompt), `e3` (matched YOLO-box prompts), `size_bin(area)`, `rles_e2`.

## Local smoke testing (Windows box, RTX 3060 Ti 8 GB — shared GPU, keep batch sizes small)
```
cd C:\Users\akari\Documents\CP461_Project
final\.venv\Scripts\python.exe final\run_local.py --upto 28 --set MAX_TEST_IMAGES=20 --append final\sections\<your_file>.py
```
- Runs in sandbox `final/_localwork/litter_esam` (outputs → `final/_localwork/litter_esam/outputs`). Never write to the project's own `data/`, `runs/`, `outputs/`.
- `MAX_TEST_IMAGES=20` for fast iteration; run the full test split once at the end if time allows (report timings).
- Section files are plain Python executed in the notebook namespace. Start each logical cell with a line `# %% [markdown]` / `# %%` so the lead can split it into notebook cells. Markdown cells: lines prefixed with `# `.
- Do not edit `run_local.py`, the v1 notebook, or another role's section file.

## Style
Match the existing figures: `plt.rcParams` already set; class colours `CLASS_COLORS`; titles `loc="left"`; save with `savefig("fig_E6_....png")`;
tables `.round(3).to_csv(OUT / "TABLE_E6_....csv")`. English in code/figures (slides are translated later).
