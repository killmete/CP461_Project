# Automatic Litter Detection & Instance Segmentation using EfficientSAM

**CP461 Computer Vision · Final project · ทีม อุดมไปด้วยธาตุเหล็ก**

TACO litter photos → **YOLO11s detector** (trained here, 5 material classes) → each box is a prompt for
**EfficientSAM-Ti / -S** (pretrained, zero-shot) → instance masks, class and a count per class.

| | |
|---|---|
| Reference paper | *EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything*, Xiong et al., CVPR 2024 |
| Dataset | [TACO](https://github.com/pedropro/TACO): 1,500 images, 4,784 objects, 60 categories regrouped into Plastic, Paper & Cardboard, Metal, Glass, Other Waste |
| Split | 1,050 / 225 / 225 images (train / val / test), definitions in `data/coco5_*.json` |

## Main results (test split, 225 images, 710 objects)

| | v1 (progress) | v2 (final: 960 px + repeat-factor sampling) |
|---|---|---|
| Box mAP50 | 0.246 | **0.270** |
| Recall | 0.264 | **0.357** |
| Macro ROC-AUC (object level) | 0.674 | **0.685** |
| MCC | 0.248 | **0.267** |

| Segmentation (detector v2) | Mask AP | ms / image |
|---|---|---|
| YOLO → EfficientSAM-S | **0.196** | 324 |
| YOLO → EfficientSAM-Ti | 0.190 | 167 |
| YOLO11s-seg (one-stage baseline) | 0.142 | 22 |

EfficientSAM with ground-truth boxes reaches mIoU 0.787 (S) / 0.781 (Ti) on TACO. The paper reports 0.769 / 0.757 on COCO.
The detector is the bottleneck: it finds about a third of the litter, while the masks are good once an object is found.
All tables and figures are in [`final/colab_outputs`](final/colab_outputs) (`results_summary.md` is the overview).

## Repository layout

| Path | What it is |
|---|---|
| [`Litter_EfficientSAM_TACO_FINAL.ipynb`](Litter_EfficientSAM_TACO_FINAL.ipynb) | **Final notebook** (Google Colab, T4). Run all ≈ 1.5–2 h. Experiments E1–E7, benchmark, Gradio demo |
| `Litter_EfficientSAM_TACO.ipynb` | Progress-report notebook (v1) |
| `Automatic Litter Detection and Instance Segmentation Using EfficientSAM.pdf` | Progress-report slides |
| [`final/deck/`](final/deck) | Final slides (`CP461_Final_Litter_EfficientSAM.pptx`), 5-minute script, slide builder |
| [`final/colab_outputs/`](final/colab_outputs) | Outputs of the final Colab run: CSV tables, figures, trained weights |
| `outputs/` | Outputs of the progress-report run |
| [`final/web/`](final/web) | Browser-only web app (ONNX Runtime Web): no server, deployable as a static site |
| `final/app/app.py` | Local Gradio app (`python final/app/app.py`) |
| `final/hf_space/` | Gradio version packaged for Hugging Face Spaces (copy `final/colab_outputs/weights/yolo_det_best.pt` next to `app.py`) |
| `final/build_notebook.py`, `final/sections/` | Builds the final notebook from the v1 notebook + new sections (E1b, E6, E7, benchmark) |
| `final/benchmark/` | Published numbers we compare against, with citations |
| `final/pilot_detector.py`, `final/pilot_results.md` | Local pilot that chose the detector v2 recipe |

## Run it

**Notebook:** open the final notebook in Colab ▸ Runtime ▸ T4 GPU ▸ Run all. It downloads TACO, trains the detectors, runs every
experiment and saves `outputs.zip`.

**Web app (browser only):** serve `final/web` with any static host (Hugging Face *Static* Space, GitHub Pages) or locally:

```bash
python -m http.server 8765 --directory final/web
```

**Gradio app (Python):**

```bash
pip install ultralytics gradio opencv-python torch torchvision
python final/app/app.py
```

## Team

| Member | Role |
|---|---|
| นายกฤชตฤณ บุญสู่ | Researcher |
| นายณฐพล สวนใหญ่ | Tester |
| นายอานัส มีสัตย์ | Developer |
| นายโกวิทย์ เพชรนิล | Researcher / Developer |
| นายธนภูมิ ด้วงมาก | Tester |
