"""Prototype web app: litter photo → YOLO11s detector (v2) → box prompts → EfficientSAM masks → count per class.

Run (from the project folder):
    pip install ultralytics gradio opencv-python torch torchvision
    python final/app/app.py                      # opens http://127.0.0.1:7860
Options: --weights <detector .pt>  --esam <folder that contains EfficientSAM/>  --share (public Gradio link)
The EfficientSAM code + weights are cloned from github.com/yformer/EfficientSAM the first time if missing.
"""
import argparse, subprocess, sys, zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CLASSES = ["Plastic", "Paper & Cardboard", "Metal", "Glass", "Other Waste"]
CLASSES_TH = ["พลาสติก", "กระดาษ", "โลหะ", "แก้ว", "อื่น ๆ"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DET_IMGSZ, MAX_SIDE = 960, 1280
DEFAULT_SEG = "EfficientSAM-S" if DEVICE == "cuda" else "EfficientSAM-Ti"   # Ti is ~2x faster on a free CPU

ap = argparse.ArgumentParser()
LOCAL_W = HERE / "yolo_det_best.pt"            # Hugging Face Space layout: weights next to app.py
ap.add_argument("--weights", default=str(LOCAL_W if LOCAL_W.exists() else ROOT / "final/colab_outputs/weights/yolo_det_best.pt"))
ap.add_argument("--esam", default=str(ROOT / "final/_localwork/repos" if (ROOT / "final/_localwork/repos").exists() else HERE / "repos"))
ap.add_argument("--share", action="store_true")
ap.add_argument("--port", type=int, default=7860)
args = ap.parse_args()

# ---------------------------------------------------------------- models
repo = Path(args.esam) / "EfficientSAM"
if not repo.exists():
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/yformer/EfficientSAM", str(repo)], check=True)
sys.path.insert(0, str(repo))
from efficient_sam.efficient_sam import build_efficient_sam  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def load_esam(variant):
    w = repo / "weights"
    if variant == "s":
        if not (w / "efficient_sam_vits.pt").exists():
            with zipfile.ZipFile(w / "efficient_sam_vits.pt.zip") as z:
                z.extractall(w)
        dim, heads, ckpt = 384, 6, w / "efficient_sam_vits.pt"
    else:
        dim, heads, ckpt = 192, 3, w / "efficient_sam_vitt.pt"
    m = build_efficient_sam(encoder_patch_embed_dim=dim, encoder_num_heads=heads, checkpoint=None)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    return m.eval().to(DEVICE)


det = YOLO(args.weights)
ESAM = {"EfficientSAM-Ti": load_esam("ti"), "EfficientSAM-S": load_esam("s")}


@torch.inference_mode()
def segment(model, img, boxes, chunk=32):
    H, W = img.shape[:2]
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
    emb = model.get_image_embeddings(x)
    out = []
    for k in range(0, len(boxes), chunk):
        b = torch.as_tensor(boxes[k:k + chunk], dtype=torch.float32, device=DEVICE)
        n = b.shape[0]
        lbl = torch.tensor([2, 3], device=DEVICE).view(1, 1, 2).expand(1, n, 2)        # 2/3 = box corners
        logits, iou = model.predict_masks(emb, b.view(1, n, 2, 2), lbl, multimask_output=True,
                                          input_h=H, input_w=W, output_h=-1, output_w=-1)
        low = logits[0, torch.arange(n, device=DEVICE), iou[0].argmax(-1)]
        out.append((F.interpolate(low[:, None], (H, W), mode="bicubic")[:, 0] >= 0).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, H, W), bool)


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def draw(img, boxes, cls, scores, masks):
    vis = img.copy()
    fs, th = 0.5 * max(1.0, img.shape[1] / 700), max(1, round(img.shape[1] / 600))
    for b, c, s, mk in sorted(zip(boxes, cls, scores, masks), key=lambda t: -t[3].sum()):
        col = np.array(hex2rgb(COLORS[c]), np.float32)
        vis[mk] = (0.5 * vis[mk] + 0.5 * col).astype(np.uint8)
        cnts, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, tuple(int(v) for v in col), th + 1)
        b = b.astype(int)
        cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), tuple(int(v) for v in col), th)
        lab = f"{CLASSES[c]} {s:.2f}"
        (tw, tht), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.rectangle(vis, (b[0], max(0, b[1] - tht - 10)), (b[0] + tw + 8, max(tht + 10, b[1])), tuple(int(v) for v in col), -1)
        cv2.putText(vis, lab, (b[0] + 4, max(tht + 4, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return vis


def drop_nested(boxes, scores, thr=0.8):
    """NMS keeps a small box inside a bigger one (IoU is low), e.g. a can and its label → 2 detections.
    Drop a box when ≥ thr of its area lies inside a higher-scoring box."""
    order, keep = np.argsort(-scores), []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in order:
        ok = True
        for j in keep:
            iw = max(0.0, min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0]))
            ih = max(0.0, min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1]))
            if iw * ih >= thr * min(area[i], area[j]):
                ok = False
                break
        if ok:
            keep.append(i)
    return np.array(sorted(keep), dtype=int)


def predict(img, conf, segmenter):
    if img is None:
        return None, []
    img = np.ascontiguousarray(np.asarray(img)[..., :3].astype(np.uint8))
    s = min(1.0, MAX_SIDE / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    r = det.predict(np.ascontiguousarray(img[..., ::-1]), conf=conf, iou=0.5, imgsz=DET_IMGSZ, agnostic_nms=True, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    sc = r.boxes.conf.cpu().numpy()
    keep = drop_nested(boxes, sc)
    boxes, cls, sc = boxes[keep], cls[keep], sc[keep]
    masks = segment(ESAM[segmenter], img, boxes) if len(boxes) else np.zeros((0, *img.shape[:2]), bool)
    n = Counter(cls.tolist())
    table = [[CLASSES[k], CLASSES_TH[k], n.get(k, 0)] for k in range(5)] + [["Total", "รวม", len(cls)]]
    return draw(img, boxes, cls, sc, masks), table


if __name__ == "__main__":
    import gradio as gr
    ex_dir = HERE / "examples"
    examples = [[str(p), 0.25, DEFAULT_SEG] for p in sorted(ex_dir.glob("*.jpg"))] if ex_dir.exists() else None
    with gr.Blocks(title="Litter Detection · YOLO + EfficientSAM") as demo:
        gr.Markdown("## 🗑️ Automatic Litter Detection & Segmentation · YOLO11 + EfficientSAM\n"
                    "ทีม อุดมไปด้วยธาตุเหล็ก · CP461 — อัปโหลดภาพขยะ ระบบจะตรวจจับ 5 ประเภท ทำ mask ทีละชิ้น และนับจำนวน")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="ภาพขยะ (Litter photo)")
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Detection confidence")
                seg = gr.Radio(list(ESAM), value=DEFAULT_SEG, label="Segmenter")
                btn = gr.Button("Detect & segment", variant="primary")
            with gr.Column():
                out = gr.Image(label="Boxes + EfficientSAM masks")
                cnt = gr.Dataframe(headers=["Class", "ประเภท", "Count"], label="จำนวนขยะ (Litter count)")
        btn.click(predict, [inp, conf, seg], [out, cnt])
        if examples:
            gr.Examples(examples, [inp, conf, seg], [out, cnt], fn=predict)
    demo.launch(share=args.share, server_port=args.port)
