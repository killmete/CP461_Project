---
title: Litter Detection with EfficientSAM
emoji: 🗑️
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: YOLO11 + EfficientSAM litter detection, runs in the browser
---

# Automatic Litter Detection & Instance Segmentation using EfficientSAM

CP461 Computer Vision final project · ทีม อุดมไปด้วยธาตุเหล็ก

Everything runs in the visitor's browser with ONNX Runtime Web; no server and no image upload.
YOLO11s detector (trained on TACO, 5 material classes) → each box is a prompt for EfficientSAM-Ti → masks + count per class.

Models: `models/detector.onnx` (our YOLO11s v2, exported at 960 px) and the EfficientSAM-Ti encoder/decoder ONNX files
from [yformer/EfficientSAM](https://github.com/yformer/EfficientSAM) (Apache-2.0).
