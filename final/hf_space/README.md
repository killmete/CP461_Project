---
title: Litter Detection with EfficientSAM
emoji: 🗑️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: YOLO11 + EfficientSAM litter detection, masks and counts
---

# Automatic Litter Detection & Instance Segmentation using EfficientSAM

CP461 Computer Vision final project · ทีม อุดมไปด้วยธาตุเหล็ก

Upload a litter photo: a YOLO11s detector (trained on TACO, 5 material classes) finds each piece of litter,
its box is used as a prompt for EfficientSAM (pretrained, zero-shot), and the app shows the masks and a count per class.

The EfficientSAM code and weights are downloaded from [yformer/EfficientSAM](https://github.com/yformer/EfficientSAM) on first start.
Runs on the free CPU hardware (≈ 2–6 s per image with EfficientSAM-Ti).
