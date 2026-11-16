# Detector v2 pilots (local, RTX 8 GB, yolo11s @ 960 px, 80 epochs, batch 8, seed 42), val split

| Pilot | Recipe | best val mAP50 | best val mAP50-95 | AP50 Plastic / Paper / Metal / Glass / Other |
|---|---|---|---|---|
| v1 (progress) | 640 px | 0.241 | – | – |
| A | 960 px | 0.295 | 0.213 | 0.45 / 0.35 / 0.37 / 0.14 / 0.16 |
| B | 960 px + RFS (t=0.3) | **0.302** | **0.221** | 0.46 / 0.32 / 0.29 / **0.31** / 0.13 |

Decision: B (DET_V2_RFS=True). RFS more than doubles Glass AP50 (rarest class), with a small cost on Metal.
Recall per class A: 0.51/0.38/0.42/0.21/0.24 · B: 0.47/0.34/0.31/0.31/0.21.
