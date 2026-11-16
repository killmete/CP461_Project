# Results summary (auto-generated)

- Test split: 225 images, 710 objects; detector `yolo11s.pt` at 640px, 80 epochs max
- Detector (test): P=0.393  R=0.264  F1=0.316  mAP50=0.246  mAP50-95=0.189
- Classification accuracy of localised objects (conf ≥ 0.25): 0.675
- E2 GT-box mIoU: EfficientSAM-Ti 0.781, EfficientSAM-S 0.787
- E3 pipeline mask AP: EfficientSAM-Ti 0.180, EfficientSAM-S 0.184, YOLO-seg 0.142
- Best EfficientSAM variant on mask AP: EfficientSAM-S
- Speed: YOLO detector only 25 ms, YOLO + EfficientSAM-Ti 160 ms, YOLO + EfficientSAM-S 292 ms, YOLO-seg 23 ms

## E1 detection (Ultralytics, test)

| class             |   GT objects |   Precision |   Recall |    F1 |   mAP50 |   mAP50-95 |
|:------------------|-------------:|------------:|---------:|------:|--------:|-----------:|
| Plastic           |          321 |       0.518 |    0.421 | 0.464 |   0.407 |      0.323 |
| Paper & Cardboard |           74 |       0.49  |    0.284 | 0.359 |   0.258 |      0.212 |
| Metal             |           89 |       0.508 |    0.517 | 0.513 |   0.465 |      0.363 |
| Glass             |           46 |       0.144 |    0.043 | 0.067 |   0.027 |      0.013 |
| Other Waste       |          180 |       0.303 |    0.056 | 0.094 |   0.074 |      0.034 |
| all               |          710 |       0.393 |    0.264 | 0.316 |   0.246 |      0.189 |

## E1 detection at conf ≥ 0.25

|                   |   TP |   FP |   FN |   Precision |   Recall |    F1 |
|:------------------|-----:|-----:|-----:|------------:|---------:|------:|
| Plastic           |  133 |  215 |  188 |       0.382 |    0.414 | 0.398 |
| Paper & Cardboard |   20 |   33 |   54 |       0.377 |    0.27  | 0.315 |
| Metal             |   45 |   76 |   44 |       0.372 |    0.506 | 0.429 |
| Glass             |    2 |   12 |   44 |       0.143 |    0.043 | 0.067 |
| Other Waste       |   18 |   72 |  162 |       0.2   |    0.1   | 0.133 |
| all (micro)       |  218 |  408 |  492 |       0.348 |    0.307 | 0.326 |

## Main segmentation comparison

| Method                     | Prompt        |   Mask AP |   Mask AP50 |   Mask AP75 |   mIoU |   mDice |   IoU≥0.5 rate |   ms / image |   Params (M) |
|:---------------------------|:--------------|----------:|------------:|------------:|-------:|--------:|---------------:|-------------:|-------------:|
| GT box → EfficientSAM-Ti   | oracle box    |   nan     |     nan     |     nan     |  0.781 |   0.863 |          0.901 |      nan     |       10.223 |
| GT box → EfficientSAM-S    | oracle box    |   nan     |     nan     |     nan     |  0.787 |   0.867 |          0.903 |      nan     |       26.413 |
| YOLO box → EfficientSAM-Ti | predicted box |     0.18  |       0.233 |       0.189 |  0.877 |   0.929 |          0.978 |      160.131 |       19.653 |
| YOLO box → EfficientSAM-S  | predicted box |     0.184 |       0.235 |       0.193 |  0.884 |   0.933 |          0.991 |      291.578 |       35.843 |
| YOLO-seg (one-stage)       | —             |     0.142 |       0.218 |       0.153 |  0.832 |   0.902 |          0.971 |       22.525 |       10.084 |

## E2 mean IoU by class (GT boxes)

| cls               |   EfficientSAM-S |   EfficientSAM-Ti |
|:------------------|-----------------:|------------------:|
| Plastic           |            0.82  |             0.807 |
| Paper & Cardboard |            0.901 |             0.885 |
| Metal             |            0.831 |             0.832 |
| Glass             |            0.719 |             0.735 |
| Other Waste       |            0.677 |             0.678 |

## E2 mean IoU by object size (GT boxes)

| size   |   EfficientSAM-S |   EfficientSAM-Ti |   objects |
|:-------|-----------------:|------------------:|----------:|
| small  |            0.656 |             0.66  |       306 |
| medium |            0.859 |             0.843 |       238 |
| large  |            0.926 |             0.914 |       166 |

## Mask AP by class

|                   |   EfficientSAM-Ti |   EfficientSAM-S |   YOLO-seg |
|:------------------|------------------:|-----------------:|-----------:|
| Plastic           |             0.299 |            0.307 |      0.253 |
| Paper & Cardboard |             0.212 |            0.217 |      0.133 |
| Metal             |             0.353 |            0.362 |      0.286 |
| Glass             |             0.012 |            0.013 |      0.007 |
| Other Waste       |             0.024 |            0.022 |      0.03  |
