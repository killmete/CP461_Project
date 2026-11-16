# Results summary (auto-generated)

- Test split: 225 images, 710 objects; detector `yolo11s.pt` at 960px (v2: RFS + 960 px)
- Detector (test): P=0.288  R=0.357  F1=0.319  mAP50=0.270  mAP50-95=0.206
- Classification accuracy of localised objects (conf ≥ 0.25): 0.703
- E2 GT-box mIoU: EfficientSAM-Ti 0.781, EfficientSAM-S 0.787
- E3 pipeline mask AP: EfficientSAM-Ti 0.190, EfficientSAM-S 0.196, YOLO-seg 0.142
- Best EfficientSAM variant on mask AP: EfficientSAM-S
- Speed: YOLO detector only 27 ms, YOLO + EfficientSAM-Ti 167 ms, YOLO + EfficientSAM-S 324 ms, YOLO-seg 22 ms

## E1 detection (Ultralytics, test)

| class             |   GT objects |   Precision |   Recall |    F1 |   mAP50 |   mAP50-95 |
|:------------------|-------------:|------------:|---------:|------:|--------:|-----------:|
| Plastic           |          321 |       0.309 |    0.611 | 0.41  |   0.463 |      0.372 |
| Paper & Cardboard |           74 |       0.357 |    0.297 | 0.324 |   0.229 |      0.173 |
| Metal             |           89 |       0.418 |    0.596 | 0.491 |   0.512 |      0.416 |
| Glass             |           46 |       0.151 |    0.109 | 0.126 |   0.054 |      0.028 |
| Other Waste       |          180 |       0.206 |    0.172 | 0.187 |   0.09  |      0.042 |
| all               |          710 |       0.288 |    0.357 | 0.319 |   0.27  |      0.206 |

## E1 detection at conf ≥ 0.25

|                   |   TP |   FP |   FN |   Precision |   Recall |    F1 |
|:------------------|-----:|-----:|-----:|------------:|---------:|------:|
| Plastic           |  173 |  265 |  148 |       0.395 |    0.539 | 0.456 |
| Paper & Cardboard |   13 |   23 |   61 |       0.361 |    0.176 | 0.236 |
| Metal             |   45 |   55 |   44 |       0.45  |    0.506 | 0.476 |
| Glass             |    2 |   16 |   44 |       0.111 |    0.043 | 0.062 |
| Other Waste       |   16 |   39 |  164 |       0.291 |    0.089 | 0.136 |
| all (micro)       |  249 |  398 |  461 |       0.385 |    0.351 | 0.367 |

## Main segmentation comparison

| Method                     | Prompt        |   Mask AP |   Mask AP50 |   Mask AP75 |   mIoU |   mDice |   IoU≥0.5 rate |   ms / image |   Params (M) |
|:---------------------------|:--------------|----------:|------------:|------------:|-------:|--------:|---------------:|-------------:|-------------:|
| GT box → EfficientSAM-Ti   | oracle box    |   nan     |     nan     |     nan     |  0.781 |   0.863 |          0.901 |      nan     |       10.223 |
| GT box → EfficientSAM-S    | oracle box    |   nan     |     nan     |     nan     |  0.787 |   0.867 |          0.903 |      nan     |       26.413 |
| YOLO box → EfficientSAM-Ti | predicted box |     0.19  |       0.245 |       0.201 |  0.864 |   0.919 |          0.969 |      167.045 |       19.653 |
| YOLO box → EfficientSAM-S  | predicted box |     0.196 |       0.248 |       0.207 |  0.876 |   0.927 |          0.975 |      324.009 |       35.843 |
| YOLO-seg (one-stage)       | —             |     0.142 |       0.218 |       0.153 |  0.832 |   0.902 |          0.971 |       21.557 |       10.084 |

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
| Plastic           |             0.351 |            0.364 |      0.253 |
| Paper & Cardboard |             0.146 |            0.157 |      0.133 |
| Metal             |             0.403 |            0.413 |      0.286 |
| Glass             |             0.022 |            0.023 |      0.007 |
| Other Waste       |             0.026 |            0.023 |      0.03  |

## E1 detector v1 vs v2 (Ultralytics, test)

|    | Unnamed: 0        | Precision   | Precision.1   | Recall   | Recall.1   | F1    | F1.1   | mAP50   | mAP50.1   | mAP50-95   | mAP50-95.1   |
|---:|:------------------|:------------|:--------------|:---------|:-----------|:------|:-------|:--------|:----------|:-----------|:-------------|
|  0 | nan               | v1          | v2            | v1       | v2         | v1    | v2     | v1      | v2        | v1         | v2           |
|  1 | class             | nan         | nan           | nan      | nan        | nan   | nan    | nan     | nan       | nan        | nan          |
|  2 | Plastic           | 0.518       | 0.309         | 0.421    | 0.611      | 0.464 | 0.41   | 0.407   | 0.463     | 0.323      | 0.372        |
|  3 | Paper & Cardboard | 0.49        | 0.357         | 0.284    | 0.297      | 0.359 | 0.324  | 0.258   | 0.229     | 0.212      | 0.173        |
|  4 | Metal             | 0.508       | 0.418         | 0.517    | 0.596      | 0.513 | 0.491  | 0.465   | 0.512     | 0.363      | 0.416        |
|  5 | Glass             | 0.144       | 0.151         | 0.043    | 0.109      | 0.067 | 0.126  | 0.027   | 0.054     | 0.013      | 0.028        |
|  6 | Other Waste       | 0.303       | 0.206         | 0.056    | 0.172      | 0.094 | 0.187  | 0.074   | 0.09      | 0.034      | 0.042        |
|  7 | all               | 0.393       | 0.288         | 0.264    | 0.357      | 0.316 | 0.319  | 0.246   | 0.27      | 0.189      | 0.206        |

## E7 imbalance-aware detector comparison

|    | detector / setting       |   F1 (macro) |   Balanced accuracy |   Precision (macro) |   Recall (macro) |   F1 (weighted) |   F1 (micro) |   Precision (micro) |   Recall (micro) |   Cohen kappa |   MCC |   Accuracy (all GT objects) |   Accuracy (localised objects) |   FP on background |   ROC-AUC (macro) |   PR-AUC (macro) |   F1 Plastic |   F1 Paper & Cardboard |   F1 Metal |   F1 Glass |   F1 Other Waste |   Recall Plastic |   Recall Paper & Cardboard |   Recall Metal |   Recall Glass |   Recall Other Waste |
|---:|:-------------------------|-------------:|--------------------:|--------------------:|-----------------:|----------------:|-------------:|--------------------:|-----------------:|--------------:|------:|----------------------------:|-------------------------------:|-------------------:|------------------:|-----------------:|-------------:|-----------------------:|-----------:|-----------:|-----------------:|-----------------:|---------------------------:|---------------:|---------------:|---------------------:|
|  0 | v1 · conf 0.25           |        0.268 |               0.267 |               0.295 |            0.267 |           0.304 |        0.326 |               0.348 |            0.307 |         0.191 | 0.248 |                       0.307 |                          0.675 |                303 |             0.674 |            0.391 |        0.398 |                  0.315 |      0.429 |      0.067 |            0.133 |            0.414 |                      0.27  |          0.506 |          0.043 |                0.1   |
|  1 | v1 · per-class thr (val) |        0.289 |               0.245 |               0.365 |            0.245 |           0.329 |        0.343 |               0.427 |            0.287 |         0.178 | 0.239 |                       0.287 |                          0.689 |                182 |             0.674 |            0.391 |        0.435 |                  0.325 |      0.484 |      0.067 |            0.133 |            0.396 |                      0.257 |          0.427 |          0.043 |                0.1   |
|  2 | v2 · conf 0.25           |        0.273 |               0.271 |               0.322 |            0.271 |           0.329 |        0.367 |               0.385 |            0.351 |         0.214 | 0.267 |                       0.351 |                          0.703 |                293 |             0.685 |            0.422 |        0.456 |                  0.236 |      0.476 |      0.062 |            0.136 |            0.539 |                      0.176 |          0.506 |          0.043 |                0.089 |
|  3 | v2 · per-class thr (val) |        0.275 |               0.221 |               0.446 |            0.221 |           0.341 |        0.368 |               0.479 |            0.299 |         0.177 | 0.237 |                       0.299 |                          0.716 |                147 |             0.685 |            0.422 |        0.484 |                  0.205 |      0.462 |      0.071 |            0.152 |            0.464 |                      0.122 |          0.371 |          0.043 |                0.106 |

## E7 object-level ROC-AUC / PR-AUC

|    | detector   | class             |   GT objects |   ROC-AUC |   PR-AUC (AP) |   PR baseline (prevalence) |
|---:|:-----------|:------------------|-------------:|----------:|--------------:|---------------------------:|
|  0 | v1         | Plastic           |          321 |     0.722 |         0.676 |                      0.452 |
|  1 | v1         | Paper & Cardboard |           74 |     0.691 |         0.341 |                      0.104 |
|  2 | v1         | Metal             |           89 |     0.785 |         0.493 |                      0.125 |
|  3 | v1         | Glass             |           46 |     0.536 |         0.082 |                      0.065 |
|  4 | v1         | Other Waste       |          180 |     0.634 |         0.361 |                      0.254 |
|  5 | v1         | macro             |          710 |     0.674 |         0.391 |                    nan     |
|  6 | v2         | Plastic           |          321 |     0.733 |         0.708 |                      0.452 |
|  7 | v2         | Paper & Cardboard |           74 |     0.675 |         0.306 |                      0.104 |
|  8 | v2         | Metal             |           89 |     0.806 |         0.576 |                      0.125 |
|  9 | v2         | Glass             |           46 |     0.586 |         0.132 |                      0.065 |
| 10 | v2         | Other Waste       |          180 |     0.623 |         0.386 |                      0.254 |
| 11 | v2         | macro             |          710 |     0.685 |         0.422 |                    nan     |

## E6 prompt study (overall)

|    | prompt                            | key   | group     | objects              |   mIoU Ti |   mDice Ti |   IoU≥0.5 Ti |   IoU≥0.75 Ti |   mIoU S |   mDice S |   IoU≥0.5 S |   IoU≥0.75 S |   n objects |
|---:|:----------------------------------|:------|:----------|:---------------------|----------:|-----------:|-------------:|--------------:|---------:|----------:|------------:|-------------:|------------:|
|  0 | GT box                            | P1    | practical | all GT objects       |     0.781 |      0.863 |        0.901 |         0.658 |    0.787 |     0.867 |       0.903 |        0.656 |         710 |
|  1 | Box-centre point                  | P2    | practical | all GT objects       |     0.622 |      0.713 |        0.687 |         0.448 |    0.673 |     0.757 |       0.759 |        0.517 |         710 |
|  2 | GT box + centre point             | P3    | practical | all GT objects       |     0.784 |      0.864 |        0.91  |         0.665 |    0.783 |     0.863 |       0.897 |        0.654 |         710 |
|  3 | Oracle interior point             | P4    | oracle    | all GT objects       |     0.644 |      0.741 |        0.696 |         0.465 |    0.688 |     0.777 |       0.765 |        0.52  |         710 |
|  4 | GT box + oracle point             | P5    | oracle    | all GT objects       |     0.787 |      0.867 |        0.91  |         0.661 |    0.787 |     0.866 |       0.901 |        0.663 |         710 |
|  5 | GT box + 3 oracle points          | P6    | oracle    | all GT objects       |     0.793 |      0.872 |        0.918 |         0.682 |    0.782 |     0.863 |       0.892 |        0.648 |         710 |
|  6 | GT box ±5% jitter                 | J05   | box error | all GT objects       |     0.78  |      0.862 |        0.9   |         0.646 |    0.786 |     0.866 |       0.903 |        0.658 |         710 |
|  7 | GT box ±10% jitter                | J10   | box error | all GT objects       |     0.774 |      0.857 |        0.896 |         0.642 |    0.784 |     0.864 |       0.901 |        0.655 |         710 |
|  8 | GT box ±20% jitter                | J20   | box error | all GT objects       |     0.75  |      0.84  |        0.872 |         0.587 |    0.768 |     0.853 |       0.897 |        0.621 |         710 |
|  9 | GT box ±5% jitter + centre point  | J05c  | box error | all GT objects       |     0.783 |      0.864 |        0.904 |         0.658 |    0.783 |     0.863 |       0.897 |        0.655 |         710 |
| 10 | GT box ±10% jitter + centre point | J10c  | box error | all GT objects       |     0.774 |      0.857 |        0.899 |         0.644 |    0.779 |     0.861 |       0.893 |        0.644 |         710 |
| 11 | GT box ±20% jitter + centre point | J20c  | box error | all GT objects       |     0.749 |      0.838 |        0.877 |         0.585 |    0.762 |     0.846 |       0.882 |        0.617 |         710 |
| 12 | GT box                            | P1    | practical | YOLO-matched objects |     0.879 |      0.931 |        0.977 |         0.876 |    0.888 |     0.936 |       0.986 |        0.89  |         354 |
| 13 | GT box + centre point             | P3    | practical | YOLO-matched objects |     0.881 |      0.932 |        0.983 |         0.873 |    0.883 |     0.932 |       0.977 |        0.879 |         354 |
| 14 | GT box + oracle point             | P5    | oracle    | YOLO-matched objects |     0.883 |      0.934 |        0.983 |         0.873 |    0.887 |     0.935 |       0.986 |        0.887 |         354 |
| 15 | YOLO box                          | Y1    | pipeline  | YOLO-matched objects |     0.864 |      0.919 |        0.969 |         0.845 |    0.876 |     0.927 |       0.975 |        0.859 |         354 |
| 16 | YOLO box + centre point           | Y3    | pipeline  | YOLO-matched objects |     0.867 |      0.922 |        0.969 |         0.847 |    0.872 |     0.924 |       0.969 |        0.853 |         354 |

## E6 prompt study by fill ratio (thin objects)

|    | method          | prompt   | prompt_name              |   thin (<0.3) (n=60) |   mid (0.3–0.6) (n=315) |   compact (≥0.6) (n=335) |
|---:|:----------------|:---------|:-------------------------|---------------------:|------------------------:|-------------------------:|
|  0 | EfficientSAM-Ti | P1       | GT box                   |                0.548 |                   0.751 |                    0.85  |
|  1 | EfficientSAM-Ti | P2       | Box-centre point         |                0.365 |                   0.615 |                    0.675 |
|  2 | EfficientSAM-Ti | P3       | GT box + centre point    |                0.535 |                   0.756 |                    0.856 |
|  3 | EfficientSAM-Ti | P4       | Oracle interior point    |                0.465 |                   0.63  |                    0.69  |
|  4 | EfficientSAM-Ti | P5       | GT box + oracle point    |                0.562 |                   0.757 |                    0.855 |
|  5 | EfficientSAM-Ti | P6       | GT box + 3 oracle points |                0.585 |                   0.764 |                    0.858 |
|  6 | EfficientSAM-S  | P1       | GT box                   |                0.573 |                   0.759 |                    0.852 |
|  7 | EfficientSAM-S  | P2       | Box-centre point         |                0.393 |                   0.647 |                    0.747 |
|  8 | EfficientSAM-S  | P3       | GT box + centre point    |                0.552 |                   0.754 |                    0.852 |
|  9 | EfficientSAM-S  | P4       | Oracle interior point    |                0.516 |                   0.664 |                    0.741 |
| 10 | EfficientSAM-S  | P5       | GT box + oracle point    |                0.579 |                   0.757 |                    0.851 |
| 11 | EfficientSAM-S  | P6       | GT box + 3 oracle points |                0.574 |                   0.753 |                    0.848 |

## Benchmark comparison

|    | metric                       | source                                    | setting                    | model                   |   value | caveat                            |
|---:|:-----------------------------|:------------------------------------------|:---------------------------|:------------------------|--------:|:----------------------------------|
|  0 | Box mAP50 (multi-class TACO) | Majchrowska et al. 2022, Waste Management | TACO+ · 7 material classes | EfficientDet-D2         |   0.162 | more images (≈4.5k), 7 classes    |
|  1 | Box mAP50 (multi-class TACO) | Saakyan & Solntsev 2026, arXiv 2602.23043 | TACO · 59 classes · 640 px | YOLO26-S                |   0.124 | 59 fine classes (harder)          |
|  2 | Box mAP50 (multi-class TACO) | Saakyan & Solntsev 2026, arXiv 2602.23043 | TACO · 59 classes · 640 px | D-FINE-S                |   0.244 | 59 fine classes (harder)          |
|  3 | Box mAP50 (1-class 'litter') | Córdova et al. 2022, Sensors              | TACO · 1 class             | YOLOv5s @640            |   0.547 | no classification needed (easier) |
|  4 | Mask AP (TACO)               | Proença & Simões 2020 (TACO paper)        | TACO-10 · 4-fold CV        | Mask R-CNN R50          |   0.176 | 10 classes, ±0.016 over folds     |
|  5 | Mask AP (TACO)               | Saakyan & Solntsev 2026, arXiv 2602.23043 | TACO · 59 classes · 640 px | YOLO26-seg S            |   0.111 | 59 fine classes                   |
|  6 | Mask AP (TACO)               | Saakyan & Solntsev 2026, arXiv 2602.23043 | TACO · 59 classes · 640 px | D-FINE-seg S            |   0.177 | 59 fine classes                   |
|  7 | Box-prompt mIoU (GT box)     | Xiong et al. 2024, EfficientSAM (CVPR)    | COCO                       | EfficientSAM-Ti         |   0.757 | reference paper; COCO not TACO    |
|  8 | Box-prompt mIoU (GT box)     | Xiong et al. 2024, EfficientSAM (CVPR)    | COCO                       | EfficientSAM-S          |   0.769 | reference paper; COCO not TACO    |
|  9 | Box-prompt mIoU (GT box)     | Xiong et al. 2024, EfficientSAM (CVPR)    | COCO                       | SAM ViT-H (636M)        |   0.784 | reference paper; 25–60× bigger    |
| 10 | Box mAP50 (multi-class TACO) | OURS                                      | TACO · 5 material classes  | YOLO11s v1 (progress)   |   0.246 | nan                               |
| 11 | Box mAP50 (multi-class TACO) | OURS                                      | TACO · 5 material classes  | YOLO11s v2 (final)      |   0.27  | nan                               |
| 12 | Mask AP (TACO)               | OURS                                      | TACO · 5 material classes  | YOLO → EfficientSAM-Ti  |   0.19  | detector v2                       |
| 13 | Mask AP (TACO)               | OURS                                      | TACO · 5 material classes  | YOLO → EfficientSAM-S   |   0.196 | detector v2                       |
| 14 | Mask AP (TACO)               | OURS                                      | TACO · 5 material classes  | YOLO11s-seg (one-stage) |   0.142 | detector v2                       |
| 15 | Box-prompt mIoU (GT box)     | OURS                                      | TACO test (710 objects)    | EfficientSAM-Ti         |   0.781 | zero-shot                         |
| 16 | Box-prompt mIoU (GT box)     | OURS                                      | TACO test (710 objects)    | EfficientSAM-S          |   0.787 | zero-shot                         |
