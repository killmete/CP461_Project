# Benchmark notes: TACO detection / instance segmentation and EfficientSAM

This answers the instructor's question: *"What is the benchmark for this task? Is the result we got higher or lower than it?"*
Every number below was read from the primary source (arXiv PDF, publisher or PMC full text) on 2026-11-13. Machine-readable version: `benchmark_table.csv` (same folder). The `OURS-v2 (pending)` row in the CSV is a placeholder for the improved detector.

Confidence labels:
- **[V-text]**: verified from the source's own text or table text (PDF text layer / HTML table).
- **[V-img]**: verified by rendering the PDF table as an image and reading it, because the text layer was scrambled.
- **[Unverified]**: seen but not confirmed. Kept out of the CSV.

---

## 1. Sources and where each number comes from

### S1. TACO paper (the dataset's own benchmark)
Proença, P. F., & Simões, P. (2020). *TACO: Trash Annotations in Context for Litter Detection.* arXiv:2003.06975 (preprint, not peer-reviewed). https://arxiv.org/abs/2003.06975
- Setup (Sec. III, III-A): Mask R-CNN (Matterport implementation), ResNet-50-FPN, COCO-pretrained, input 1024×1024 (resize + pad), 100 epochs, SGD. **4-fold cross-validation**, each fold random 80/10/10 train/val/test. Metric: AP "averaged over IoU thresholds" on **instance masks** (COCO-style AP; the paper cites VOC for AP but averages over IoU thresholds). [V-text]
- Tasks: **TACO-1** = class-agnostic litter; **TACO-10** = 9 most frequent classes + "Other" (Bottle, Bottle cap, Can, Cigarette, Cup, Lid, Other, Plastic bag/wrapper, Pop tab, Straw). [V-text, Fig. 5/9 labels]
- **Table I** (mask AP, mean ± std over 4 folds) [V-text]:

| Dataset | class_score | litter_score | ratio_score |
|---|---|---|---|
| TACO-1 | 15.9 ± 1.0 | 26.2 ± 1.0 | 26.1 ± 1.0 |
| TACO-10 | 17.6 ± 1.6 | 18.4 ± 1.5 | 19.4 ± 1.5 |

- The authors say the low score comes mostly from tiny cigarettes and many objects under 20×20 px after downscaling. [V-text]

### S2. Majchrowska et al. (Detect Waste in Pomerania): the closest **material-class** regrouping of TACO
Majchrowska, S., Mikołajczyk, A., Ferlin, M., Klawikowska, Z., Plantykow, M. A., Kwasigroch, A., & Majek, K. (2022). *Deep learning-based waste detection in natural and urban environments.* Waste Management 138, 274–284. doi:10.1016/j.wasman.2021.12.001. Preprint: arXiv:2105.06808 ("Waste detection in Pomerania…"). https://arxiv.org/abs/2105.06808
- "Extended TACO": the official TACO plus about 3,000 unofficial TACO images that the authors box-annotated (Table 1: 4,562 images, 14,286 objects, 7 classes). Classes follow the Gdańsk sorting rules: bio, glass, metals & plastic, non-recyclable, other, paper, unknown. Split: random 80% train, rest val/test, stratified by class. Model: EfficientDet-D2, COCO-pretrained, 20 epochs, default EfficientDet input size. [V-text]
- **Table 6** (box detection) [V-text]:
  - Extended TACO, **7 classes**: mAP@0.50:0.95 **11.9**, mAP@0.50 **16.2**, mAP@0.75 13.0, APs/APm/APl 6.4/9.4/15.0
  - Extended TACO, 1 class: 40.4 / 56.8 / 43.5
- Text: within the 7-class model only "metals and plastic" reaches AP50 = 43.3. The other six classes range from 0.1 (bio) to 8.9 (unknown). [V-text]

### S3. Córdova et al.: YOLOv5 and others on TACO, single "litter" class
Córdova, M., Pinto, A., Hellevik, C. C., Alaliyat, S. A.-A., Hameed, I. A., Pedrini, H., & Torres, R. da S. (2022). *Litter Detection with Deep Learning: A Comparative Study.* Sensors 22(2), 548. https://doi.org/10.3390/s22020548 (full text: PMC8812282)
- Split: 80% train / 20% test, 5-fold CV inside train for model selection (Sec. 4). YOLO-v5 at 640×640 for 100 epochs; Faster/Mask R-CNN and RetinaNet at 800×800 (Table 3). AP@ / AR@ = average over IoU 0.50:0.95. [V-text]
- **Table 6, "Litter detection results on TACO"** (box) [V-text]:

| Model | AP50 | AP@[.5:.95] | AR@ | F1@ |
|---|---|---|---|---|
| RetinaNet | 50.6 | 26.7 | 37.1 | 31.1 |
| Faster R-CNN | 51.1 | 28.1 | 36.9 | 31.9 |
| Mask R-CNN | 52.3 | 29.2 | 38.6 | 33.2 |
| EfficientDet-d0 | 32.7 | 23.8 | 28.4 | 25.9 |
| EfficientDet-d5 | 42.3 | 35.2 | 40.3 | 37.6 |
| YOLO-v5s | 54.7 | 38.8 | 58.1 | 46.5 |
| YOLO-v5x | 63.3 | 48.4 | 66.4 | 56.0 |

- Caveat: the paper treats TACO as a "litter detection" (one-class) task and points to the one-category TACO set-up. It never states the class count in one explicit sentence, so I read it as 1 class. Its F1@ is computed from AP@/AR@, not at a fixed confidence threshold, so it is **not** comparable to our F1.

### S4. Saakyan & Solntsev (D-FINE-seg): most recent TACO numbers with the same YOLO size and input size as ours
Saakyan, A., & Solntsev, D. (2026). *D-FINE-seg: Object Detection and Instance Segmentation Framework with multi-backend deployment.* arXiv:2602.23043 (preprint). https://arxiv.org/abs/2602.23043
- Sec. 5.2/5.3: TACO, **59 effective classes** (original fine categories), split 86/14 train/val **by batch ID**, metrics on the val set (212 images), 640×640 input, YOLO26 for 100 epochs, D-FINE-seg for 50 epochs. COCO-style AP with conf threshold 0.01 and max 100 detections. P/R/F1 at default threshold (0.25 YOLO26, 0.5 D-FINE), TP when IoU > 0.5 and the class matches. [V-text]
- **Table 1 (mask AP)** [V-img]: YOLO26-seg S 0.111 (50-95) / 0.165 (50); D-FINE-seg S 0.177 / 0.250; YOLO26-seg X 0.210 / 0.291; D-FINE-seg X 0.242 / 0.340.
- **Table 2 (box AP)** [V-img]: YOLO26 S 0.098 / 0.124; D-FINE S 0.202 / 0.244; YOLO26 X 0.256 / 0.300; D-FINE X 0.269 / 0.336.
- **Table 4 (detection, P/R/F1)** [V-img]: YOLO26 S P 0.279, R 0.122, F1 0.170; D-FINE S P 0.327, R 0.240, F1 0.274.

### S5. Ramadhani & Widyaningrum (JAIC 2026): YOLOv11 on TACO regrouped to 4 waste classes (lower trust)
Ramadhani, L. K., & Widyaningrum, B. N. (2026). *Benchmarking YOLO26 Against YOLOv11 for Minority Class Waste Detection on an Augmented TACO Dataset.* Journal of Applied Informatics and Computing 10(4), 4100–4108. https://jurnal.polibatam.ac.id/index.php/JAIC/article/download/13712/3915/44521
- TACO **plus** RecyBat24 and a Food-Waste dataset (2,326 images), remapped to 4 classes (Inorganic, Organic, B3/hazardous, Residue), stratified 70/15/15, 640×640, 50 epochs. [V-text]
- **Table 4** (test): YOLOv11 P 0.584, R 0.495, mAP50 **0.507**, mAP50-95 **0.385**. [V-text]
- The overall number is inflated by the easy external battery images (B3 mAP50 about 0.99). The "Residue" class, closest to our *Other Waste*, gets only mAP50 0.076–0.085. Regional journal, so lower trust. I could not confirm the YOLOv11 size variant; their YOLO26 has 2.4M params, i.e. nano.

### S6. EfficientSAM (our reference paper)
Xiong, Y., Varadarajan, B., Wu, L., Xiang, X., Xiao, F., Zhu, C., Dai, X., Wang, D., Sun, F., Iandola, F., Krishnamoorthi, R., & Chandra, V. (2024). *EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything.* CVPR 2024, pp. 16111–16121. arXiv:2312.00863. https://arxiv.org/abs/2312.00863. Code: https://github.com/yformer/EfficientSAM
- **Table 4: zero-shot single valid mask, mIoU.** The box prompt is the tightest box around the GT mask, which is the **same protocol as our GT-box test**. [V-text]

| Model | COCO box | COCO 1 click | COCO 3 click | LVIS box | LVIS 1 click | LVIS 3 click |
|---|---|---|---|---|---|---|
| SAM (ViT-H) | 78.4 | 55.6 | 74.1 | 78.9 | 59.8 | 75.2 |
| MobileSAM | 74.2 | 43.7 | 59.7 | 73.8 | 51.0 | 54.4 |
| EfficientSAM-Ti | 75.7 | 45.5 | 67.2 | 74.3 | 52.7 | 66.8 |
| EfficientSAM-S | 76.9 | 50.0 | 69.8 | 75.4 | 56.2 | 68.7 |

- **Table 5: zero-shot instance segmentation, mask AP, with ViTDet-H boxes as prompts.** Same detector→SAM design as our pipeline. [V-text]

| Model | COCO AP | APs | APm | APl | LVIS AP |
|---|---|---|---|---|---|
| ViTDet-H (its own masks) | 51.0 | 32.0 | 54.3 | 68.9 | 46.6 |
| SAM | 46.5 | 30.8 | 51.0 | 61.7 | 44.7 |
| FastSAM | 37.9 | 23.9 | 43.4 | 50.0 | 34.5 |
| MobileSAM | 38.7 | 23.7 | 42.2 | 54.3 | 34.4 |
| EfficientSAM-Ti | 42.3 | 26.7 | 46.2 | 57.4 | 39.9 |
| EfficientSAM-S | 44.4 | 28.4 | 48.3 | 60.1 | 42.3 |

- **Table 8 (supplementary): efficiency**, 1024×1024 input, one box, single A100 [V-img]: SAM 636M params, 2 img/s; EfficientSAM-Ti 10M, 54 img/s; EfficientSAM-S 25M, 47 img/s. The text also gives 9.8M for EfficientSAM-Ti. [V-text]

### S7. SAM on TACO (the only published SAM-on-TACO number found)
Abou Baker, N., & Handmann, U. (2026). *Don't waste SAM.* arXiv:2606.10696 (preprint, 6 pages). https://arxiv.org/abs/2606.10696
- Box prompts from GT masks, 80/20 split when no official split exists, mean IoU. **Table 1, TACO** [V-text]: SAM-ViT-B 40.25, SAM-ViT-L 41.33, **SAM-ViT-H 46.29 (zero-shot)**. Fine-tuned decoder: ViT-B 50.40, ViT-L 52.67, ViT-H 61.06.
- Caveat: our EfficientSAM GT-box mIoU on TACO (0.78) is far above their zero-shot SAM-ViT-H (0.46), even though SAM-ViT-H beats EfficientSAM on COCO. Their IoU computation or preprocessing is probably different (the paper does not give enough detail). Treat it as a reference only, not a head-to-head comparison.

### S8. pLitterStreet (number found, metric unclear, so left out of the CSV)
Mandhati, S. R., et al. (2024). *pLitterStreet: Street Level Plastic Litter Detection and Mapping.* arXiv:2401.14719. Table 4 gives 1-class TACO scores (Faster R-CNN trained on TACO: 43.6; YOLO-v5l trained on TACO: 45.4; 1024 px input). The Table 4 caption does not say whether this is AP or AP50. The paper defines both mAP50 and mAP elsewhere. **[Unverified metric]**, so it is not in the CSV.

### Searched but not found
- No peer-reviewed paper that combines a **detector + SAM/EfficientSAM pipeline on TACO** and reports mask AP. The closest matches are EfficientSAM Table 5 (same design, on COCO/LVIS) and "Don't waste SAM" (SAM on TACO with GT boxes only). Our YOLO→EfficientSAM on TACO appears to be uncommon or new for this dataset.
- No paper using **exactly our 5 material classes**. The nearest is Majchrowska et al. (7 classes, metal and plastic merged).

---

## 2. Fair comparison: our numbers against the closest published numbers

| Our number | Closest published | Apples-to-apples? |
|---|---|---|
| **Detector mAP50 0.246 / mAP50-95 0.189** (YOLO11s, 5 material classes, 640) | **Majchrowska 2022**: EfficientDet-D2, 7 material classes, mAP50 **0.162**, mAP50-95 **0.119** | *Partly.* Both regroup TACO by material. Differences: they have about 3× more images (Extended TACO), 7 vs 5 classes (more classes is harder), an 80/20 random split, and an older detector. **Ours is higher by +8.4 / +7.0 points.** |
| same | **D-FINE-seg 2026**: YOLO26 S (same size, same 640 input) box mAP50 0.124 / 50-95 0.098; D-FINE S 0.244 / 0.202, all on 59 classes | *Partly.* Same model size and resolution, but 59 fine-grained classes is a much harder task than 5. Ours is above YOLO26 S and roughly equal to D-FINE S. |
| same | **Córdova 2022**: YOLO-v5s, 1 class, AP50 **0.547**, AP **0.388** | *No.* With one class there is no classification error, so it shows the localisation ceiling. Ours is lower, as expected for 5 classes. Note the gap: our recall is low. |
| **Detector P 0.393 / R 0.264 / F1 0.316** | **D-FINE-seg Table 4**: YOLO26 S P 0.279 / R 0.122 / F1 0.170; D-FINE S 0.327 / 0.240 / 0.274 (59 classes, fixed threshold) | *Partly* (same fixed-threshold definition, different class count). Ours is higher. Córdova's F1@ is a different definition and should not be compared. |
| **Pipeline mask AP 0.184 (S) / 0.180 (Ti)**, AP50 0.235 / 0.233 | **TACO paper Table I**, Mask R-CNN TACO-10 mask AP **0.176** (standard class_score), 0.194 with the authors' ratio_score | *Partly.* Same metric (mask AP over IoU thresholds), same base images. Differences: 10 object-type classes vs our 5 material classes, 4-fold CV vs one fixed split, 1024 vs 640 input. **Ours is on par: +0.8 pt above the standard score, −1.0 pt below their best ranking trick, inside their ±1.5 std.** |
| same | **D-FINE-seg Table 1**: YOLO26-seg S mask AP 0.111 / AP50 0.165; D-FINE-seg S 0.177 / 0.250 (59 classes) | *Partly.* Ours is higher than YOLO26-seg S and similar to D-FINE-seg S. |
| same | **EfficientSAM Table 5 (COCO)**: ViTDet-H → EfficientSAM-S 0.444, -Ti 0.423 | *No* as an absolute benchmark: different dataset (COCO), and a much stronger detector. It does show the **same pattern**: S beats Ti by about 2 points on COCO, and by 0.4 points in our pipeline. |
| **YOLO11s-seg baseline mask AP 0.142** | YOLO26-seg S 0.111 (59 classes), TACO paper Mask R-CNN 0.176 (10 classes) | *Partly.* Our end-to-end baseline sits between them. The pipeline beats our own baseline by +4.2 AP, which matches EfficientSAM's message that box-prompted SAM masks beat lightweight learned mask heads. |
| **EfficientSAM GT-box mIoU 0.781 (Ti) / 0.787 (S)** on TACO | **EfficientSAM Table 4, COCO box**: 0.757 (Ti) / 0.769 (S); LVIS 0.743 / 0.754 | *Yes*, the most apples-to-apples comparison we have: same model weights, same prompt protocol (tight GT box, top-1 mask), same metric. Only the dataset differs. **Ours is +2.4 / +1.8 points higher**, so EfficientSAM transfers to litter zero-shot without loss. TACO objects are mostly medium/large in our resized 1280 images, which likely helps. |
| same | **Don't waste SAM**: SAM ViT-H zero-shot on TACO, mean IoU 0.463 | *Questionable.* Same idea, but their number is implausibly low against SAM's COCO 0.784. Their IoU definition likely differs. Do not claim we "beat SAM by 32 points". |

**What would make the comparison fair**
1. Re-score our TACO-10 baseline the way the TACO paper does (60→10 classes, 4-fold CV, mask AP). This gives a direct row against 17.6 ± 1.6.
2. Or re-score our predictions with Majchrowska's 7-class mapping. This is only possible on official TACO; their Extended TACO images are not all public under the same annotations.
3. Report mean ± std over at least 3 seeds or folds. With 225 test images, the TACO paper's ±1.5 AP fold spread shows that differences under about 1.5 AP are within noise.
4. Report APs/APm/APl so the small-object weakness (cigarettes, bottle caps) is visible, as in the TACO paper and Majchrowska.

---

## 3. Verdict

**English**
- **Detection:** our YOLO11s (5 material classes) reaches mAP50 0.246 / mAP50-95 0.189. That is **higher** than the closest material-class benchmark (EfficientDet-D2, 7 classes: 0.162 / 0.119) and higher than a same-size YOLO26-S on 59-class TACO (0.124 / 0.098). It is **lower** than single-class "litter vs background" results (YOLOv5s AP50 0.547), which do not count classification errors.
- **Segmentation:** our YOLO→EfficientSAM-S mask AP 0.184 is **on par with** the official TACO benchmark (Mask R-CNN TACO-10: 0.176–0.194 ± 0.015). It beats our own YOLO11s-seg baseline (0.142) by +4.2 AP.
- **EfficientSAM:** with GT boxes, it reaches mIoU 0.78–0.79 on TACO, slightly **higher** than the paper's COCO figures (0.757 / 0.769). The bottleneck is the detector, especially recall (0.264), not the segmenter.
- **Caveat:** no published work uses exactly our 5 classes and 70/15/15 split, so these are near-comparisons. The likely reasons ours is not higher overall are 640 px training (tiny objects are lost), class imbalance, and only 1,050 training images. A seed/fold-averaged run on the TACO-10 protocol would make the comparison fully fair.

**ภาษาไทย**
- **การตรวจจับ (Detection):** YOLO11s ของเรา (5 คลาสวัสดุ) ได้ mAP50 0.246 / mAP50-95 0.189 **สูงกว่า** benchmark ที่ใกล้ที่สุดซึ่งแบ่งคลาสตามวัสดุ (EfficientDet-D2, 7 คลาส: 0.162 / 0.119) และสูงกว่า YOLO26-S ขนาดเดียวกันบน TACO 59 คลาส (0.124 / 0.098) แต่ **ต่ำกว่า** งานที่ตรวจจับแบบคลาสเดียว "ขยะ/ไม่ใช่ขยะ" (YOLOv5s AP50 0.547) ซึ่งไม่ต้องจำแนกประเภท
- **การแบ่งส่วน (Segmentation):** pipeline YOLO→EfficientSAM-S ได้ mask AP 0.184 **ใกล้เคียง/เทียบเท่า** benchmark ทางการของ TACO (Mask R-CNN, TACO-10: 0.176–0.194 ± 0.015) และดีกว่า baseline YOLO11s-seg ของเราเอง (0.142) อยู่ +4.2 AP
- **EfficientSAM:** เมื่อใช้กล่อง ground-truth ได้ mIoU 0.78–0.79 บน TACO ซึ่ง **สูงกว่าเล็กน้อย** จากค่าในเปเปอร์บน COCO (0.757 / 0.769) แสดงว่าคอขวดอยู่ที่ตัวตรวจจับ (โดยเฉพาะ recall 0.264) ไม่ใช่ตัวแบ่งส่วน
- **ข้อควรระวัง:** ยังไม่มีงานตีพิมพ์ที่ใช้ 5 คลาสและการแบ่งข้อมูล 70/15/15 แบบเดียวกับเรา จึงเป็นการเทียบแบบใกล้เคียงเท่านั้น สาเหตุที่ผลยังไม่สูงกว่านี้น่าจะมาจากการเทรนที่ 640 px (วัตถุเล็กหายไป) ความไม่สมดุลของคลาส และข้อมูลเทรนเพียง 1,050 ภาพ หากต้องการเทียบอย่างยุติธรรมควรรันตามโปรโตคอล TACO-10 และเฉลี่ยหลาย seed/fold

---

## 4. Confidence summary per number
| Number | Confidence |
|---|---|
| TACO Table I (all 6 values) | V-text |
| Majchrowska Table 6, Table 5 | V-text (PDF text layer is a clean table) |
| Córdova Table 6, Table 3 settings | V-text (PMC HTML table) |
| Córdova "1 class" setting | V-text for the description; class count inferred, not stated in one explicit sentence |
| D-FINE-seg Tables 1, 2, 4 | V-img (text layer scrambled, read from rendered page) |
| JAIC Table 4 | V-text; YOLOv11 size variant could not be confirmed |
| EfficientSAM Tables 4, 5 | V-text |
| EfficientSAM Table 8 (params / throughput) | V-img |
| EfficientSAM venue (CVPR 2024, pp. 16111–16121) | Verified via CVF open-access listing (search result) and CVPR virtual poster page; arXiv PDF itself is v1 |
| Don't waste SAM Table 1 | V-text; how IoU is computed is unclear |
| pLitterStreet Table 4 (43.6 / 45.4) | **Could not fully verify the metric (AP vs AP50)**, so excluded from the CSV |
