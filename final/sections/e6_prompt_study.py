# %% [markdown]
# ## 13 · E6: effect of the prompt on EfficientSAM
# **Question.** How much does the *type* and the *quality* of the prompt change the mask EfficientSAM returns, and can a better prompt fix the thin / elongated objects (ropes, straps) where a box makes EfficientSAM fill the whole box?
# **Prompts** (each test image is encoded once per model; every prompt reuses that embedding):
# * *Practical* (no ground truth needed): **P1** GT box (= E2) · **P2** one positive point at the box centre · **P3** box + box-centre point · and the real pipeline prompt, the **YOLO box** (± its centre point), on the objects the detector found.
# * *Oracle "user click"* (upper bound of what a human click adds): **P4** one point at the most interior pixel of the GT mask (distance-transform maximum) · **P5** box + that point · **P6** box + 3 interior points spread out by farthest-point sampling.
# * *Box quality*: GT boxes whose 4 sides are moved randomly by ±5 / 10 / 20 % of the box width/height (fixed seed), alone and + their centre point: how fast does the mask degrade when the detector's box is imprecise?
# Metrics per object: mask IoU, Dice, IoU ≥ 0.5 and ≥ 0.75 rates, split by class, size and **fill ratio** = mask area / box area (low fill = thin, diagonal or hollow object).

# %%
# ---- E6 setup: prompt definitions + helpers ----
free_gpu()
E6_PROMPTS = {   # key → (label, group)
    "P1": ("GT box", "practical"),
    "P2": ("Box-centre point", "practical"),
    "P3": ("GT box + centre point", "practical"),
    "P4": ("Oracle interior point", "oracle"),
    "P5": ("GT box + oracle point", "oracle"),
    "P6": ("GT box + 3 oracle points", "oracle"),
    "J05": ("GT box ±5% jitter", "box error"),
    "J10": ("GT box ±10% jitter", "box error"),
    "J20": ("GT box ±20% jitter", "box error"),
    "J05c": ("GT box ±5% jitter + centre point", "box error"),
    "J10c": ("GT box ±10% jitter + centre point", "box error"),
    "J20c": ("GT box ±20% jitter + centre point", "box error"),
    "Y1": ("YOLO box", "pipeline"),
    "Y3": ("YOLO box + centre point", "pipeline"),
}
E6_JITTER = {"J05": 0.05, "J10": 0.10, "J20": 0.20}
E6_MAIN = ["P1", "P2", "P3", "P4", "P5", "P6"]
E6_TAG = {v: {"ti": "Ti", "s": "S"}.get(v, v) for v in SAM_VARIANTS}
E6_PROMPT_COLORS = dict(zip(E6_MAIN, ["#2a78d6", "#a9c8ef", "#0d366b", "#f2b290", "#eb6834", "#a3361a"]))
FILL_BINS, FILL_LABELS = [0, 0.3, 0.6, 1.01], ["thin (<0.3)", "mid (0.3–0.6)", "compact (≥0.6)"]

def e6_box(b):
    return np.array([[b[0], b[1]], [b[2], b[3]]], np.float32), np.array([2, 3], np.int64)

def e6_pts(p):
    p = np.asarray(p, np.float32).reshape(-1, 2)
    return p, np.ones(len(p), np.int64)

def e6_cat(*ps):
    return np.concatenate([p[0] for p in ps]), np.concatenate([p[1] for p in ps])

def e6_centre(b):
    return [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]]

def interior_points(mk, k=3, max_cand=4000):
    # k positive "clicks" inside the GT mask: the most interior pixel (distance-transform max), then farthest-point
    # sampling over the mask core (pixels at ≥30% of the max distance to the border) so the points are spread out
    if not mk.any():
        return None
    dt = cv2.distanceTransform(np.pad(mk.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y0, x0 = np.unravel_index(int(dt.argmax()), dt.shape)
    ys, xs = np.nonzero(dt >= 0.3 * dt.max())
    if len(xs) > max_cand:
        sel = np.linspace(0, len(xs) - 1, max_cand).astype(int); ys, xs = ys[sel], xs[sel]
    pts, d = [(x0, y0)], np.hypot(xs - x0, ys - y0)
    for _ in range(k - 1):
        j = int(d.argmax()); pts.append((xs[j], ys[j])); d = np.minimum(d, np.hypot(xs - xs[j], ys - ys[j]))
    return np.array(pts, np.float32)

def jitter_box(b, s, rng, H, W):
    w, h = b[2] - b[0], b[3] - b[1]
    nb = b + rng.uniform(-s, s, 4) * np.array([w, h, w, h])
    nb = np.clip(nb, 0, [W - 1, H - 1, W - 1, H - 1])
    nb[2] = max(nb[2], nb[0] + 1); nb[3] = max(nb[3], nb[1] + 1)
    return nb.astype(np.float32)

@torch.inference_mode()
def esam_decode_prompts(model, emb, prompts, H, W, gt_masks=None, gt_idx=None, chunk=16, keep_masks=False):
    """Decode mixed point/box prompts in chunks.
    prompts: list of (points K×2 in px, labels K) — 1 = positive point, 2/3 = box corners; shorter prompts are padded with label -1.
    Returns pred_iou (n×3) for the 3 candidate masks, mask IoU vs gt_masks[gt_idx] (n×3, if given) and, with keep_masks,
    the candidate chosen by predicted IoU (same rule as esam_decode) as N×H×W bool."""
    piou_all, iou_all, masks = [], [], []
    for k in range(0, len(prompts), chunk):
        part = prompts[k:k + chunk]; m = len(part); K = max(len(l) for _, l in part)
        pts = np.full((m, K, 2), -1, np.float32); lbl = np.full((m, K), -1, np.int64)
        for j, (p, l) in enumerate(part):
            pts[j, :len(l)] = p; lbl[j, :len(l)] = l
        logits, piou = model.predict_masks(emb, torch.from_numpy(pts).to(DEVICE)[None], torch.from_numpy(lbl).to(DEVICE)[None],
                                           multimask_output=True, input_h=H, input_w=W, output_h=-1, output_w=-1)
        logits, piou = logits[0], piou[0]                                            # m×3×256×256, m×3
        piou_all.append(piou.float().cpu().numpy())
        if gt_masks is not None:
            mk = F.interpolate(logits.reshape(m * 3, 1, *logits.shape[-2:]), (H, W), mode="bicubic").view(m, 3, H, W) >= 0
            g = gt_masks[torch.as_tensor(np.asarray(gt_idx[k:k + chunk]), device=DEVICE)][:, None]
            inter = (mk & g).sum((-1, -2)).float(); uni = (mk | g).sum((-1, -2)).float()
            iou_all.append((inter / uni.clamp(min=1)).cpu().numpy()); del mk, g
        if keep_masks:
            low = logits[torch.arange(m, device=DEVICE), piou.argmax(-1)]
            masks.append((F.interpolate(low[:, None], (H, W), mode="bicubic")[:, 0] >= 0).cpu().numpy())
    cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape)
    return (cat(piou_all, (0, 3)), cat(iou_all, (0, 3)) if gt_masks is not None else None,
            cat(masks, (0, H, W)).astype(bool) if keep_masks else None)

def e6_build_prompts(i, H, W):
    # every prompt of one image (model independent) → list of (key, gt_idx, (pts, lbl), prompt box or None)
    g, out = GT[i], []
    for k, b in enumerate(g["boxes"]):
        mk = mask_utils.decode(g["rles"][k]).astype(bool)
        ip = interior_points(mk, 3)
        if ip is None:                                     # degenerate annotation → fall back to the box centre
            ip = np.repeat(np.array(e6_centre(b), np.float32), 3, 0)
        out += [("P1", k, e6_box(b), b), ("P2", k, e6_pts(e6_centre(b)), None),
                ("P3", k, e6_cat(e6_box(b), e6_pts(e6_centre(b))), b),
                ("P4", k, e6_pts(ip[:1]), None), ("P5", k, e6_cat(e6_box(b), e6_pts(ip[:1])), b),
                ("P6", k, e6_cat(e6_box(b), e6_pts(ip)), b)]
        for key, s in E6_JITTER.items():
            rng = np.random.default_rng([SEED, int(i), k, int(round(s * 100))])
            jb = jitter_box(b, s, rng, H, W)
            out += [(key, k, e6_box(jb), jb), (key + "c", k, e6_cat(e6_box(jb), e6_pts(e6_centre(jb))), jb)]
    pim = per_image[i]
    for pk, gk in pim["pairs"]:                            # the real pipeline prompt (YOLO box at CONF_THRES, matched objects)
        yb = pim["pb"][pk].astype(np.float32)
        out += [("Y1", gk, e6_box(yb), yb), ("Y3", gk, e6_cat(e6_box(yb), e6_pts(e6_centre(yb))), yb)]
    return out

# %%
# ---- E6 run: every GT object of the test split × every prompt × Ti and S ----
free_gpu()
e6_rows, E6_PROMPT_CACHE = [], {}
e6_sec = {v: 0.0 for v in SAM_VARIANTS}
t_all = time.perf_counter()
for i in tqdm(test_ids, desc="E6 prompt study"):
    g = GT[i]
    if len(g["boxes"]) == 0:
        continue
    img = load_rgb(i); H, W = img.shape[:2]
    plist = e6_build_prompts(i, H, W)
    for key, k, pr, _ in plist:
        E6_PROMPT_CACHE[(i, k, key)] = pr
    gt_np = np.stack([mask_utils.decode(r).astype(bool) for r in g["rles"]])
    fill = np.array([mask_utils.area(r) for r in g["rles"]], float) / np.maximum(
        (g["boxes"][:, 2] - g["boxes"][:, 0]) * (g["boxes"][:, 3] - g["boxes"][:, 1]), 1)
    pbox_iou = [float(box_iou(pb[None], g["boxes"][k:k + 1])[0, 0]) if pb is not None else np.nan for _, k, _, pb in plist]
    for v in SAM_VARIANTS:
        sync(); t0 = time.perf_counter()
        gt_t = torch.from_numpy(gt_np).to(DEVICE)
        emb = esam_embed(ESAM[v], img)
        piou, miou, _ = esam_decode_prompts(ESAM[v], emb, [p[2] for p in plist], H, W, gt_t, [p[1] for p in plist])
        del emb, gt_t; sync(); e6_sec[v] += time.perf_counter() - t0
        best = piou.argmax(1)
        for j, (key, k, _, _) in enumerate(plist):
            iou = float(miou[j, best[j]])
            e6_rows.append(dict(image_id=i, gt_idx=k, method=SAM_NAME[v], prompt=key, prompt_name=E6_PROMPTS[key][0],
                                group=E6_PROMPTS[key][1], cls=CLASSES[g["cls"][k]], area=g["area"][k], size=size_bin(g["area"][k]),
                                fill=float(fill[k]), iou=iou, dice=2 * iou / (1 + iou), out_idx=int(best[j]),
                                pred_iou=float(piou[j, best[j]]), iou_m0=float(miou[j, 0]), iou_m1=float(miou[j, 1]),
                                iou_m2=float(miou[j, 2]), prompt_box_iou=pbox_iou[j]))
e6 = pd.DataFrame(e6_rows)
e6["fill_bin"] = pd.cut(e6["fill"], FILL_BINS, labels=FILL_LABELS, right=False).astype(str)
e6.to_csv(OUT / "E6_prompt_per_object.csv", index=False)
free_gpu()
t_all = time.perf_counter() - t_all
n_obj = e6[e6.prompt == "P1"].groupby("method").size().max() if len(e6) else 0
print(f"E6: {len(e6)} rows · {n_obj} GT objects × {len(E6_PROMPTS)} prompt types · {len(SAM_VARIANTS)} models · "
      f"wall time {t_all / 60:.1f} min (" + ", ".join(f"{SAM_NAME[v]} {e6_sec[v]:.0f}s GPU" for v in SAM_VARIANTS) + ")")

# sanity checks: P1 must reproduce E2 (same box prompt), Y1 must reproduce E3 (same YOLO boxes)
chk = e6[e6.prompt == "P1"].merge(e2, on=["image_id", "gt_idx", "method"], suffixes=("", "_e2"))
print(f"sanity P1 vs E2: {len(chk)} objects, max |ΔIoU| = {np.abs(chk.iou - chk.iou_e2).max():.2e}")
if len(e3):
    chk = e6[e6.prompt == "Y1"].merge(e3, on=["image_id", "gt_idx", "method"], suffixes=("", "_e3"))
    print(f"sanity Y1 vs E3: {len(chk)} objects, max |ΔIoU| = {np.abs(chk.iou - chk.iou_e3).max():.2e}")

# %%
# ---- E6 tables ----
def e6_summ(d):
    return {"mIoU": d.iou.mean(), "mDice": d.dice.mean(), "IoU≥0.5": (d.iou >= 0.5).mean(), "IoU≥0.75": (d.iou >= 0.75).mean()}

E6_MATCHED = set(map(tuple, e6.loc[e6.prompt == "Y1", ["image_id", "gt_idx"]].drop_duplicates().values))
e6["yolo_matched"] = [(a, b) in E6_MATCHED for a, b in zip(e6.image_id, e6.gt_idx)]

def e6_block(df, keys, subset):
    rows = []
    for key in keys:
        r = {"prompt": E6_PROMPTS[key][0], "key": key, "group": E6_PROMPTS[key][1], "objects": subset}
        for v in SAM_VARIANTS:
            d = df[(df.prompt == key) & (df.method == SAM_NAME[v])]
            r.update({f"{m} {E6_TAG[v]}": x for m, x in e6_summ(d).items()})
        r["n objects"] = len(d)
        rows.append(r)
    return rows

E6_ALL_KEYS = E6_MAIN + ["J05", "J10", "J20", "J05c", "J10c", "J20c"]
E6_SUB_KEYS = ["P1", "P3", "P5", "Y1", "Y3"]
e6_overall = pd.DataFrame(e6_block(e6, E6_ALL_KEYS, "all GT objects")
                          + e6_block(e6[e6.yolo_matched], E6_SUB_KEYS, "YOLO-matched objects"))
e6_overall.round(3).to_csv(OUT / "TABLE_E6_prompt_overall.csv", index=False)
display(e6_overall.round(3))

def e6_by(col, order):
    d = e6[e6.prompt.isin(E6_MAIN)]
    t = d.pivot_table(index=["method", "prompt"], columns=col, values="iou", aggfunc="mean")
    order = [o for o in order if o in t.columns]
    t = t[order]
    n = d[(d.prompt == "P1") & (d.method == SAM_NAME[SAM_VARIANTS[0]])][col].value_counts()
    t.columns = [f"{c} (n={int(n.get(c, 0))})" for c in order]
    t = t.reindex([(SAM_NAME[v], k) for v in SAM_VARIANTS for k in E6_MAIN])
    t.insert(0, "prompt_name", [E6_PROMPTS[k][0] for _, k in t.index])
    return t

e6_cls = e6_by("cls", CLASSES); e6_size = e6_by("size", ["small", "medium", "large"]); e6_fill = e6_by("fill_bin", FILL_LABELS)
e6_cls.round(3).to_csv(OUT / "TABLE_E6_prompt_by_class.csv")
e6_size.round(3).to_csv(OUT / "TABLE_E6_prompt_by_size.csv")
e6_fill.round(3).to_csv(OUT / "TABLE_E6_prompt_by_fill.csv")
display(e6_fill.round(3)); display(e6_size.round(3))

# box-quality robustness: IoU vs jitter level (0 % = exact GT box), box alone and box + centre point
jrows = []
for lvl, kb, kc in [(0, "P1", "P3"), (5, "J05", "J05c"), (10, "J10", "J10c"), (20, "J20", "J20c")]:
    r = {"box prompt": f"GT box ±{lvl}%", "jitter %": lvl, "objects": "all GT objects",
         "mean box IoU vs GT": e6[(e6.prompt == kb)].prompt_box_iou.mean()}
    for v in SAM_VARIANTS:
        r[f"mIoU box {E6_TAG[v]}"] = e6[(e6.prompt == kb) & (e6.method == SAM_NAME[v])].iou.mean()
        r[f"mIoU box+centre {E6_TAG[v]}"] = e6[(e6.prompt == kc) & (e6.method == SAM_NAME[v])].iou.mean()
    r["n objects"] = int((e6.prompt == kb).sum() / len(SAM_VARIANTS)); jrows.append(r)
if E6_MATCHED:
    r = {"box prompt": "YOLO box (matched)", "jitter %": np.nan, "objects": "YOLO-matched objects",
         "mean box IoU vs GT": e6[e6.prompt == "Y1"].prompt_box_iou.mean()}
    for v in SAM_VARIANTS:
        r[f"mIoU box {E6_TAG[v]}"] = e6[(e6.prompt == "Y1") & (e6.method == SAM_NAME[v])].iou.mean()
        r[f"mIoU box+centre {E6_TAG[v]}"] = e6[(e6.prompt == "Y3") & (e6.method == SAM_NAME[v])].iou.mean()
    r["n objects"] = len(E6_MATCHED); jrows.append(r)
e6_jit = pd.DataFrame(jrows)
e6_jit.round(3).to_csv(OUT / "TABLE_E6_box_jitter.csv", index=False)
display(e6_jit.round(3))

# multimask output selection: predicted-IoU pick (used everywhere) vs a fixed output index vs oracle best of 3
mrows = []
for key in E6_MAIN:
    for v in SAM_VARIANTS:
        d = e6[(e6.prompt == key) & (e6.method == SAM_NAME[v])]
        mrows.append({"prompt": E6_PROMPTS[key][0], "model": SAM_NAME[v], "pick by predicted IoU": d.iou.mean(),
                      "always output 0": d.iou_m0.mean(), "always output 1": d.iou_m1.mean(), "always output 2": d.iou_m2.mean(),
                      "oracle best of 3": d[["iou_m0", "iou_m1", "iou_m2"]].max(axis=1).mean()})
e6_mm = pd.DataFrame(mrows)
e6_mm.round(3).to_csv(OUT / "TABLE_E6_multimask_selection.csv", index=False)
display(e6_mm.round(3))

# %%
# ---- E6 figures ----
sam_methods = [SAM_NAME[v] for v in SAM_VARIANTS]
ov_all = e6_overall[e6_overall.objects == "all GT objects"].set_index("key")
ov_sub = e6_overall[e6_overall.objects == "YOLO-matched objects"].set_index("key")

def e6_bars(ax, keys, tab, title):
    x = np.arange(len(keys)); w = 0.8 / len(SAM_VARIANTS)
    for j, v in enumerate(SAM_VARIANTS):
        vals = tab.loc[keys, f"mIoU {E6_TAG[v]}"].values.astype(float)
        bars = ax.bar(x + (j - (len(SAM_VARIANTS) - 1) / 2) * w, vals, w * 0.92, label=SAM_NAME[v], color=METHOD_COLORS[SAM_NAME[v]])
        for b_, v_ in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v_ + 0.01, f"{v_:.3f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, [E6_PROMPTS[k][0].replace(" + ", "\n+ ") for k in keys], fontsize=8)
    ax.set_ylim(0, 1.25); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel("mean mask IoU"); ax.set_title(title, loc="left")
    ax.legend(fontsize=8, ncol=len(SAM_VARIANTS), loc="upper left")

fig, ax = plt.subplots(1, 2, figsize=(15, 4.3), gridspec_kw={"width_ratios": [6, 4.2]})
e6_bars(ax[0], E6_MAIN, ov_all, f"E6 · Prompt type vs mask IoU (all {int(ov_all['n objects'].iloc[0])} GT objects)")
ax[0].axvspan(2.5, 5.5, color="#f4f3ee", zorder=0); ax[0].text(4, 1.03, "oracle clicks (upper bound)", ha="center", fontsize=8, color=MUTED)
ax[0].text(1, 1.03, "practical (no GT needed)", ha="center", fontsize=8, color=MUTED)
if len(ov_sub):
    e6_bars(ax[1], E6_SUB_KEYS, ov_sub, f"Same objects, real YOLO boxes (n={int(ov_sub['n objects'].iloc[0])})")
else:
    ax[1].axis("off")
plt.tight_layout(); savefig("fig_E6_prompt_overall.png")

fig, axes = plt.subplots(len(SAM_VARIANTS), 2, figsize=(13, 3.6 * len(SAM_VARIANTS)), squeeze=False)
for r_, v in enumerate(SAM_VARIANTS):
    for c_, (tab, name) in enumerate([(e6_fill, "fill ratio (mask area / box area)"), (e6_size, "object size")]):
        a_ = axes[r_, c_]; t = tab.loc[SAM_NAME[v]].drop(columns="prompt_name")
        x = np.arange(t.shape[1]); w = 0.8 / len(E6_MAIN)
        for j, k in enumerate(E6_MAIN):
            a_.bar(x + (j - (len(E6_MAIN) - 1) / 2) * w, t.loc[k].values.astype(float), w * 0.92, color=E6_PROMPT_COLORS[k], label=E6_PROMPTS[k][0])
        a_.set_xticks(x, [c.replace(" (", "\n(") for c in t.columns]); a_.set_ylim(0, 1.2); a_.set_yticks(np.arange(0, 1.01, 0.2))
        a_.set_ylabel("mean mask IoU"); a_.set_title(f"{SAM_NAME[v]} · IoU by {name}", loc="left")
        if r_ == 0 and c_ == 0:
            a_.legend(fontsize=7.5, ncol=3, loc="upper left")
plt.tight_layout(); savefig("fig_E6_prompt_by_fill_or_size.png")

fig, ax = plt.subplots(figsize=(7.2, 4))
jj = e6_jit[e6_jit["jitter %"].notna()]
for v in SAM_VARIANTS:
    c = METHOD_COLORS[SAM_NAME[v]]
    ax.plot(jj["jitter %"], jj[f"mIoU box {E6_TAG[v]}"], "-o", color=c, label=f"{SAM_NAME[v]} · box")
    ax.plot(jj["jitter %"], jj[f"mIoU box+centre {E6_TAG[v]}"], "--s", color=c, alpha=0.75, label=f"{SAM_NAME[v]} · box + centre point")
ax.set_xticks(jj["jitter %"], [f"±{int(l)}%\n(box IoU {b:.2f})" for l, b in zip(jj["jitter %"], jj["mean box IoU vs GT"])])
ax.set_xlabel("random shift of each box side (% of box width / height)"); ax.set_ylabel("mean mask IoU"); ax.grid(axis="both")
ax.set_title("E6 · Robustness of EfficientSAM to box error", loc="left"); ax.legend(fontsize=7.5)
plt.tight_layout(); savefig("fig_E6_box_jitter.png")

# %%
# ---- E6 qualitative: thin / hard objects, box vs point prompts ----
def e6_crop_box(b, H, W, pad=0.3):          # same crop rule as crop_box in the qualitative-results section
    x0, y0, x1, y1 = b; w, h = x1 - x0, y1 - y0; p = pad * max(w, h) + 8
    return int(max(0, x0 - p)), int(max(0, y0 - p)), int(min(W, x1 + p)), int(min(H, y1 + p))

ref_v = SAM_VARIANTS[-1]; ref = SAM_NAME[ref_v]
p1 = e6[(e6.method == ref) & (e6.prompt == "P1")]
cand = p1[(p1.area >= 40 ** 2) & (p1.fill < 0.4)].sort_values("iou")
if len(cand) < 4:
    cand = p1[p1.area >= 40 ** 2].sort_values("iou")
picks, seen = [], set()
for rw in cand.itertuples():                       # worst box-prompt masks, at most one object per image
    if rw.image_id not in seen:
        picks.append(rw); seen.add(rw.image_id)
    if len(picks) == 4:
        break
E6_SHOW = [("P1", "Box"), ("P4", "Point"), ("P5", "Box + point"), ("P6", "Box + 3 points")]
if picks:
    fig, axes = plt.subplots(len(picks), 1 + len(E6_SHOW), figsize=(2.4 * (1 + len(E6_SHOW)), 2.45 * len(picks)), squeeze=False)
    for r_, rw in enumerate(picks):
        i, k = rw.image_id, rw.gt_idx
        img = load_rgb(i); H, W = img.shape[:2]; g = GT[i]
        x0, y0, x1, y1 = e6_crop_box(g["boxes"][k], H, W)
        th = max(2, int(0.006 * max(x1 - x0, y1 - y0))); rad = max(4, int(0.025 * max(x1 - x0, y1 - y0)))
        emb = esam_embed(ESAM[ref_v], img)
        _, _, mks = esam_decode_prompts(ESAM[ref_v], emb, [E6_PROMPT_CACHE[(i, k, key)] for key, _ in E6_SHOW], H, W, keep_masks=True)
        del emb
        gt_mask = mask_utils.decode(g["rles"][k]).astype(bool)
        panels = [("GT", None, gt_mask)] + [(lab, key, mk) for (key, lab), mk in zip(E6_SHOW, mks)]
        for c_, (lab, key, mk) in enumerate(panels):
            a_ = axes[r_, c_]; a_.axis("off")
            vis = overlay(img, mk, "#1baf7a" if key is None else METHOD_COLORS[ref], thick=th)
            if key in (None, "P1", "P5", "P6"):
                bx = g["boxes"][k].astype(int); cv2.rectangle(vis, tuple(bx[:2]), tuple(bx[2:]), (255, 255, 255), th)
            if key is not None:
                pts, lbl = E6_PROMPT_CACHE[(i, k, key)]
                for (px, py), l in zip(pts, lbl):
                    if l == 1:
                        cv2.circle(vis, (int(px), int(py)), rad, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(vis, (int(px), int(py)), max(2, rad - th), (27, 175, 122), -1, cv2.LINE_AA)
            a_.imshow(vis[y0:y1, x0:x1])
            if key is None:
                a_.set_title(f"GT · {rw.cls}\nfill {rw.fill:.2f}", fontsize=8, color=INK)
            else:
                iou = e6[(e6.image_id == i) & (e6.gt_idx == k) & (e6.method == ref) & (e6.prompt == key)].iou.iloc[0]
                a_.set_title(f"{lab}\nIoU {iou:.2f}", fontsize=8, color=INK2)
    plt.suptitle(f"E6 · {ref}: hardest thin objects for the box prompt, and what oracle clicks add (green dots = positive points)",
                 fontweight="bold", y=1.01)
    plt.tight_layout(); savefig("fig_E6_prompt_examples.png")
free_gpu()

# %% [markdown]
# ### E6 findings
# The bullets below are computed from the E6 tables (all numbers are printed, not typed). *Practical* prompts can be used by the real pipeline; *oracle* points use the GT mask and are an upper bound for "a user clicks on the object" or for a future point generator.

# %%
def _m(key, v, sub=False):
    t = ov_sub if sub else ov_all
    return float(t.loc[key, f"mIoU {E6_TAG[v]}"])

def _fill(key, v, b):
    col = [c for c in e6_fill.columns if c.startswith(b)]
    return float(e6_fill.loc[(SAM_NAME[v], key), col[0]]) if col else float("nan")

def _size(key, v, b):
    col = [c for c in e6_size.columns if c.startswith(b)]
    return float(e6_size.loc[(SAM_NAME[v], key), col[0]]) if col else float("nan")

V = SAM_VARIANTS; vS = V[-1]
best = {v: max(E6_MAIN, key=lambda k: _m(k, v)) for v in V}
best_pr = {v: max(["P1", "P2", "P3"], key=lambda k: _m(k, v)) for v in V}
thin, comp = FILL_LABELS[0].split(" ")[0], FILL_LABELS[-1].split(" ")[0]
gain_thin = {v: _fill(best[v], v, thin) - _fill("P1", v, thin) for v in V}
gain_comp = {v: _fill(best[v], v, comp) - _fill("P1", v, comp) for v in V}
n_thin = int(((e6.prompt == "P1") & (e6.method == SAM_NAME[vS]) & (e6.fill_bin == FILL_LABELS[0])).sum())
mm = e6_mm.set_index(["prompt", "model"])
find = [
    "Best prompt: " + "; ".join(f"{SAM_NAME[v]} → {E6_PROMPTS[best[v]][0]} (mIoU {_m(best[v], v):.3f} vs {_m('P1', v):.3f} for the box alone, "
                                f"{_m(best[v], v) - _m('P1', v):+.3f})" for v in V) + ".",
    "Practical prompts: a single box-centre point is far worse than the box (" + ", ".join(f"{E6_TAG[v]} {_m('P2', v):.3f}" for v in V)
    + "); box + centre point gives " + ", ".join(f"{E6_TAG[v]} {_m('P3', v):.3f} ({_m('P3', v) - _m('P1', v):+.3f})" for v in V)
    + f"; the best prompt without ground truth for {SAM_NAME[vS]} is: {E6_PROMPTS[best_pr[vS]][0]}.",
    f"Is the gain concentrated in thin objects? {'yes' if min(gain_thin.values()) > max(2 * max(gain_comp.values()), 0.01) else 'not clearly'}"
    f" — thin objects (fill < 0.3, n={n_thin}), best prompt vs box: " + ", ".join(
        f"{E6_TAG[v]} {_fill('P1', v, thin):.3f} → {_fill(best[v], v, thin):.3f} ({gain_thin[v]:+.3f})" for v in V)
    + ", compared with " + ", ".join(f"{gain_comp[v]:+.3f}" for v in V) + " on compact objects (fill ≥ 0.6). Small objects: box "
    + ", ".join(f"{E6_TAG[v]} {_size('P1', v, 'small'):.3f} → {_size(best[v], v, 'small'):.3f}" for v in V) + ".",
    "Box quality matters: mask mIoU with ±5/10/20% box jitter = " + "; ".join(
        f"{E6_TAG[v]} " + " / ".join(f"{e6_jit.loc[e6_jit['jitter %'] == l, f'mIoU box {E6_TAG[v]}'].iloc[0]:.3f}" for l in (5, 10, 20)) for v in V)
    + f" (exact box: " + ", ".join(f"{_m('P1', v):.3f}" for v in V) + "); with the jittered box's centre point added at ±20%: "
    + ", ".join(f"{e6_jit.loc[e6_jit['jitter %'] == 20, f'mIoU box+centre {E6_TAG[v]}'].iloc[0]:.3f}" for v in V) + ".",
]
if len(ov_sub):
    yb = float(e6_jit.loc[e6_jit['box prompt'] == 'YOLO box (matched)', 'mean box IoU vs GT'].iloc[0])
    find.append(f"Pipeline (n={int(ov_sub['n objects'].iloc[0])} YOLO-matched objects, mean YOLO box IoU {yb:.2f}): GT box vs YOLO box = "
                + ", ".join(f"{E6_TAG[v]} {_m('P1', v, True):.3f} vs {_m('Y1', v, True):.3f}" for v in V)
                + "; adding the YOLO box centre point → " + ", ".join(f"{E6_TAG[v]} {_m('Y3', v, True):.3f} ({_m('Y3', v, True) - _m('Y1', v, True):+.3f})" for v in V)
                + "; an oracle click on top of the GT box would reach " + ", ".join(f"{E6_TAG[v]} {_m('P5', v, True):.3f}" for v in V) + ".")
find.append("Mask selection: picking the candidate with the highest predicted IoU gives " + ", ".join(
    f"{E6_TAG[v]} {mm.loc[(E6_PROMPTS['P1'][0], SAM_NAME[v]), 'pick by predicted IoU']:.3f}" for v in V)
    + " for the box prompt, vs an oracle best-of-3 of " + ", ".join(f"{mm.loc[(E6_PROMPTS['P1'][0], SAM_NAME[v]), 'oracle best of 3']:.3f}" for v in V)
    + " (single point: " + ", ".join(f"{mm.loc[(E6_PROMPTS['P4'][0], SAM_NAME[v]), 'pick by predicted IoU']:.3f} vs "
                                    f"{mm.loc[(E6_PROMPTS['P4'][0], SAM_NAME[v]), 'oracle best of 3']:.3f}" for v in V) + ").")
print("E6 findings")
for f_ in find:
    print(" •", f_)
with open(OUT / "E6_findings.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join("- " + f_ for f_ in find) + "\n")
