"""Final presentation (Thai) built from the Colab run's outputs.
   final/.venv/Scripts/python.exe final/deck/build_deck.py  →  final/deck/CP461_Final_Litter_EfficientSAM.pptx
Every number on a slide is read from final/colab_outputs/*.csv (nothing typed by hand except published benchmark values,
which come from TABLE_benchmark_comparison.csv too)."""
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
O = HERE.parent / "colab_outputs"
FIG = O / "figures"
A = HERE / "assets"
OUT = HERE / "CP461_Final_Litter_EfficientSAM.pptx"

# ---------------------------------------------------------------- data (all from the Colab run)
r3 = lambda v: f"{float(v):.3f}"
stats = pd.read_csv(O / "dataset_stats.csv").set_index("split")
e1v1 = pd.read_csv(O / "E1_detection_metrics.csv").set_index("class")
e1v2 = pd.read_csv(O / "E1_detection_metrics_v2.csv").set_index("class")
e7 = pd.read_csv(O / "TABLE_E7_detector_comparison.csv").set_index("detector / setting")
auc = pd.read_csv(O / "TABLE_E7_auc.csv")
seg = pd.read_csv(O / "TABLE_main_segmentation_results.csv").set_index("Method")
e6 = pd.read_csv(O / "TABLE_E6_prompt_overall.csv")
fill = pd.read_csv(O / "TABLE_E6_prompt_by_fill.csv")
jit = pd.read_csv(O / "TABLE_E6_box_jitter.csv")
bench = pd.read_csv(O / "TABLE_benchmark_comparison.csv")
rfs = pd.read_csv(O / "TABLE_E1b_repeat_factors.csv", index_col=0)
CLASSES = ["Plastic", "Paper & Cardboard", "Metal", "Glass", "Other Waste"]
TH = {"Plastic": "พลาสติก", "Paper & Cardboard": "กระดาษ", "Metal": "โลหะ", "Glass": "แก้ว", "Other Waste": "อื่น ๆ"}
V1, V1T, V2, V2T = "v1 · conf 0.25", "v1 · per-class thr (val)", "v2 · conf 0.25", "v2 · per-class thr (val)"
s1, s2, s2t = e7.loc[V1], e7.loc[V2], e7.loc[V2T]
aucm = auc[auc["class"] == "macro"].set_index("detector")
segS, segTi, segY = seg.loc["YOLO box → EfficientSAM-S"], seg.loc["YOLO box → EfficientSAM-Ti"], seg.loc["YOLO-seg (one-stage)"]
gtS, gtTi = seg.loc["GT box → EfficientSAM-S"], seg.loc["GT box → EfficientSAM-Ti"]
allg = e6[e6.objects == "all GT objects"].set_index("key")
yolo_m = e6[e6.objects == "YOLO-matched objects"].set_index("key")
tot = stats.loc["total"]
ours = bench[bench.source == "OURS"].set_index("model")
pub = bench[bench.source != "OURS"].set_index("model")
fillS = fill[fill.method == "EfficientSAM-S"].set_index("prompt")
fillTi = fill[fill.method == "EfficientSAM-Ti"].set_index("prompt")
jit_ = jit.set_index("box prompt")

# ---------------------------------------------------------------- design
FONT = "Leelawadee UI"
GREEN = RGBColor(0x24, 0x4F, 0x34)      # dominant: forest
LEAF = RGBColor(0x5E, 0x9B, 0x4E)
MINT = RGBColor(0xEC, 0xF4, 0xEE)
ORANGE = RGBColor(0xE0, 0x5D, 0x26)     # "ours" / accent (same as the notebook figures)
INK = RGBColor(0x1E, 0x28, 0x22)
MUTED = RGBColor(0x5B, 0x66, 0x5F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD5, 0xE2, 0xD8)
RED = RGBColor(0xB4, 0x32, 0x2A)
CLS_COL = [RGBColor(0x2A, 0x78, 0xD6), RGBColor(0xEB, 0x68, 0x34), RGBColor(0x1B, 0xAF, 0x7A), RGBColor(0xED, 0xA1, 0x00), RGBColor(0xE8, 0x7B, 0xA4)]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5


def bg(slide, color):
    f = slide.background.fill
    f.solid(); f.fore_color.rgb = color


def text(slide, x, y, w, h, content, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=None, name=None):
    """content: str | list of paragraphs; paragraph = str | list of runs; run = str | (str, {size,bold,color})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = content if isinstance(content, list) else [content]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        if spacing:
            para.space_after = Pt(spacing)
        runs = p if isinstance(p, list) else [p]
        for r in runs:
            t, o = (r, {}) if isinstance(r, str) else r
            run = para.add_run()
            run.text = t
            f = run.font
            f.name = FONT
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.color.rgb = o.get("color", color)
    return tb


def box(slide, x, y, w, h, fill=WHITE, line=None, radius=0.08, shape=MSO_SHAPE.ROUNDED_RECTANGLE, shadow=False):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(1)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if not shadow:
        s.shadow.inherit = False
    return s


def image(slide, path, x, y, w, h, align="center"):
    """Fit an image inside (x, y, w, h) keeping its aspect ratio."""
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    dw, dh = iw * s, ih * s
    dx = x + (w - dw) / 2 if align == "center" else x
    return slide.shapes.add_picture(str(path), Inches(dx), Inches(y + (h - dh) / 2), Inches(dw), Inches(dh))


def header(slide, kicker, title, comment=None):
    text(slide, 0.6, 0.38, 9.5, 0.35, kicker.upper(), size=13, bold=True, color=ORANGE)
    text(slide, 0.6, 0.68, 12.1, 0.75, title, size=30, bold=True, color=GREEN)
    if comment:   # motif: pill that ties the slide to an instructor comment
        p = box(slide, SW - 0.6 - 2.55, 0.36, 2.55, 0.42, fill=ORANGE, radius=0.5)
        tf = p.text_frame; tf.margin_left = tf.margin_right = Inches(0.05)
        para = tf.paragraphs[0]; para.alignment = PP_ALIGN.CENTER
        run = para.add_run(); run.text = f"ตอบคอมเมนต์ข้อ {comment}"
        run.font.name, run.font.size, run.font.bold, run.font.color.rgb = FONT, Pt(13), True, WHITE
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def footer(slide, n, dark=False):
    text(slide, 0.6, SH - 0.42, 8, 0.25, "CP461 · ทีม อุดมไปด้วยธาตุเหล็ก · Litter Detection & Segmentation with EfficientSAM",
         size=10, color=(LINE if dark else MUTED))
    text(slide, SW - 1.1, SH - 0.42, 0.5, 0.25, str(n), size=10, color=(LINE if dark else MUTED), align=PP_ALIGN.RIGHT)


def note(slide, who, script):
    slide.notes_slide.notes_text_frame.text = f"[{who}]\n{script}"


def table(slide, x, y, w, rows, col_w, size=13, header_fill=GREEN, row_h=0.42, highlight=None, bold_cols=(), left_cols=(0,)):
    nr, nc = len(rows), len(rows[0])
    t = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr)).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = header_fill
            elif highlight and i in highlight:
                c.fill.fore_color.rgb = RGBColor(0xFD, 0xEB, 0xE2)
            else:
                c.fill.fore_color.rgb = WHITE if i % 2 else MINT
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j in left_cols else PP_ALIGN.CENTER
            run = p.add_run(); run.text = str(v)
            f = run.font
            f.name, f.size = FONT, Pt(size)
            f.bold = i == 0 or (highlight and i in highlight and j in bold_cols)
            f.color.rgb = WHITE if i == 0 else INK
    return t


def style_chart(ch, size=12, legend=True):
    ch.font.name = FONT
    ch.font.size = Pt(size)
    ch.font.color.rgb = MUTED
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(size)
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size - 1)
    ca = ch.category_axis
    ca.format.line.color.rgb = LINE
    ca.tick_labels.font.size = Pt(size)
    ca.has_major_gridlines = False


def stat(slide, x, y, w, value, label, color=GREEN, vsize=40, lsize=13, h=1.55, fill=MINT):
    box(slide, x, y, w, h, fill=fill)
    text(slide, x + 0.25, y + 0.15, w - 0.5, 0.75, value, size=vsize, bold=True, color=color)
    text(slide, x + 0.25, y + 0.15 + vsize / 72 + 0.12, w - 0.5, h - 0.4 - vsize / 72, label, size=lsize, color=INK)


def bullets(slide, x, y, w, h, items, size=15, gap=8, color=INK):
    paras = []
    for it in items:
        if isinstance(it, tuple):
            head, body = it
            paras.append([("■  ", {"color": ORANGE, "size": size - 4}), (head, {"bold": True}), (" " + body, {})])
        else:
            paras.append([("■  ", {"color": ORANGE, "size": size - 4}), it])
    return text(slide, x, y, w, h, paras, size=size, spacing=gap, color=color)


SPEAK = {  # members and roles (from the progress report)
    "krit": ("นายกฤชตฤณ บุญสู่", "67102010157", "Researcher"),
    "nathaphon": ("นายณฐพล สวนใหญ่", "67102010162", "Tester"),
    "anas": ("นายอานัส มีสัตย์", "67102010180", "Developer"),
    "kowit": ("นายโกวิทย์ เพชรนิล", "67102010509", "Researcher / Developer"),
    "thanaphum": ("นายธนภูมิ ด้วงมาก", "67102010515", "Tester"),
}
n = 0


def new(dark=False):
    global n
    n += 1
    s = prs.slides.add_slide(BLANK)
    bg(s, GREEN if dark else WHITE)
    return s


# ================================================================ 1 · title
s = new(dark=True)
text(s, 0.8, 1.15, 11, 0.4, "CP461 COMPUTER VISION · FINAL PROJECT", size=15, bold=True, color=RGBColor(0xF2, 0xA0, 0x7B))
text(s, 0.8, 1.65, 11.5, 2.2, ["Automatic Litter Detection and", "Instance Segmentation Using EfficientSAM"], size=40, bold=True, color=WHITE)
text(s, 0.8, 3.75, 11, 0.9, "ตรวจจับ จำแนก และทำ mask ขยะจากภาพถ่ายจริงด้วย YOLO11 + EfficientSAM บนชุดข้อมูล TACO",
     size=20, color=MINT)
for k, key in enumerate(SPEAK):
    x = 0.8 + k * 2.4
    image(s, A / f"member_{key}.png", x, 5.0, 0.95, 0.95)
    text(s, x + 1.05, 5.05, 1.32, 0.95, [[(SPEAK[key][0].replace("นาย", ""), {"bold": True})], SPEAK[key][2]], size=12, color=WHITE)
text(s, 0.8, 6.55, 8, 0.35, "ทีม อุดมไปด้วยธาตุเหล็ก", size=15, bold=True, color=RGBColor(0xF2, 0xA0, 0x7B))
note(s, "กฤชตฤณ · ~10 วินาที", "สวัสดีครับ ทีมอุดมไปด้วยธาตุเหล็ก วันนี้นำเสนอผลงานฉบับสมบูรณ์ ระบบตรวจจับและทำ mask ขยะอัตโนมัติด้วย YOLO11 ร่วมกับ EfficientSAM")

# ================================================================ 2 · team
s = new()
header(s, "Team members", "สมาชิกและหน้าที่")
JOBS = {
    "krit": "ค้นคว้า paper EfficientSAM และงานที่เกี่ยวข้อง · จัดกลุ่ม class ของ TACO · นำเสนอปัญหาและข้อมูล",
    "nathaphon": "ทดสอบ detector v1 vs v2 · วัด E7 (macro-F1, κ, MCC, ROC-AUC) · confusion matrix",
    "anas": "เขียน pipeline YOLO → EfficientSAM · เทรน detector v2 (960 px + RFS) · notebook บน Colab",
    "kowit": "หา benchmark จาก paper · ทำ web app (Gradio) · สรุปผลและงานต่อยอด",
    "thanaphum": "ทดสอบ segmentation (E2–E5) · prompt study E6 · คัดตัวอย่างถูก/ผิด",
}
for k, key in enumerate(SPEAK):
    x = 0.6 + k * 2.47
    box(s, x, 1.75, 2.27, 4.95, fill=MINT)
    image(s, A / f"member_{key}.png", x + 0.43, 1.98, 1.4, 1.4)
    nm, sid, role = SPEAK[key]
    text(s, x + 0.06, 3.55, 2.15, 0.3, role.upper().replace("RESEARCHER / DEVELOPER", "RESEARCHER / DEV"), size=11, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
    text(s, x + 0.12, 3.88, 2.03, 0.62, [[(nm, {"bold": True})], sid], size=13, color=INK, align=PP_ALIGN.CENTER)
    text(s, x + 0.18, 4.7, 1.91, 1.95, JOBS[key], size=13, color=MUTED)
footer(s, n)
note(s, "กฤชตฤณ · ~10 วินาที", "ทีมเรามี 5 คน แบ่งเป็น researcher, developer และ tester ตามที่แสดง และทุกคนจะนำเสนอส่วนที่ตัวเองรับผิดชอบครับ")

# ================================================================ 3 · problem + objectives
s = new()
header(s, "Problem & objectives", "ปัญหาและวัตถุประสงค์")
box(s, 0.6, 1.75, 4.7, 4.95, fill=GREEN)
text(s, 0.95, 2.0, 4.0, 0.4, "ปัญหา", size=18, bold=True, color=RGBColor(0xF2, 0xA0, 0x7B))
text(s, 0.95, 2.5, 4.0, 4.0, [
    "ขยะในที่สาธารณะมีรูปร่าง ขนาด สี และพื้นหลังหลากหลาย การตรวจด้วยคนช้าและพลาดง่าย",
    "กรอบสี่เหลี่ยมอย่างเดียวไม่พอสำหรับการคัดแยก ต้องรู้รูปร่าง (mask) ของแต่ละชิ้น",
    [("โจทย์: ", {"bold": True}), "ให้ detector สร้าง box prompt ให้ EfficientSAM อัตโนมัติ แทนคนคลิก"],
], size=15, color=WHITE, spacing=12)
objs = [
    ("1", "ตรวจจับ จำแนก 5 classes และทำ mask ขยะอัตโนมัติ", "P, R, F1, mAP, + macro-F1, MCC, ROC-AUC"),
    ("2", "เปรียบเทียบ EfficientSAM-Ti กับ -S", "IoU, Dice, mask AP, ms/ภาพ"),
    ("3", "ศึกษาผลของ prompt (box / จุด / box+จุด / box คลาดเคลื่อน)", "mIoU ต่อ prompt, class, ขนาด, รูปทรง"),
    ("4", "เทียบ baseline one-stage (YOLO-seg)", "mask AP เทียบความเร็ว"),
    ("5", "ต้นแบบเว็บ: อัปโหลดภาพ → mask + จำนวนขยะ", "ใช้งานได้จริงบน Gradio"),
]
for i, (k, o, m) in enumerate(objs):
    y = 1.75 + i * 1.0
    box(s, 5.6, y, 7.13, 0.86, fill=MINT)
    c = box(s, 5.78, y + 0.18, 0.5, 0.5, fill=ORANGE, shape=MSO_SHAPE.OVAL)
    text(s, 5.78, y + 0.18, 0.5, 0.5, k, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 6.45, y + 0.1, 6.1, 0.4, o, size=15, bold=True)
    text(s, 6.45, y + 0.48, 6.1, 0.32, "วัดด้วย: " + m, size=12, color=MUTED)
footer(s, n)
note(s, "กฤชตฤณ · ~15 วินาที", "ปัญหาคือขยะในภาพจริงมีความหลากหลายมาก และกรอบอย่างเดียวไม่พอ เราจึงใช้ detector สร้าง box prompt ให้ EfficientSAM ทำ mask โดยอัตโนมัติ วัตถุประสงค์มี 5 ข้อตามนี้ ข้อ 3 คือการศึกษาผลของ prompt ที่อาจารย์ถามมา")

# ================================================================ 4 · what changed (the 4 comments)
s = new()
header(s, "Since the progress report", "คอมเมนต์จากอาจารย์ และสิ่งที่เราแก้")
CMT = [
    ("1", "“ศึกษาผลของ prompt” หมายถึงอะไร", f"ทดลอง E6: เปลี่ยนชนิด prompt 6 แบบ + box คลาดเคลื่อน ±5/10/20% บน {int(stats.loc['test', 'objects'])} ชิ้น วัด mIoU แยกตาม class ขนาด และรูปทรง", "สไลด์ 12"),
    ("2", "benchmark คืออะไร สูงหรือต่ำกว่า", "เทียบกับ TACO paper, งาน TACO อื่น 3 ชิ้น และ paper EfficientSAM พร้อมบอกว่าสูง/ต่ำกว่า", "สไลด์ 14"),
    ("3", "confusion matrix ยังไม่ดี", "เทรน detector v2: ภาพ 960 px + repeat-factor sampling ให้ class หายากถูกเห็นบ่อยขึ้น และลอง per-class threshold", "สไลด์ 8–10"),
    ("4", "class imbalance ใช้ accuracy อย่างเดียวไม่พอ", "เพิ่ม balanced accuracy, macro-F1, Cohen’s κ, MCC, ROC-AUC และ PR-AUC ราย class", "สไลด์ 7"),
]
for i, (k, q, a, where) in enumerate(CMT):
    x, y = 0.6 + (i % 2) * 6.17, 1.75 + (i // 2) * 2.5
    box(s, x, y, 5.97, 2.3, fill=MINT)
    p = box(s, x + 0.25, y + 0.25, 0.55, 0.55, fill=ORANGE, shape=MSO_SHAPE.OVAL)
    text(s, x + 0.25, y + 0.25, 0.55, 0.55, k, size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.98, y + 0.3, 4.8, 0.5, q, size=16, bold=True, color=GREEN)
    text(s, x + 0.98, y + 0.85, 4.75, 1.1, a, size=13, color=INK)
    text(s, x + 0.98, y + 1.88, 4.75, 0.3, "→ " + where, size=12, bold=True, color=ORANGE)
footer(s, n)
note(s, "กฤชตฤณ · ~15 วินาที", "จาก progress report อาจารย์ให้คอมเมนต์ 4 ข้อ เราแก้ครบทุกข้อ และจะมีป้ายสีส้มบอกที่มุมขวาบนของสไลด์ว่าสไลด์ไหนตอบคอมเมนต์ข้อไหน")

# ================================================================ 5 · data
s = new()
header(s, "Data · TACO", "ข้อมูล: 1,500 ภาพ · 4,784 ชิ้น · 60 → 5 classes")
image(s, FIG / "fig_class_examples.png", 0.6, 1.6, 7.3, 4.45)
text(s, 0.6, 6.15, 7.3, 0.6, "ตัวอย่าง 3 ชิ้นต่อ class พร้อม polygon mask จาก annotation · แบ่ง train / val / test = "
     f"{int(stats.loc['train', 'images']):,} / {int(stats.loc['val', 'images'])} / {int(stats.loc['test', 'images'])} ภาพ", size=12, color=MUTED)
cd = CategoryChartData()
cd.categories = [TH[c] for c in CLASSES]
cd.add_series("ชิ้นขยะ", [int(tot[c]) for c in CLASSES])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(8.2), Inches(1.75), Inches(4.55), Inches(3.4), cd)
ch = gf.chart
style_chart(ch, legend=False)
ch.has_title = True
ch.chart_title.text_frame.text = "จำนวนชิ้นต่อ class (ทั้งชุด)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = INK
ch.category_axis.reverse_order = True
pl = ch.plots[0]
pl.gap_width = 45
pl.has_data_labels = True
pl.data_labels.font.size = Pt(12)
pl.data_labels.font.color.rgb = INK
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
for i, pt in enumerate(pl.series[0].points):
    pt.format.fill.solid(); pt.format.fill.fore_color.rgb = CLS_COL[i]
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
share_p = tot["Plastic"] / tot["objects"]
share_g = tot["Glass"] / tot["objects"]
stat(s, 8.2, 5.3, 4.55, f"{share_p:.0%}  vs  {share_g:.0%}", "Plastic เทียบ Glass: ข้อมูลไม่สมดุลมาก → ต้องวัดแบบ macro / balanced", color=ORANGE, vsize=28, h=1.4)
footer(s, n)
note(s, "กฤชตฤณ · ~15 วินาที", f"ข้อมูลคือ TACO ภาพขยะจริง 1,500 ภาพ เรารวม 60 หมวดเป็น 5 classes ตามวัสดุ จุดสำคัญคือข้อมูลไม่สมดุล Plastic มี {share_p:.0%} ของทั้งหมด ส่วน Glass มีแค่ {share_g:.0%} ซึ่งเป็นเหตุผลที่ accuracy อย่างเดียวไม่พอ")

# ================================================================ 6 · method
s = new()
header(s, "Method", "ขั้นตอนของระบบ และวิธีที่นำมาเปรียบเทียบ")
steps = [("ภาพขยะ", "TACO photo"), ("YOLO11s detector", "เทรนเอง 5 classes\nv2: 960 px + RFS"), ("Box prompt", "กรอบ = มุม 2 จุด\n(label 2, 3)"),
         ("EfficientSAM", "Ti 10 M / S 26 M\npretrained, zero-shot"), ("ผลลัพธ์", "class + mask\n+ จำนวนขยะ")]
for i, (h, d) in enumerate(steps):
    x = 0.6 + i * 2.5
    hl = i in (1, 3)
    box(s, x, 1.75, 2.1, 1.55, fill=(GREEN if hl else MINT))
    text(s, x + 0.12, 1.88, 1.86, 0.4, h, size=15, bold=True, color=(WHITE if hl else GREEN), align=PP_ALIGN.CENTER)
    text(s, x + 0.12, 2.33, 1.86, 0.9, d, size=12, color=(MINT if hl else MUTED), align=PP_ALIGN.CENTER)
    if i < 4:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.14), Inches(2.37), Inches(0.32), Inches(0.32))
        a.fill.solid(); a.fill.fore_color.rgb = ORANGE; a.line.fill.background()
cmp_ = [
    ("เปรียบเทียบ segmentation (3 วิธี)", ["YOLO → EfficientSAM-Ti", "YOLO → EfficientSAM-S", "YOLO11s-seg (one-stage baseline)"]),
    ("เปรียบเทียบ detector (4 แบบ)", ["v1 (progress): 640 px", "v2 (final): 960 px + repeat-factor sampling", "แต่ละตัว × threshold รวม 0.25 / per-class (จูนบน val)"]),
    ("Repeat-factor sampling (RFS)", [f"ภาพที่มี class หายากถูกเห็นซ้ำ r = max(1, √(t/f)), t = 0.3", f"Glass ×{rfs.loc['Glass', 'repeat factor']:.2f} · Metal ×{rfs.loc['Metal', 'repeat factor']:.2f} · Paper ×{rfs.loc['Paper & Cardboard', 'repeat factor']:.2f}"]),
]
for i, (h, items) in enumerate(cmp_):
    x = 0.6 + i * 4.11
    box(s, x, 3.65, 3.91, 3.0, fill=WHITE, line=LINE)
    text(s, x + 0.25, 3.85, 3.4, 0.4, h, size=15, bold=True, color=GREEN)
    bullets(s, x + 0.25, 4.35, 3.45, 2.2, items, size=14, gap=8)
footer(s, n)
note(s, "อานัส · ~25 วินาที", "ระบบรับภาพ ส่งเข้า YOLO11s ที่เราเทรนเอง ได้กรอบและ class จากนั้นใช้กรอบเป็น prompt ให้ EfficientSAM ทำ mask ทีละชิ้นโดยไม่ต้อง fine-tune เราเทียบ 3 วิธีทำ mask และ detector 4 แบบ ตัวใหม่ v2 ใช้ภาพ 960 px และ repeat-factor sampling ที่ทำให้ภาพที่มี Glass ถูกเห็นบ่อยขึ้นประมาณ 2 เท่า")

# ================================================================ 7 · metrics (comment 4)
s = new()
header(s, "Metrics", "ทำไม accuracy อย่างเดียวไม่พอ", comment=4)
stat(s, 0.6, 1.75, 3.9, f"{s2['Accuracy (localised objects)']:.3f}", "accuracy แบบเดิม (นับเฉพาะชิ้นที่หาเจอ)\nดูดี แต่ซ่อนชิ้นที่หาไม่เจอ", color=MUTED, vsize=44, h=1.95)
stat(s, 0.6, 3.9, 3.9, f"{s2['Accuracy (all GT objects)']:.3f}", "accuracy เมื่อนับขยะทุกชิ้น\n(หาไม่เจอ = ผิด)", color=ORANGE, vsize=44, h=1.95)
text(s, 0.6, 6.0, 3.9, 0.6, f"detector v2 · test {int(stats.loc['test', 'objects'])} ชิ้น · conf ≥ 0.25", size=12, color=MUTED)
rows = [["Metric", "ทำไมเลือกใช้"],
        ["Balanced accuracy", "ค่าเฉลี่ย recall ทุก class เท่ากัน ไม่ถูก Plastic ครอบงำ"],
        ["Macro P / R / F1", "เฉลี่ยราย class: class หายากมีน้ำหนักเท่า class ใหญ่"],
        ["Cohen’s κ · MCC", "ปรับความบังเอิญ / ใช้ทั้งตาราง confusion"],
        ["ROC-AUC · PR-AUC", "ไม่ขึ้นกับ threshold, PR-AUC เทียบกับ baseline = สัดส่วน class"],
        ["mAP50 · Mask AP", "มาตรฐาน COCO สำหรับ detection / segmentation"],
        ["IoU · Dice · ms/ภาพ", "คุณภาพ mask และความเร็ว"]]
table(s, 4.85, 1.75, 7.88, rows, [2.45, 5.43], size=14, row_h=0.6, left_cols=(0, 1))
footer(s, n)
note(s, "อานัส · ~25 วินาที", f"ตอบคอมเมนต์ข้อ 4: ถ้าดู accuracy แบบเดิมจะได้ {s2['Accuracy (localised objects)']:.2f} ซึ่งดูดี แต่นับแค่ชิ้นที่หาเจอ ถ้านับขยะทุกชิ้นจะเหลือ {s2['Accuracy (all GT objects)']:.2f} เราจึงเพิ่ม balanced accuracy, macro-F1, kappa, MCC และ ROC/PR-AUC ราย class")

# ================================================================ 8 · detection results table (comment 3 & 4)
s = new()
header(s, "Results · detector comparison (test)", "detector v2 ดีขึ้นทุกตัวชี้วัดหลัก แต่ยังไม่มาก", comment=3)
rows = [["วิธี", "Acc (ทุกชิ้น)", "Balanced Acc", "Precision", "Recall", "F1", "ROC-AUC", "MCC", "mAP50"]]
for key, lbl, det in [(V1, "v1 · conf 0.25", "v1"), (V1T, "v1 · per-class thr", "v1"), (V2, "v2 · conf 0.25", "v2"), (V2T, "v2 · per-class thr", "v2")]:
    r = e7.loc[key]
    m50 = (e1v1 if det == "v1" else e1v2).loc["all", "mAP50"]
    rows.append([lbl, r3(r["Accuracy (all GT objects)"]), r3(r["Balanced accuracy"]), r3(r["Precision (macro)"]), r3(r["Recall (macro)"]),
                 r3(r["F1 (macro)"]), r3(r["ROC-AUC (macro)"]), r3(r["MCC"]), r3(m50)])
table(s, 0.6, 1.7, 12.13, rows, [2.45, 1.3, 1.3, 1.15, 1.08, 1.08, 1.25, 1.08, 1.44], size=14, row_h=0.5, highlight={3}, bold_cols=range(9))
text(s, 0.6, 4.28, 12.1, 0.3, "P / R / F1 = macro (เฉลี่ย 5 class) · ROC-AUC ไม่ขึ้นกับ threshold จึงเท่ากันทั้ง 2 แถวของ detector เดียวกัน · mAP50 จาก Ultralytics", size=11, color=MUTED)
d_rec = e1v2.loc["all", "Recall"] - e1v1.loc["all", "Recall"]
cards = [
    (f"{e1v1.loc['all', 'mAP50']:.3f} → {e1v2.loc['all', 'mAP50']:.3f}", "mAP50 v1 → v2", ORANGE),
    (f"{e1v1.loc['all', 'Recall']:.3f} → {e1v2.loc['all', 'Recall']:.3f}", "recall รวม: หาขยะเจอมากขึ้น", ORANGE),
    (f"{e1v1.loc['Glass', 'mAP50']:.3f} → {e1v2.loc['Glass', 'mAP50']:.3f}", "Glass AP50 เพิ่ม 2 เท่า แต่ยังต่ำมาก", RED),
    (f"{s2['Balanced accuracy']:.3f} → {s2t['Balanced accuracy']:.3f}", "per-class threshold: precision ขึ้นแต่ balanced acc ลด → ไม่ใช้", RED),
]
for i, (v, l, col) in enumerate(cards):
    stat(s, 0.6 + i * 3.08, 4.75, 2.9, v, l, color=col, vsize=24, lsize=12, h=1.85)
footer(s, n)
note(s, "ณฐพล · ~25 วินาที",
     f"ตอบคอมเมนต์ข้อ 3 และ 4: เทียบ detector 4 แบบบน test set ด้วย Acc, F1, Precision, Recall, AUC และ MCC v2 ดีกว่า v1 ทุกตัวหลัก mAP50 จาก {e1v1.loc['all', 'mAP50']:.3f} เป็น {e1v2.loc['all', 'mAP50']:.3f} recall เพิ่มขึ้น {d_rec:.2f} "
     f"แต่ต้องยอมรับว่ายังเพิ่มไม่มาก ส่วน per-class threshold ทำให้ precision สูงขึ้น แต่ balanced accuracy ลดลงจาก {s2['Balanced accuracy']:.3f} เหลือ {s2t['Balanced accuracy']:.3f} เพราะ threshold หาชิ้นที่ detector มองไม่เห็นไม่ได้ เราจึงใช้ v2 ที่ conf 0.25")

# ================================================================ 9 · confusion matrix
s = new()
header(s, "Results · confusion matrix (test)", "ก่อน / หลัง: ปัญหาหลักคือหาไม่เจอ ไม่ใช่ทายผิด class", comment=3)
image(s, FIG / "fig_E7_confusion_before_after.png", 0.6, 1.6, 8.3, 4.95)
pb = pd.read_csv(O / "TABLE_E7_v2_conf_0_25_per_class.csv").set_index("class")
pa = pd.read_csv(O / "TABLE_E7_v1_conf_0_25_per_class.csv").set_index("class")
bullets(s, 9.15, 1.75, 3.6, 4.9, [
    ("Plastic", f"recall {pa.loc['Plastic', 'Recall']:.2f} → {pb.loc['Plastic', 'Recall']:.2f} หาเจอมากขึ้นชัดเจน"),
    ("Glass / Other", "77–78% ยังตกไปคอลัมน์ background = หาไม่เจอ (ชิ้นเล็ก โปร่งใส)"),
    ("Paper", f"recall {pa.loc['Paper & Cardboard', 'Recall']:.2f} → {pb.loc['Paper & Cardboard', 'Recall']:.2f} แย่ลง ถูกทายเป็น Plastic 42%"),
    ("สรุป", "สับสนระหว่าง class น้อย ส่วนใหญ่ของ error คือ missed detection → คอขวดคือ detector"),
], size=13, gap=10)
footer(s, n)
note(s, "ณฐพล · ~20 วินาที", "เทียบ confusion matrix ก่อนและหลัง Plastic หาเจอเพิ่มชัดเจน แต่ Glass กับ Other Waste ยังหาไม่เจอเกือบ 80% สังเกตว่าตัวเลขส่วนใหญ่ไปอยู่คอลัมน์ background ไม่ใช่ทายผิด class ปัญหาจริงคือหาไม่เจอ ส่วน Paper แย่ลงเพราะถูกทายเป็น Plastic")

# ================================================================ 10 · ROC / PR
s = new()
header(s, "Results · ROC curve (object level, one-vs-rest)", "ROC-AUC และ PR-AUC ราย class", comment=4)
image(s, FIG / "fig_E7_roc_curves.png", 0.6, 1.6, 7.6, 4.95)
a2 = auc[auc.detector == "v2"].set_index("class")
a1 = auc[auc.detector == "v1"].set_index("class")
rows = [["Class (v2)", "ROC-AUC", "PR-AUC", "baseline"]]
for c in CLASSES:
    rows.append([TH[c], r3(a2.loc[c, "ROC-AUC"]), r3(a2.loc[c, "PR-AUC (AP)"]), r3(a2.loc[c, "PR baseline (prevalence)"])])
rows.append(["macro", r3(a2.loc["macro", "ROC-AUC"]), r3(a2.loc["macro", "PR-AUC (AP)"]), "–"])
table(s, 8.45, 1.75, 4.28, rows, [1.35, 1.0, 1.0, 0.93], size=13, row_h=0.45, highlight={6}, bold_cols=range(4))
text(s, 8.45, 5.05, 4.28, 1.6, [
    f"macro ROC-AUC {aucm.loc['v1', 'ROC-AUC']:.3f} → {aucm.loc['v2', 'ROC-AUC']:.3f} · PR-AUC {aucm.loc['v1', 'PR-AUC (AP)']:.3f} → {aucm.loc['v2', 'PR-AUC (AP)']:.3f}",
    f"Glass ROC-AUC {a2.loc['Glass', 'ROC-AUC']:.2f} ใกล้เส้นสุ่ม (0.5) แต่ Metal {a2.loc['Metal', 'ROC-AUC']:.2f} แยกได้ดี"], size=13, spacing=8)
footer(s, n)
note(s, "ณฐพล · ~20 วินาที", f"ROC curve ระดับวัตถุ แยกทีละ class macro ROC-AUC เพิ่มจาก {aucm.loc['v1', 'ROC-AUC']:.3f} เป็น {aucm.loc['v2', 'ROC-AUC']:.3f} Metal แยกได้ดีที่สุด ส่วน Glass ใกล้เส้นสุ่ม ตรงกับ confusion matrix")

# ================================================================ 11 · segmentation methods
s = new()
header(s, "Results · segmentation (test, detector v2)", "YOLO + EfficientSAM ได้ mask ดีกว่า YOLO-seg")
cd = CategoryChartData()
cd.categories = ["YOLO → EfficientSAM-Ti", "YOLO → EfficientSAM-S", "YOLO11s-seg (one-stage)"]
cd.add_series("Mask AP", [float(segTi["Mask AP"]), float(segS["Mask AP"]), float(segY["Mask AP"])])
cd.add_series("Mask AP50", [float(segTi["Mask AP50"]), float(segS["Mask AP50"]), float(segY["Mask AP50"])])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.65), Inches(6.2), Inches(4.0), cd)
ch = gf.chart
style_chart(ch, size=12)
pl = ch.plots[0]; pl.gap_width = 60; pl.overlap = -10
pl.has_data_labels = True; pl.data_labels.number_format = "0.000"; pl.data_labels.number_format_is_linked = False
pl.data_labels.font.size = Pt(11); pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
for ser, col in zip(pl.series, [ORANGE, RGBColor(0xF2, 0xA0, 0x7B)]):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
ch.value_axis.maximum_scale = 0.3; ch.value_axis.major_unit = 0.05
ch.value_axis.tick_labels.number_format = "0.00"; ch.value_axis.tick_labels.number_format_is_linked = False
rows = [["วิธี", "Mask AP", "mIoU*", "ms/ภาพ", "Params"]]
for nm, r in [("YOLO → ESAM-Ti", segTi), ("YOLO → ESAM-S", segS), ("YOLO11s-seg", segY)]:
    rows.append([nm, r3(r["Mask AP"]), r3(r["mIoU"]), f"{r['ms / image']:.0f}", f"{r['Params (M)']:.1f} M"])
table(s, 7.1, 1.75, 5.63, rows, [1.85, 0.98, 0.9, 0.95, 0.95], size=13, row_h=0.5, highlight={2}, bold_cols=range(5))
text(s, 7.1, 3.85, 5.63, 0.3, "* mIoU บนชิ้นที่ detector หาเจอ · Params รวม detector", size=11, color=MUTED)
gain = segS["Mask AP"] / segY["Mask AP"] - 1
slow = segS["ms / image"] / segY["ms / image"]
bullets(s, 7.1, 4.3, 5.63, 2.3, [
    ("EfficientSAM-S ดีที่สุด", f"mask AP สูงกว่า YOLO-seg {gain:.0%} ({segS['Mask AP']:.3f} vs {segY['Mask AP']:.3f})"),
    ("แลกกับความเร็ว", f"ช้ากว่า {slow:.0f} เท่า ({segS['ms / image']:.0f} vs {segY['ms / image']:.0f} ms) · Ti เร็วกว่า S {segS['ms / image'] / segTi['ms / image']:.1f} เท่า แต่ AP ต่ำกว่าแค่ {segS['Mask AP'] - segTi['Mask AP']:.3f}"),
    ("detector ดีขึ้น → mask ดีขึ้น", f"mask AP (S) 0.184 ใน progress → {segS['Mask AP']:.3f} ด้วย v2"),
], size=13, gap=8)
footer(s, n)
note(s, "ธนภูมิ · ~20 วินาที", f"ด้าน segmentation ใช้ detector ตัวเดียวกัน EfficientSAM-S ได้ mask AP {segS['Mask AP']:.3f} สูงกว่า YOLO-seg {gain:.0%} แต่ช้ากว่าประมาณ {slow:.0f} เท่า และเมื่อ detector ดีขึ้น mask AP ก็ขึ้นจาก 0.184 เป็น {segS['Mask AP']:.3f}")

# ================================================================ 12 · prompt study (comment 1)
s = new()
header(s, "E6 · effect of the prompt (710 test objects)", "ศึกษาผลของ prompt: box ดีที่สุดในทางปฏิบัติ", comment=1)
text(s, 0.6, 1.6, 12.1, 0.45, [[("ความหมาย: ", {"bold": True, "color": GREEN}),
     "ป้อน prompt แบบต่าง ๆ ให้ EfficientSAM กับวัตถุชิ้นเดียวกัน แล้ววัดว่า mask เปลี่ยนไปแค่ไหน (prompt = สิ่งที่บอกโมเดลว่าวัตถุอยู่ตรงไหน)"]], size=14)
keys = ["P1", "P2", "P3", "P4", "P5", "P6"]
lbl = {"P1": "box", "P2": "จุดกลาง box", "P3": "box + จุดกลาง", "P4": "จุดบนวัตถุ*", "P5": "box + จุด*", "P6": "box + 3 จุด*"}
cd = CategoryChartData()
cd.categories = [lbl[k] for k in keys]
cd.add_series("EfficientSAM-Ti", [float(allg.loc[k, "mIoU Ti"]) for k in keys])
cd.add_series("EfficientSAM-S", [float(allg.loc[k, "mIoU S"]) for k in keys])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(2.15), Inches(7.3), Inches(4.3), cd)
ch = gf.chart
style_chart(ch, size=12)
pl = ch.plots[0]; pl.gap_width = 55; pl.overlap = -5
pl.has_data_labels = True; pl.data_labels.number_format = "0.00"; pl.data_labels.number_format_is_linked = False
pl.data_labels.font.size = Pt(10); pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
for ser, col in zip(pl.series, [GREEN, ORANGE]):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
ch.value_axis.minimum_scale = 0.5; ch.value_axis.maximum_scale = 0.85
ch.value_axis.tick_labels.number_format = "0.00"; ch.value_axis.tick_labels.number_format_is_linked = False
ch.has_title = True
ch.chart_title.text_frame.text = "mIoU ของ mask ต่อชนิด prompt (GT)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = INK
text(s, 0.6, 6.5, 7.3, 0.3, "* จุด oracle = จุดที่อยู่ลึกที่สุดในวัตถุ (ใช้เฉลย) ใช้เป็นขอบบน", size=11, color=MUTED)
pt_loss_ti = allg.loc["P1", "mIoU Ti"] - allg.loc["P2", "mIoU Ti"]
pt_loss_s = allg.loc["P1", "mIoU S"] - allg.loc["P2", "mIoU S"]
bullets(s, 8.2, 2.2, 4.55, 4.5, [
    ("จุดอย่างเดียวแย่", f"mIoU ลด {pt_loss_s:.2f}–{pt_loss_ti:.2f} เทียบกับ box"),
    ("เพิ่มจุดบน box แทบไม่ช่วย", f"ดีสุด +{allg.loc['P6', 'mIoU Ti'] - allg.loc['P1', 'mIoU Ti']:.3f} (Ti, 3 จุด oracle)"),
    ("วัตถุบางยาว (n=60) ยังยาก", f"box ได้ {fillS.loc['P1', 'thin (<0.3) (n=60)']:.2f} เทียบชิ้นทึบ {fillS.loc['P1', 'compact (≥0.6) (n=335)']:.2f} · เพิ่มจุดไม่แก้"),
    ("box คลาดเคลื่อนไม่เป็นไร", f"±10%: {jit_.loc['GT box ±10%', 'mIoU box S']:.3f} vs {jit_.loc['GT box ±0%', 'mIoU box S']:.3f} (S) · box จาก YOLO เสียแค่ {yolo_m.loc['P1', 'mIoU S'] - yolo_m.loc['Y1', 'mIoU S']:.3f}"),
], size=13, gap=10)
footer(s, n)
note(s, "ธนภูมิ · ~25 วินาที",
     "ตอบคอมเมนต์ข้อ 1: การศึกษาผลของ prompt คือป้อน prompt หลายแบบให้วัตถุชิ้นเดียวกัน แล้ววัดคุณภาพ mask เราทดลอง 6 แบบบน 710 ชิ้น "
     f"ผลคือ box ดีที่สุดในทางปฏิบัติ จุดอย่างเดียวแย่กว่ามาก เพิ่มจุดบน box ช่วยได้ไม่ถึง 0.02 วัตถุบางยาวยังยากอยู่ และ box จาก YOLO ทำให้ mask แย่ลงแค่ประมาณ 0.01 แปลว่าคอขวดไม่ใช่ prompt แต่คือการหาวัตถุให้เจอ")

# ================================================================ 13 · correct / wrong examples
s = new()
header(s, "Results · examples", "ตัวอย่างที่ทำนายถูกและผิด")
image(s, FIG / "fig_detection_correct_vs_wrong.png", 0.6, 1.55, 8.3, 4.95)
bullets(s, 9.15, 1.75, 3.58, 4.8, [
    ("ถูก", "ชิ้นใหญ่ พื้นหลังเรียบ (ขวด กระป๋อง ถุง)"),
    ("ผิด", "ฉากมีชิ้นเล็กจำนวนมาก เศษแก้ว ก้นบุหรี่ บนพื้นรก → หาไม่เจอ (เหลือง)"),
    ("mask แย่", "วัตถุบางยาว เช่น เชือก สายรัด mask ระบายเต็มกรอบ (IoU ต่ำสุดในชุดทดสอบ)"),
    ("สับสน class", "Paper ↔ Plastic และ Other Waste ↔ Plastic เพราะวัสดุหน้าตาคล้ายกัน"),
], size=13, gap=10)
text(s, 0.6, 6.55, 8.3, 0.3, "เขียว = ถูก · แดง = class ผิด / false positive · เหลือง = หาไม่เจอ", size=11, color=MUTED)
footer(s, n)
note(s, "ธนภูมิ · ~15 วินาที", "ตัวอย่างที่ถูกคือชิ้นใหญ่บนพื้นเรียบ ส่วนที่ผิดคือฉากที่มีขยะชิ้นเล็กจำนวนมาก เช่นเศษแก้วบนถนน ซึ่ง detector หาไม่เจอ และ mask ที่แย่ที่สุดคือวัตถุบางยาวอย่างเชือก")

# ================================================================ 14 · benchmark (comment 2)
s = new()
header(s, "Benchmark", "เทียบกับงานที่ตีพิมพ์: สูงหรือต่ำกว่า", comment=2)
B = lambda m: float(pub.loc[m, "value"])
U = lambda m: float(ours.loc[m, "value"])
rows = [["ตัวชี้วัด", "ของเรา", "งานอ้างอิง", "ค่า", "ผล"],
        ["Box mAP50 (TACO หลาย class)", r3(U("YOLO11s v2 (final)")), "Majchrowska 2022 · EfficientDet-D2, 7 classes", r3(B("EfficientDet-D2")), "สูงกว่า ▲"],
        ["Box mAP50 (TACO หลาย class)", r3(U("YOLO11s v2 (final)")), "Saakyan 2026 · D-FINE-S, 59 classes", r3(B("D-FINE-S")), "สูงกว่า ▲ *"],
        ["Box mAP50 (1 class ‘litter’)", r3(U("YOLO11s v2 (final)")), "Córdova 2022 · YOLOv5s ไม่ต้องจำแนก", r3(B("YOLOv5s @640")), "ต่ำกว่า ▼ *"],
        ["Mask AP (TACO)", r3(U("YOLO → EfficientSAM-S")), "TACO paper · Mask R-CNN (±0.016)", r3(B("Mask R-CNN R50")), "สูงกว่าเล็กน้อย ▲"],
        ["mIoU box prompt · ESAM-S", r3(U("EfficientSAM-S")), "EfficientSAM paper (COCO)", r3(B("EfficientSAM-S")), "สูงกว่า ▲"],
        ["mIoU box prompt · ESAM-Ti", r3(U("EfficientSAM-Ti")), "EfficientSAM paper (COCO)", r3(B("EfficientSAM-Ti")), "สูงกว่า ▲"]]
t = table(s, 0.6, 1.7, 12.13, rows, [3.05, 1.2, 4.83, 1.1, 1.95], size=14, row_h=0.52, left_cols=(0, 2))
for i in range(1, len(rows)):
    c = t.cell(i, 4).text_frame.paragraphs[0].runs[0].font
    c.bold = True
    c.color.rgb = RED if "ต่ำกว่า" in rows[i][4] else LEAF
text(s, 0.6, 5.5, 12.1, 0.3, "* ตั้งค่าไม่เหมือนกัน (จำนวน class / ชุดข้อมูลต่างกัน) จึงเทียบได้แค่คร่าว ๆ · แหล่งอ้างอิงเต็มอยู่ใน notebook ส่วน 12.5",
     size=11, color=MUTED)
box(s, 0.6, 5.9, 12.13, 0.8, fill=GREEN)
text(s, 0.85, 5.9, 11.6, 0.8, [[("สรุป: ", {"bold": True, "color": RGBColor(0xF2, 0xA0, 0x7B)}),
     "detection สูงกว่างานที่แบ่ง class ตามวัสดุ · mask AP สูงกว่า TACO paper เล็กน้อย · EfficientSAM บนขยะได้ mIoU สูงกว่าตัวเลขใน paper (บน COCO)"]],
     size=14, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
footer(s, n)
note(s, "โกวิทย์ · ~25 วินาที",
     f"ตอบคอมเมนต์ข้อ 2: benchmark คือผลจากงานที่ตีพิมพ์บน TACO และ paper EfficientSAM detector ของเราได้ mAP50 {U('YOLO11s v2 (final)'):.3f} สูงกว่างานที่แบ่ง 7 class ตามวัสดุซึ่งได้ {B('EfficientDet-D2'):.3f} "
     f"แต่ต่ำกว่างานที่ตรวจแค่ว่าเป็นขยะหรือไม่ ซึ่งง่ายกว่า mask AP {U('YOLO → EfficientSAM-S'):.3f} สูงกว่า Mask R-CNN ใน TACO paper เล็กน้อย และ EfficientSAM บนขยะได้ mIoU สูงกว่าค่าใน paper ด้วย")

# ================================================================ 15 · prototype
s = new()
header(s, "Prototype", "ต้นแบบเว็บแอป (Gradio)")
pic = image(s, A / "app_screenshot.jpg", 0.6, 1.6, 8.1, 4.95)
pic.line.color.rgb = LINE; pic.line.width = Pt(1)
bullets(s, 9.0, 1.75, 3.73, 4.9, [
    ("อัปโหลดภาพ", "ภาพขยะจากกล้องหรือไฟล์"),
    ("ปรับได้", "ค่า confidence และเลือก EfficientSAM-Ti / -S"),
    ("ผลลัพธ์", "กรอบ + ชนิด + ความมั่นใจ + mask สีตาม class + ตารางนับจำนวนต่อประเภท"),
    ("รันได้ 2 แบบ", "เซลล์สุดท้ายของ notebook บน Colab หรือ final/app/app.py บนเครื่อง"),
], size=13, gap=10)
footer(s, n)
note(s, "โกวิทย์ · ~15 วินาที", "นี่คือต้นแบบเว็บแอป อัปโหลดภาพ ระบบตรวจจับ ทำ mask และนับจำนวนขยะแต่ละประเภท ภาพนี้ตรวจเจอ 9 ชิ้น ปรับ confidence และเลือก Ti หรือ S ได้")

# ================================================================ 16 · conclusions
s = new(dark=True)
text(s, 0.6, 0.38, 9.5, 0.35, "CONCLUSIONS", size=13, bold=True, color=RGBColor(0xF2, 0xA0, 0x7B))
text(s, 0.6, 0.68, 12.1, 0.75, "สรุปผล จุดแข็ง จุดอ่อน และงานต่อยอด", size=30, bold=True, color=WHITE)
cols = [
    ("ตอบโจทย์หรือไม่", [f"ตอบได้: ระบบตรวจจับ ทำ mask และนับขยะได้จริง mask คุณภาพสูงเมื่อหาวัตถุเจอ (mIoU {segS['mIoU']:.2f})",
                       f"แต่ detector ยังหาเจอแค่ {e1v2.loc['all', 'Recall']:.0%} ของขยะ (recall)"]),
    ("ดีกว่า paper อ้างอิงไหม", [f"EfficientSAM บนขยะ mIoU {gtS['mIoU']:.3f} / {gtTi['mIoU']:.3f} สูงกว่าค่าใน paper บน COCO (0.769 / 0.757)",
                              "mAP50 สูงกว่างาน 7 classes · mask AP สูงกว่า TACO paper เล็กน้อย"]),
    ("จุดแข็ง", [f"ไม่ต้อง fine-tune EfficientSAM แต่ mask AP สูงกว่า YOLO-seg {gain:.0%}", "box prompt ทนต่อกรอบคลาดเคลื่อน · วัดผลครบหลายมุม"]),
    ("จุดอ่อน", [f"Glass / Other recall ต่ำมาก ({pb.loc['Glass', 'Recall']:.2f} / {pb.loc['Other Waste', 'Recall']:.2f})", f"ช้ากว่า YOLO-seg ~{slow:.0f} เท่า · วัตถุบางยาว mask ยังแย่"]),
]
for i, (h, items) in enumerate(cols):
    x = 0.6 + i * 3.08
    box(s, x, 1.65, 2.9, 3.55, fill=RGBColor(0x2F, 0x63, 0x43))
    text(s, x + 0.22, 1.82, 2.5, 0.4, h, size=15, bold=True, color=RGBColor(0xF2, 0xA0, 0x7B))
    bullets(s, x + 0.22, 2.3, 2.5, 2.85, items, size=14, gap=10, color=WHITE)
box(s, 0.6, 5.4, 12.13, 1.3, fill=WHITE)
text(s, 0.85, 5.52, 11.7, 1.1, [[("งานต่อยอด  ", {"bold": True, "color": ORANGE, "size": 15})],
     "เพิ่มข้อมูล class หายาก (copy-paste augmentation, TACO unofficial) · ตัดภาพเป็น tile (SAHI) ให้เห็นชิ้นเล็ก · detector ใหญ่ขึ้น · "
     "fine-tune decoder ของ EfficientSAM สำหรับวัตถุบางยาว · export ONNX ให้เร็วขึ้น"], size=14, color=INK)
footer(s, n, dark=True)
note(s, "โกวิทย์ · ~25 วินาที",
     "สรุป ระบบตอบโจทย์ได้ ตรวจจับ ทำ mask และนับขยะได้จริง EfficientSAM ทำ mask บนขยะได้ดีเท่าหรือดีกว่าใน paper โดยไม่ต้อง fine-tune "
     "จุดอ่อนคือ detector ยังหา Glass และขยะชิ้นเล็กไม่เจอ และช้ากว่า YOLO-seg งานต่อไปคือเพิ่มข้อมูล class หายาก ใช้ tiling สำหรับชิ้นเล็ก และ fine-tune สำหรับวัตถุบางยาว ขอบคุณครับ")

from pptx.oxml.ns import qn
from lxml import etree


def thai_font(root):
    """python-pptx only sets <a:latin>; Thai glyphs use <a:cs>. Set both (and ea) on every run property."""
    for tag in ("a:rPr", "a:defRPr", "a:endParaRPr"):
        for rpr in root.iter(qn(tag)):
            for t in ("a:latin", "a:ea", "a:cs"):
                el = rpr.find(qn(t))
                if el is None:
                    el = etree.SubElement(rpr, qn(t))
                    # keep schema order: latin, ea, cs must come before sym / hlinkClick / hlinkMouseOver / rtl / extLst
                    for later in ("a:sym", "a:hlinkClick", "a:hlinkMouseOver", "a:rtl", "a:extLst"):
                        nxt = rpr.find(qn(later))
                        if nxt is not None:
                            nxt.addprevious(el)
                            break
                el.set("typeface", FONT)
            lat, ea, cs = (rpr.find(qn(t)) for t in ("a:latin", "a:ea", "a:cs"))
            lat.addnext(ea); ea.addnext(cs)


for sl in prs.slides:
    thai_font(sl.shapes._spTree)
    for shp in sl.shapes:
        if shp.has_chart:
            thai_font(shp.chart._chartSpace)
    thai_font(sl.notes_slide.notes_text_frame._txBody)

prs.save(OUT)
print(f"wrote {OUT} · {n} slides")
