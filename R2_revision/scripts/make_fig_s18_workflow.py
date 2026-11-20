# -*- coding: utf-8 -*-
"""Fig. S18: experimental framework and internal validation (Reviewer 6, comment 50: flowchart readable at print size).

Layout and colours of the R1 flowchart; content updated to Revision 2 (repeated adulteration experiment, tests of the
complete procedures, partitions and hold-outs). Drawn at 174 mm width, body text 8 pt (Arial).
Output: manuscript/REVISE2-MANUSCRIPT/figures/FigS18.{tif,png,pdf}
"""
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'manuscript' / 'REVISE2-MANUSCRIPT' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'pdf.fonttype': 42})
W_MM = 174
EDGE, FILL, LIMIT_FILL, TEXT = '#5B6F86', '#EEF2F6', '#F5F0E8', '#1F2D3D'
BODY, HEAD, TITLE = 8, 9.5, 11
LH = 4.3                                            # line height in mm at 8 pt

TOP = ['Materials: 58 recorded N. speciosa samples; FTR 22, WTR 12, AST 12, SFTR 12',
       'Ground, 60-mesh sieve, sealed storage in a desiccator; three technical readings averaged per sample',
       'NIR and CIELAB measurements; colorimetric TFC, TPC and TSS reference values']
COLS = [
    ('A  Botanical categories',
     [['58 sample IDs', 'NIR PLS-DA; exploratory PCA, PLS-DA and OPLS-DA (SIMCA)'],
      ['Outer: stratified 5-fold × 10 repeats', 'Inner: stratified 5-fold', 'Select 24 preprocessing options + 1–8 LVs'],
      ['Training-only transformations and fitting', 'Inner selection, then locked outer prediction',
       'All 58 samples predicted once per repeat', 'Mean repeat accuracy: 99.48%'],
      ['Pooled nested confusion matrix (580 predictions)', 'Permutation of the complete procedure: p = 0.001 (999)',
       'Supplier and category confounded for AST and SFTR']]),
    ('B  Phytochemical prediction',
     [['58 sample IDs; TFC, TPC and TSS', 'Separate full-spectrum PLSR models'],
      ['Outer: shuffled 5-fold × 10 repeats', 'Inner: 5-fold', 'Select 24 preprocessing options + 1–15 LVs'],
      ['Training-only transformations and fitting', 'Inner selection, then locked outer prediction',
       'No wavelength selection or separate PCA', 'Summarize 10 complete-repeat metrics'],
      ['Nested R²: TFC 0.734, TPC 0.864, TSS 0.851', 'Category-mean baselines: differences between, not within, categories',
       'Limitations of the reference assays remain']]),
    ('C  Repeated adulteration',
     [['22 parent powders × 0, 5, 15, 30 and 50% starch', '110 states; stepwise addition; three repacked readings'],
      ['Outer: parent-grouped 5-fold, fixed plan', 'Also 10 random partitions, batch and supplier hold-out',
       'Inner and stacking: parent-grouped 3-fold'],
      ['Pipeline of the original submission, unchanged', 'Four NIR windows reselected in each training subset',
       'Preprocessing, base models and fusion rule selected inside', 'All states of a parent kept together'],
      ['All 110 states: RMSE 3.45 pp, R² 0.964', 'Random partitions 2.74–4.17 pp; batch hold-out 6.59 pp',
       'Permutation (99) and refit bootstrap (100) of the pipeline', 'Detected from 15% starch; 5% not reliable']]),
]
LIMITS = ['Internal validation only: one collection, one starch product and one instrument; no external test set.',
          'The colour increment over NIR alone is not statistically resolved; holding out a whole measurement batch '
          'degrades the stacked pipeline.',
          'Recorded IDs define the computational groups; independence above the commercial batch cannot be documented.']

COL_W = (W_MM - 2 * 6 - 2 * 6) / 3                    # three columns, 6 mm margins and gaps
CHARS = int(COL_W / 1.62)                              # approximate characters per line at 8 pt Arial


from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
_FP = FontProperties(family=['Arial', 'DejaVu Sans'], size=BODY)


def text_mm(s):
    return TextPath((0, 0), s, prop=_FP).get_extents().width / 72 * 25.4


def wrap(lines, width_mm):
    """Wrap each line to the rendered width of the box (mm)."""
    out = []
    for l in lines:
        cur = ''
        for w in l.split(' '):
            trial = (cur + ' ' + w).strip()
            if cur and text_mm(trial) > width_mm:
                out.append(cur)
                cur = w
            else:
                cur = trial
        out.append(cur)
    return out


def box(ax, x, y_top, w, lines, head=None, fill=FILL):
    h = (len(lines) + (1.25 if head else 0)) * LH + 4
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h, boxstyle='round,pad=0,rounding_size=1.6', fc=fill, ec=EDGE, lw=0.9))
    y = y_top - 2.2
    if head:
        ax.text(x + 2.2, y, head, fontsize=HEAD, fontweight='bold', color=TEXT, va='top')
        y -= 1.25 * LH
    for l in lines:
        ax.text(x + 2.2, y, l, fontsize=BODY, color=TEXT, va='top')
        y -= LH
    return y_top - h


def arrow(ax, x, y0, y1):
    ax.annotate('', xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.1, mutation_scale=8))


# heights first, so that the rows line up across columns
rows = [[wrap(c[1][r], COL_W - 4.4) for c in COLS] for r in range(4)]
row_h = [max(len(x) for x in r) for r in rows]
top_lines = wrap(TOP, W_MM - 12 - 4.4)
lim_lines = wrap(LIMITS, W_MM - 12 - 4.4)
GAP = 6
H = 14 + (len(top_lines) * LH + 4) + GAP + sum((h + (1.25 if i == 0 else 0)) * LH + 4 + GAP for i, h in enumerate(row_h)) \
    + (len(lim_lines) + 1.25) * LH + 4 + 6
fig = plt.figure(figsize=(W_MM / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_MM)
ax.set_ylim(0, H)
ax.axis('off')
ax.text(6, H - 4, 'Experimental framework and internal validation', fontsize=TITLE, fontweight='bold', color=TEXT, va='top')
y = box(ax, 6, H - 12, W_MM - 12, top_lines)
xs = [6 + k * (COL_W + 6) for k in range(3)]
for k in range(3):
    arrow(ax, xs[k] + COL_W / 2, y, y - GAP)
y -= GAP
for r in range(4):
    bottoms = []
    for k in range(3):
        lines = rows[r][k] + [''] * (row_h[r] - len(rows[r][k]))
        bottoms.append(box(ax, xs[k], y, COL_W, lines, head=COLS[k][0] if r == 0 else None))
    y = min(bottoms)
    if r < 3:
        for k in range(3):
            arrow(ax, xs[k] + COL_W / 2, y, y - GAP)
    y -= GAP
box(ax, 6, y, W_MM - 12, lim_lines, head='Interpretation limits', fill=LIMIT_FILL)
for ext, kw in [('png', {'dpi': 600}), ('tif', {'dpi': 600, 'pil_kwargs': {'compression': 'tiff_lzw'}}), ('pdf', {})]:
    fig.savefig(OUT / f'FigS18.{ext}', **kw)
print('Fig. S18 written:', round(W_MM), 'x', round(H), 'mm')
