# -*- coding: utf-8 -*-
"""Reviewer 6, comment 50: Figs. 1, 3, 4 and 5 redrawn at print size (174 mm wide) with legible lettering.

Each panel is drawn directly at its final size from the same data, colours and plotting choices as the R1 figures, so
that no lettering is reduced when the figure is placed: ticks 8 pt, axis labels 9 pt, panel letters 11 pt (Times New
Roman). Fig. 2 (SIMCA output) is not redrawn.

  Fig. 1  A  mean NIR spectra of the 58 samples (script/01_exploratory/plot_spectra.py: colours by category)
          B  circular heatmap of L*, b*, a* (column z-scores, blue-white-red) with the sample dendrogram. The R1 panel was
             drawn in an external tool; here the dendrogram is complete linkage on min-max scaled L*, a*, b* (Euclidean),
             the setting that reproduces the R1 leaf order most closely (Spearman 0.94).
  Fig. 3  A  SNV + second derivative + Savitzky-Golay spectra (mean-1_23_SNV_SD_SG)
          B  nested PLS-DA, 10 complete repeats pooled (R6-36)     C  LDA on CIELAB (50 repeated five-fold evaluations)
          B and C use the plotting function of the R1 panels (row percentages, Blues).
  Fig. 4  box-scatter plots of TSS, TFC, TPC with the post hoc letters (phytochemical_assays/regenerate_fig4_corrected_letters.py)
  Fig. 5  nested PLSR out-of-fold predictions (R1_revision/.../make_figures.py, pred_panel)
Output: manuscript/REVISE2-MANUSCRIPT/figures/Fig{1,3,4,5}.tif (600 dpi, LZW), .png (600 dpi), .pdf (vector)
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'manuscript' / 'REVISE2-MANUSCRIPT' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
W = 174 / 25.4                                   # double-column width, inches
TICK, LABEL, TITLE, LETTER, NOTE = 8, 9, 9, 11, 7.5
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
                     'axes.unicode_minus': False, 'axes.linewidth': 0.8, 'xtick.labelsize': TICK,
                     'ytick.labelsize': TICK, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
                     'xtick.major.size': 3, 'ytick.major.size': 3, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
GROUPS = ['AST', 'FTR', 'SFTR', 'WTR']
SPEC_COL = dict(zip(GROUPS, ['#E69F00', '#56B4E9', '#009E73', '#D55E00']))     # plot_spectra.py, groups sorted


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=600)
    fig.savefig(OUT / f'{name}.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)


def letter(fig, ax, s, dx=-0.055, dy=0.012):
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, s, fontsize=LETTER, fontweight='bold', ha='left', va='bottom')


def spectra(ax, path, ylabel, legend):
    d = pd.read_csv(path)
    wn_cols = [c for c in d.columns if str(c).startswith('wn_')]
    wn = np.array([float(re.sub('wn_', '', c)) for c in wn_cols])
    o = np.argsort(wn)[::-1]
    X = d[wn_cols].to_numpy(float)[:, o]
    for i, g in enumerate(d['group']):
        ax.plot(wn[o], X[i], color=SPEC_COL[g], lw=0.6, alpha=0.7)
    ax.set_xlim(wn.max(), wn.min())
    ax.set_xlabel('Wavenumber/cm$^{-1}$', fontsize=LABEL, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=LABEL, fontweight='bold')
    if legend:
        h = [Line2D([], [], color=SPEC_COL[g], lw=1.4, label=f'{g} (n={int((d.group == g).sum())})') for g in GROUPS]
        ax.legend(handles=h, loc='upper left', fontsize=NOTE, frameon=False, handlelength=1.4, borderaxespad=0.3)
    return ax


# ------------------------------------------------------------------ Fig. 1
def circular_heatmap(ax):
    c = pd.read_csv(ROOT / 'script' / 'data' / 'colorimetry.csv').set_index('sample_id')
    V = c[['L', 'a', 'b']].to_numpy(float)
    mm = (V - V.min(0)) / (V.max(0) - V.min(0))
    Z = linkage(mm, method='complete', metric='euclidean')
    dn = dendrogram(Z, no_plot=True)
    order = dn['leaves']
    zs = (V - V.mean(0)) / V.std(0, ddof=1)
    rings = [('L*', 0), ('b*', 2), ('a*', 1)]                     # inner to outer, as in the R1 panel
    n, gap = len(order), np.deg2rad(24)
    span = 2 * np.pi - gap
    dth = span / n
    theta = gap / 2 + dth * (np.arange(n) + 0.5)
    r0, rw = 0.62, 0.11
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    cmap = plt.get_cmap('bwr')
    for k, (name, col) in enumerate(rings):
        ax.bar(theta, rw, width=dth, bottom=r0 + k * rw, color=cmap(norm(zs[order, col])), edgecolor='black', lw=0.25)
        ax.text(gap / 2 - np.deg2rad(1.5), r0 + (k + 0.5) * rw, name, fontsize=NOTE, ha='right', va='center',
                rotation=-90 + 90, rotation_mode='anchor')
    r_out = r0 + len(rings) * rw + 0.03
    ids = c.index.to_numpy()[order]
    for t, s in zip(theta, ids):
        deg = np.rad2deg(t)
        rot = 90 - deg
        ha = 'left'
        if 180 < deg < 360:
            rot += 180
            ha = 'right'
        ax.text(t, r_out, s, fontsize=6.5, rotation=rot, rotation_mode='anchor', ha=ha, va='center', fontweight='bold')
    # dendrogram inside the rings
    ic, dc = np.array(dn['icoord']), np.array(dn['dcoord'])
    hmax = dc.max()
    for xs, ys in zip(ic, dc):
        th = gap / 2 + (xs - 5) / 10 * dth + dth / 2
        rr = 0.02 + (r0 - 0.04) * (1 - ys / hmax)
        ax.plot([th[0], th[0]], [rr[0], rr[1]], color='black', lw=0.45)
        ax.plot([th[3], th[3]], [rr[3], rr[2]], color='black', lw=0.45)
        tt = np.linspace(th[1], th[2], 30)
        ax.plot(tt, np.full_like(tt, rr[1]), color='black', lw=0.45)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, r_out + 0.2)
    ax.axis('off')
    return cmap, norm


fig = plt.figure(figsize=(W, 3.35))
axA = fig.add_axes([0.075, 0.14, 0.40, 0.80])
spectra(axA, ROOT / 'script' / 'data' / 'nir_spectra.csv', 'Absorbance', legend=True)
axB = fig.add_axes([0.53, 0.0, 0.43, 1.0], projection='polar')
cmap, norm = circular_heatmap(axB)
cax = fig.add_axes([0.93, 0.10, 0.012, 0.22])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[-3, 0, 3])
cb.ax.tick_params(labelsize=7, length=2)
cb.set_label('z-score', fontsize=7)
letter(fig, axA, 'A')
fig.text(0.53, 0.94, 'B', fontsize=LETTER, fontweight='bold', ha='left', va='bottom')
save(fig, 'Fig1')


# ------------------------------------------------------------------ Fig. 3
def confusion(fig, ax, cm, labels, title):
    """Plotting function of the R1 Fig. 3 panels (row percentages, Blues), at print size."""
    row = cm.sum(axis=1, keepdims=True)
    row[row == 0] = 1
    p = cm / row * 100
    im = ax.imshow(p, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Percentage (%)', fontsize=TICK)
    cb.ax.tick_params(labelsize=7)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted class', fontsize=LABEL, fontweight='bold')
    ax.set_ylabel('True class', fontsize=LABEL, fontweight='bold')
    ax.set_title(title, fontsize=TITLE, fontweight='bold')
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f'{p[i, j]:.1f}', ha='center', va='center', fontsize=TICK,
                    color='white' if p[i, j] > 50 else 'black')


nested = pd.read_csv(ROOT / 'R2_revision' / 'R6-36_nested_confusion' / 'nested_confusion_pooled_580.csv', index_col=0)
lda = pd.read_csv(ROOT / 'colorimetry' / 'Color_multi_model_screening_no_permutation' / 'LDA_confusion_matrix_count.csv', index_col=0)
lda.columns = [c.replace('Pred_', '') for c in lda.columns]
lda.index = [i.replace('True_', '') for i in lda.index]
assert list(nested.index) == GROUPS == list(lda.index) == list(lda.columns)

fig = plt.figure(figsize=(W, 5.6))
axA = fig.add_axes([0.085, 0.62, 0.89, 0.34])
spectra(axA, ROOT / 'script' / 'data' / '03_general_parameters' / 'mean-1_24_preprocessed' / 'mean-1_23_SNV_SD_SG.csv',
        'SNV + SD + SG (a.u.)', legend=True)
axA.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
axA.yaxis.get_offset_text().set_fontsize(TICK)
axB = fig.add_axes([0.085, 0.10, 0.33, 0.39])
axC = fig.add_axes([0.585, 0.10, 0.33, 0.39])
confusion(fig, axB, nested.to_numpy(float), GROUPS, 'PLS-DA confusion matrix (%)')
confusion(fig, axC, lda.to_numpy(float), GROUPS, 'LDA confusion matrix (%)')
letter(fig, axA, 'A', dx=-0.07)
letter(fig, axB, 'B', dx=-0.07, dy=0.035)
letter(fig, axC, 'C', dx=-0.07, dy=0.035)
save(fig, 'Fig3')


# ------------------------------------------------------------------ Fig. 4
dz = pd.read_csv(ROOT / 'phytochemical_assays' / 'DZ.CSV')
dz['group'] = dz['group'].astype(str)
let = pd.read_csv(ROOT / 'phytochemical_assays' / 'comment6_assumption_checks' / 'letters_comparison.csv')
LET = {(r['Variable'], r['Group']): str(r['Letter_chosen']) for _, r in let.iterrows()}
BOXCOL = dict(zip(GROUPS, ['#4C78A8', '#59A14F', '#B279A2', '#F28E2B']))
COLS = ['TSS (mg Glc/g)', 'TFC (μg RE/g)', 'TPC (μg GAE/g)']


def box_scatter(ax, col):
    """regenerate_fig4_corrected_letters.py, plot_box_scatter_with_letters, at print size."""
    x = np.arange(len(GROUPS))
    vals = [dz.loc[dz.group == g, col].dropna().to_numpy() for g in GROUPS]
    bp = ax.boxplot(vals, positions=x, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=1.1), boxprops=dict(lw=0.8, color='black'),
                    whiskerprops=dict(lw=0.8, color='black'), capprops=dict(lw=0.8, color='black'))
    for patch, g in zip(bp['boxes'], GROUPS):
        patch.set_facecolor(BOXCOL[g])
        patch.set_alpha(0.45)
    allv = np.concatenate(vals)
    rng_y = allv.max() - allv.min()
    for i, g in enumerate(GROUPS):
        rng = np.random.default_rng(2026 + i)
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(vals[i])), vals[i], s=14, color=BOXCOL[g], edgecolor='white',
                   lw=0.4, alpha=0.9, zorder=3)
        ax.text(i, vals[i].max() + 0.06 * rng_y, LET.get((col, g), ''), ha='center', va='bottom', fontsize=LABEL,
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel(col, fontsize=LABEL)
    ax.set_title(col, fontsize=TITLE, pad=5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, max(hi, allv.max() + 0.18 * rng_y))


fig, axes = plt.subplots(1, 3, figsize=(W, 2.65))
for ax, col in zip(axes, COLS):
    box_scatter(ax, col)
fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=1.2)
fig.text(0.5, 0.01, 'Different letters indicate significant differences (p < 0.05; TSS: Kruskal–Wallis + '
         'Dunn–Bonferroni; TFC, TPC: Welch ANOVA + Games–Howell).', ha='center', va='bottom', fontsize=NOTE)
save(fig, 'Fig4')


# ------------------------------------------------------------------ Fig. 5
NEST = ROOT / 'R1_revision' / 'comment03_validation' / 'nested_PLSR_20260910' / 'results' / 'tables'
PRED = pd.read_csv(NEST / 'outer_OOF_predictions.csv')
MET = pd.read_csv(NEST / 'nested_metrics_summary.csv').set_index(['Target', 'Metric'])
BLUE, ORANGE = '#1f77b4', '#ff7f0e'
UNITS = {'TFC': 'μg RE/g', 'TPC': 'μg GAE/g', 'TSS': 'mg Glc/g'}


def pred_panel(ax, tag):
    """make_figures.py, pred_panel, at print size."""
    d = PRED[PRED.Target.eq(tag)]
    avg = d.groupby('sample_id')[['Observed', 'Predicted']].mean()
    ax.scatter(d.Observed, d.Predicted, s=5, alpha=.21, color=ORANGE, edgecolors='none', label='Outer OOF (10 repeats)')
    ax.scatter(avg.Observed, avg.Predicted, s=10, alpha=.85, color=BLUE, linewidths=.4, edgecolors='white',
               label='Mean OOF per sample')
    lo = min(d.Observed.min(), d.Predicted.min())
    hi = max(d.Observed.max(), d.Predicted.max())
    span = hi - lo
    low, high = lo - .055 * span, hi + .34 * span
    ax.plot([low, high], [low, high], color=BLUE, lw=1.0, alpha=.8)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.text(.04, .97, f'{tag} ({UNITS[tag]})', transform=ax.transAxes, ha='left', va='top', fontsize=TITLE,
            fontweight='bold')
    rm, r2 = MET.loc[(tag, 'RMSE')], MET.loc[(tag, 'R2')]
    ax.text(.04, .87, f"Nested RMSE: {rm['Mean']:.2f} ± {rm['SD']:.2f}\nNested $R^2$: {r2['Mean']:.3f} ± {r2['SD']:.3f}",
            transform=ax.transAxes, va='top', fontsize=NOTE, linespacing=1.25)
    ax.set_xlabel(f'Observed ({UNITS[tag]})', fontsize=LABEL)
    ax.set_ylabel(f'Predicted ({UNITS[tag]})', fontsize=LABEL)
    ax.legend(loc='lower right', fontsize=NOTE - 1, frameon=False, handletextpad=0.2, borderaxespad=0.2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ticks = [t for t in matplotlib.ticker.MaxNLocator(5).tick_values(low, high) if low <= t <= high]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect('equal')


fig, axes = plt.subplots(1, 3, figsize=(W, 2.3))
for ax, tag in zip(axes, ['TFC', 'TPC', 'TSS']):
    pred_panel(ax, tag)
fig.tight_layout(w_pad=1.0)
save(fig, 'Fig5')
print('figures written to', OUT)
