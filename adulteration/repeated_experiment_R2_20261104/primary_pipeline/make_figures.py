# -*- coding: utf-8 -*-
"""Figures for the repeated adulteration experiment (Reviewer 6, comment 50).

Fig. 6  (main text)  primary model = decision-level fusion (stack of NIR windows + CIELAB), all 110 states,
                     fixed five-fold parent plan: observed vs predicted coloured by parent, low-level zoom,
                     residuals, and RMSE of the five approaches with conditional 95 % intervals.
Fig. S_parent        RMSE and bias by parent powder.
Fig. S_partitions    RMSE over random parent-grouped partitions for the five approaches.
Fig. S_perm          permutation null of the complete procedure (drawn only when the run exists).

Output: R2_revision/Figures_R2/ (PNG 600 dpi, SVG, and the plotted values as CSV).
Run: python make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / 'R2_revision' / 'Figures_R2'
OUT.mkdir(parents=True, exist_ok=True)
TASK = 'quantitative_with_zero105'
NAMES = {'color_only': 'Color only', 'nir_full': 'NIR full spectrum', 'feature_fusion': 'Feature fusion',
         'nir_stack': 'NIR stack', 'fusion_stack': 'NIR + color stack'}
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none', 'legend.frameon': False})
CM = 1 / 2.54


def oof(variant):
    return pd.read_csv(HERE / 'runs' / TASK / variant / TASK / 'outer_OOF_predictions.csv')


def save(fig, stem):
    fig.savefig(OUT / (stem + '.png'), dpi=600)
    fig.savefig(OUT / (stem + '.svg'))
    plt.close(fig)


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def label_points(ax, frame, ycol, gap):
    """Label points with the parent name; when two labelled points nearly coincide, put one label above and one below."""
    pts = frame.reset_index(drop=True)
    for k, r in pts.iterrows():
        near = pts[(pts.index != k) & ((pts.Observed - r.Observed).abs() < 2) & ((pts[ycol] - r[ycol]).abs() < gap)]
        dy = 0 if near.empty else (6 if r[ycol] > near[ycol].max() else -6)
        ax.annotate(r.group_id, (r.Observed, r[ycol]), xytext=(5, dy), textcoords='offset points', fontsize=6, va='center')


def fig6():
    p = oof('fusion_stack')
    p['parent_no'] = p.group_id.str[1:].astype(int)
    p = p.sort_values(['parent_no', 'Observed'])
    p[['sample_state_id', 'group_id', 'Outer_fold', 'Observed', 'Predicted', 'Residual']].to_csv(
        OUT / 'Fig6_values.csv', index=False, encoding='utf-8-sig')
    colours = [plt.cm.tab20(i % 20) for i in range(20)] + ['black', 'dimgray']
    markers = ['o'] * 20 + ['s', '^']
    fig, ax = plt.subplots(2, 2, figsize=(17 * CM, 15 * CM))
    for k, (g, s) in enumerate(p.groupby('parent_no')):
        for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
            y = s.Residual if a is ax[1, 0] else s.Predicted
            a.scatter(s.Observed, y, s=14, color=colours[k], marker=markers[k], edgecolor='k', linewidth=0.25,
                      label='F{}'.format(g) if a is ax[0, 0] else None, zorder=3)
    ax[0, 0].plot([0, 55], [0, 55], color='k', lw=0.6)
    ax[0, 0].set(xlim=(-3, 55), ylim=(-3, 60), xlabel='Starch content from weighing (%, w/w)',
                 ylabel='Predicted starch content (%)')
    e = p.Predicted - p.Observed
    r2 = 1 - np.sum(e ** 2) / np.sum((p.Observed - p.Observed.mean()) ** 2)
    ax[0, 0].text(0.03, 0.97, 'n = 110 states, 22 parents\nR² = {:.3f}\nRMSE = {:.2f} pp'.format(r2, rmse(e)),
                  transform=ax[0, 0].transAxes, va='top')
    ax[0, 0].legend(ncol=2, fontsize=5.5, loc='lower right', handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1,
                    title='Parent', title_fontsize=6)
    ax[0, 1].plot([-2, 20], [-2, 20], color='k', lw=0.6)
    ax[0, 1].set(xlim=(-1, 17), ylim=(-1, 30), xlabel='Starch content from weighing (%, w/w)',
                 ylabel='Predicted starch content (%)')
    ax[0, 1].text(0.03, 0.97, 'Authentic, 5 % and 15 % states', transform=ax[0, 1].transAxes, va='top')
    label_points(ax[0, 1], p[(p.Observed < 10) & (p.Residual.abs() > 8)], 'Predicted', 1.5)
    ax[1, 0].axhline(0, color='k', lw=0.6)
    ax[1, 0].set(xlim=(-3, 55), xlabel='Starch content from weighing (%, w/w)', ylabel='Residual (pp)')
    label_points(ax[1, 0], p[p.Residual.abs() > 8], 'Residual', 1.5)
    m = pd.read_csv(HERE / 'summary' / 'metrics_all.csv')
    m = m[(m.task == TASK) & (m.Metric == 'RMSE')].set_index('Variant').loc[list(NAMES)]
    x = np.arange(len(m))
    ax[1, 1].bar(x, m.Estimate, color=['#bdbdbd', '#6baed6', '#74c476', '#3182bd', '#de2d26'], width=0.65,
                 edgecolor='k', linewidth=0.4)
    ax[1, 1].errorbar(x, m.Estimate, yerr=[m.Estimate - m.CI95_low, m.CI95_high - m.Estimate], fmt='none', ecolor='k',
                      elinewidth=0.7, capsize=2)
    for xi, v in zip(x, m.Estimate):
        ax[1, 1].text(xi, 0.15, '{:.2f}'.format(v), ha='center', va='bottom', fontsize=7)
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels([NAMES[v].replace(' ', '\n', 1) for v in m.index], fontsize=7)
    ax[1, 1].set(ylabel='RMSE (pp), all 110 states', ylim=(0, 7.5))
    for a, letter in zip(ax.ravel(), 'ABCD'):
        a.text(-0.16, 1.03, letter, transform=a.transAxes, fontsize=11, fontweight='bold')
    fig.tight_layout()
    save(fig, 'Fig6_adulteration_primary')


def fig_parent():
    rows = []
    for v in NAMES:
        p = oof(v)
        for g, s in p.groupby('group_id'):
            rows.append({'variant': v, 'parent': g, 'RMSE': rmse(s.Predicted - s.Observed),
                         'bias': float((s.Predicted - s.Observed).mean())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'FigS_error_by_parent_values.csv', index=False, encoding='utf-8-sig')
    order = sorted(t.parent.unique(), key=lambda g: int(g[1:]))
    fig, ax = plt.subplots(2, 1, figsize=(17 * CM, 11 * CM), sharex=True)
    width = 0.27
    for k, (v, c) in enumerate([('nir_full', '#6baed6'), ('nir_stack', '#3182bd'), ('fusion_stack', '#de2d26')]):
        s = t[t.variant == v].set_index('parent').loc[order]
        ax[0].bar(np.arange(len(order)) + (k - 1) * width, s.RMSE, width, color=c, label=NAMES[v], edgecolor='k', linewidth=0.3)
        ax[1].bar(np.arange(len(order)) + (k - 1) * width, s.bias, width, color=c, edgecolor='k', linewidth=0.3)
    ax[0].set_ylabel('RMSE by parent (pp)')
    ax[0].legend(ncol=3, loc='upper right')
    ax[1].axhline(0, color='k', lw=0.6)
    ax[1].set_ylabel('Bias by parent (pp)')
    ax[1].set_xticks(np.arange(len(order)))
    ax[1].set_xticklabels(order)
    fig.tight_layout()
    save(fig, 'FigS_error_by_parent')


def fig_partitions():
    r = pd.read_csv(HERE / 'summary' / 'repeats_by_partition.csv')
    r = r[r.task == TASK]
    col = [c for c in r.columns if c.upper() == 'RMSE'][0]
    fixed = pd.read_csv(HERE / 'summary' / 'metrics_all.csv')
    fixed = fixed[(fixed.task == TASK) & (fixed.Metric == 'RMSE')].set_index('Variant').Estimate
    fig, ax = plt.subplots(figsize=(11 * CM, 7.5 * CM))
    rng = np.random.default_rng(0)
    for k, v in enumerate(NAMES):
        y = r[r.variant == v][col].to_numpy()
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(y)), y, s=14, color='#3182bd', edgecolor='k', linewidth=0.25, zorder=3)
        ax.plot([k - 0.25, k + 0.25], [y.mean()] * 2, color='k', lw=1.2)
        ax.scatter([k], [fixed[v]], marker='D', s=22, facecolor='none', edgecolor='#de2d26', linewidth=1, zorder=4)
        ax.text(k, 1.3, 'n = {}'.format(len(y)), ha='center', fontsize=6.5)
    ax.set_xticks(range(len(NAMES)))
    ax.set_xticklabels([n.replace(' ', '\n', 1) for n in NAMES.values()], fontsize=7)
    ax.set(ylabel='RMSE (pp), all 110 states', ylim=(1, 7))
    ax.scatter([], [], s=14, color='#3182bd', edgecolor='k', linewidth=0.25, label='random parent-grouped partition')
    ax.plot([], [], color='k', lw=1.2, label='mean')
    ax.scatter([], [], marker='D', s=22, facecolor='none', edgecolor='#de2d26', linewidth=1, label='fixed partition')
    ax.legend(fontsize=6.5, loc='upper right')
    fig.tight_layout()
    save(fig, 'FigS_partitions')


def fig_perm():
    path = HERE / 'stack_confirm' / 'permutation_null.csv'
    if not path.exists():
        print('permutation null not available yet; figure skipped')
        return
    null = pd.read_csv(path)
    obs = oof('fusion_stack')
    obs_rmse = rmse(obs.Predicted - obs.Observed)
    fig, ax = plt.subplots(figsize=(9 * CM, 6.5 * CM))
    ax.hist(null.RMSE, bins=15, color='#bdbdbd', edgecolor='k', linewidth=0.4)
    ax.axvline(obs_rmse, color='#de2d26', lw=1.2)
    ax.text(obs_rmse, ax.get_ylim()[1] * 0.95, ' observed\n {:.2f} pp'.format(obs_rmse), color='#de2d26', va='top')
    ax.set(xlabel='RMSE of the complete procedure (pp)', ylabel='Number of permutations (of {})'.format(len(null)),
           xlim=(0, max(null.RMSE.max() * 1.08, 22)))
    fig.tight_layout()
    save(fig, 'FigS_permutation_null')


if __name__ == '__main__':
    fig6()
    fig_parent()
    fig_partitions()
    fig_perm()
    print('figures written to', OUT)
