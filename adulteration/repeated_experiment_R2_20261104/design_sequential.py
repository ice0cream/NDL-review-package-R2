# -*- coding: utf-8 -*-
"""Revision-2 re-run of the cassava-starch adulteration experiment, sequential-addition design.

Constraint: only about 10 g of each FTR parent powder remains, so states cannot be prepared
independently. Each parent is therefore measured as one block in one tube:
  step 0  ~7 g parent powder, measured as the authentic state (0 %);
  step 1-4  cassava starch added to reach the next nominal level, mixed and measured.
Only starch is added after step 0, so one aliquot of parent powder carries all five states.

Design (pre-specified before any measurement):
  * 22 FTR parents (F1-F22); per parent 1 authentic state + 4 mixtures = 110 states.
  * Complete design: every parent gets the same nominal levels 5, 15, 30 and 50 % w/w, so each level
    has 22 mixtures from 22 parents. No level below 5 % is tested. Levels within a parent are measured in
    ascending order (unavoidable with sequential addition); a whole block takes about 45 min.
  * Parent blocks are measured in random order over 3 days (8/7/7 blocks), chosen so that each
    level's count per day differs by at most 2 between days.
  * Actual concentration is computed from the weighings before and after each addition:
        c_k = (c_(k-1) * (W_before - tare) + (W_after - W_before)) / (W_after - tare)
    Powder left in the cup or dish is assumed to have the tube's current composition.
Outputs: design_allocation.csv, repeated_experiment_record_sheet_blank.xlsx, design_summary.txt.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

SEED = 20261104
OUT = Path(__file__).resolve().parent
PARENTS = [f'F{i}' for i in range(1, 23)]
STRATA = [(5, 5), (15, 15), (30, 30), (50, 50)]  # same four levels for every parent
START_G = 7.0
BLOCKS_PER_DAY = [8, 7, 7]


def allocate_levels(rng):
    levels = {p: [] for p in PARENTS}
    for low, high in STRATA:
        order = rng.permutation(PARENTS)
        for p in order[:11]:
            levels[p].append(low)
        for p in order[11:]:
            levels[p].append(high)
    return {p: sorted(v) for p, v in levels.items()}


def order_blocks(levels, rng, tries=20000):
    for _ in range(tries):
        order = list(rng.permutation(PARENTS))
        day_of = {}
        start = 0
        for d, n in enumerate(BLOCKS_PER_DAY, 1):
            for p in order[start:start + n]:
                day_of[p] = d
            start += n
        counts = pd.DataFrame([(day_of[p], lv) for p in PARENTS for lv in levels[p]], columns=['day', 'level'])
        table = pd.crosstab(counts['level'], counts['day'])
        if (table.max(axis=1) - table.min(axis=1)).max() <= 1:
            return order, day_of
    raise RuntimeError('No admissible block order found.')


def build_design():
    rng = np.random.default_rng(SEED)
    levels = allocate_levels(rng)
    order, day_of = order_blocks(levels, rng)
    rows, seq = [], 0
    for b, p in enumerate(order, 1):
        for k, lv in enumerate([0] + levels[p]):
            seq += 1
            rows.append({'measure_order': seq, 'day': day_of[p], 'block': b, 'parent': p, 'step': k,
                         'nominal_percent': lv, 'file_prefix': f'{p}_S{k}'})
    return pd.DataFrame(rows)


THIN = Side(style='thin', color='999999')
THICK = Side(style='medium', color='333333')
HEAD = PatternFill('solid', fgColor='DDE6F0')
FILL = PatternFill('solid', fgColor='FFF7D6')
GREY = PatternFill('solid', fgColor='EEEEEE')

HEADER = ['块', '母本', '步骤', '目标浓度 %', '空管 g', '加料前 g', '应加淀粉 g', '加料后 g', '实际浓度 %',
          '文件名前缀', 'NIR 装样 g 第1次', '第2次', '第3次',
          '色度1 L*', 'a*', 'b*', '色度2 L*', 'a*', 'b*', '色度3 L*', 'a*', 'b*', '开始时间', '备注']
WIDTHS = [4, 6, 5, 8, 8, 9, 10, 9, 9, 10, 9, 7, 7, 8, 6, 6, 8, 6, 6, 8, 6, 6, 8, 16]
INPUT_COLS = {'E', 'F', 'H', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X'}


def write_head(ws, note):
    ws.cell(1, 1, note).font = Font(bold=True)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(HEADER))
    for j, h in enumerate(HEADER, 1):
        c = ws.cell(2, j, h)
        c.font, c.fill = Font(bold=True), HEAD
        c.border = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
        c.alignment = Alignment(wrap_text=True, horizontal='center', vertical='center')
    ws.row_dimensions[2].height = 34
    for j, w in enumerate(WIDTHS, 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = 'D3'
    ws.print_title_rows = '2:2'
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True


def day_sheet(wb, d, sub):
    ws = wb.create_sheet(f'第{d}天')
    n_blocks = sub['block'].nunique()
    write_head(ws, f'第{d}天（{n_blocks} 个母本，{len(sub)} 个状态；按块序进行，块内按步骤 0→4）'
                   f'  日期：________  操作者：________  淀粉批号：________')
    r = 3
    for _, block in sub.groupby('block', sort=True):
        r0 = r
        for i, s in enumerate(block.itertuples()):
            values = [s.block, s.parent, s.step, s.nominal_percent]
            for j, v in enumerate(values, 1):
                ws.cell(r, j, v)
            ws.cell(r, 10, s.file_prefix)
            if s.step == 0:
                ws.cell(r, 6, '—')
                ws.cell(r, 7, f'母本粉约 {START_G:.1f}')
                ws.cell(r, 9, 0)
            else:
                ws.cell(r, 5, '—')
                prev = r - 1
                ws.cell(r, 7, f'=IF(OR(F{r}="",I{prev}=""),"",ROUND((F{r}-$E${r0})*(D{r}-I{prev})/(100-D{r}),3))')
                ws.cell(r, 9, f'=IF(OR(H{r}="",F{r}="",I{prev}=""),"",'
                              f'ROUND((I{prev}/100*(F{r}-$E${r0})+(H{r}-F{r}))/(H{r}-$E${r0})*100,3))')
            for j in range(1, len(HEADER) + 1):
                c = ws.cell(r, j)
                letter = get_column_letter(j)
                last = i == len(block) - 1
                c.border = Border(left=THIN, right=THIN, top=THICK if i == 0 else THIN,
                                  bottom=THICK if last else THIN)
                c.alignment = Alignment(horizontal='center', vertical='center')
                if c.value == '—':
                    c.fill = GREY
                elif letter in INPUT_COLS and c.value in (None, ''):
                    c.fill = FILL
            ws.row_dimensions[r].height = 22
            r += 1


def simple_sheet(wb, title, header, rows, widths, note):
    ws = wb.create_sheet(title)
    ws.cell(1, 1, note).font = Font(bold=True)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(header))
    for j, h in enumerate(header, 1):
        c = ws.cell(2, j, h)
        c.font, c.fill = Font(bold=True), HEAD
        c.border = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
        c.alignment = Alignment(wrap_text=True, horizontal='center', vertical='center')
    for i, row in enumerate(rows, 3):
        for j, v in enumerate(row, 1):
            c = ws.cell(i, j, v)
            c.border = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
            c.alignment = Alignment(horizontal='center', vertical='center')
            if v in (None, ''):
                c.fill = FILL
        ws.row_dimensions[i].height = 22
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True


def workbook(design):
    wb = Workbook()
    wb.remove(wb.active)
    for d in sorted(design['day'].unique()):
        day_sheet(wb, d, design[design['day'] == d])
    qc_rows = []
    for d in sorted(design['day'].unique()):
        for slot, tag in [('开始', 'start'), ('第4块后', 'mid'), ('结束', 'end')]:
            qc_rows.append([f'第{d}天', slot, '', '', '', '', '', f'STARCH_D{d}_{tag}', '', '', '', ''])
    simple_sheet(wb, '质控与环境', ['测量日', '时点', '时间', '温度 ℃', '相对湿度 %', 'NIR 背景/自检',
                                  '色度仪白板校准', '淀粉参比 NIR 文件名', '淀粉参比 L*', 'a*', 'b*', '备注'],
                 qc_rows, [8, 8, 8, 8, 10, 12, 13, 18, 10, 7, 7, 18],
                 '每天开始、第 4 块后、结束各记录一次；首日在备注中写下 OPUS 方法文件名和色度样品皿装样量')
    moist = [[p, '', '', '', '', '', '', ''] for p in PARENTS] + [['木薯淀粉', '', '', '', '', '', '', '']]
    simple_sheet(wb, '含水量', ['样品', '方法', '称量瓶 g', '干燥前(含瓶) g', '干燥后(含瓶) g', '含水量 %',
                              '测定时间', '备注'], moist, [10, 14, 10, 14, 14, 10, 14, 16],
                 '含水量：每个母本粉和淀粉各测 1 次（约 1.5–2 g），在取出 7 g 测量用粉后用剩余粉末测定')
    path = OUT / 'repeated_experiment_record_sheet_blank.xlsx'
    wb.save(path)
    return path


def main():
    design = build_design()
    design.to_csv(OUT / 'design_allocation.csv', index=False, encoding='utf-8-sig')
    path = workbook(design)
    mix = design[design.nominal_percent > 0]
    top = mix.groupby('parent')['nominal_percent'].max()
    starch = (START_G * top / (100 - top)).sum()
    summary = [
        f'seed={SEED}; states={len(design)}; mixtures={len(mix)}; start parent mass={START_G} g',
        'Levels by measurement day:', pd.crosstab(design['nominal_percent'], design['day'], margins=True).to_string(),
        'Block order (day: parents):'] + [
        f'  day {d}: ' + ', '.join(design[(design.day == d) & (design.step == 0)]['parent']) for d in sorted(design.day.unique())] + [
        'Levels per parent:', mix.groupby('parent')['nominal_percent'].apply(lambda s: ', '.join(map(str, s))).to_string(),
        f'Cassava starch needed (no losses): {starch:.1f} g',
        f'Output: {path.name}, design_allocation.csv']
    (OUT / 'design_summary.txt').write_text('\n'.join(summary), encoding='utf-8')
    print('\n'.join(summary))


if __name__ == '__main__':
    main()
