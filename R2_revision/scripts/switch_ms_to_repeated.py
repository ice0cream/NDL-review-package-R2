# -*- coding: utf-8 -*-
"""Manuscript (marked copy, edited by hand): describe only the adulteration experiment that is reported, and renumber
references to the Supplementary Material (si_numbering.py). Changed text is set in the colour of this revision's
changes (C00000). The clean copy is then written from the marked copy by finalize_submission_files.py.
"""
from pathlib import Path
import re
import shutil

import docx
from docx.shared import RGBColor

import si_numbering as SN

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / 'manuscript' / 'REVISE2-MANUSCRIPT' / 'manuscript_R2_marked.docx'
BAK = ROOT / 'R2_revision' / 'backup_manuscript_20261119' / 'manuscript_R2_marked_before_switch.docx'
RED = RGBColor(0xC0, 0x00, 0x00)

EDITS = [
    (r'For adulteration, a repeated experiment \(22 parent powders, each at 0, 5, 15, 30, and 50% starch; 110 states\) was analyzed',
     'For adulteration, an experiment with 22 parent powders, each at 0, 5, 15, 30, and 50% starch (110 states), was analyzed'),
    (r' of 3\.45 percentage points, reproducing the original experiment\.', ' of 3.45 percentage points.'),
    (r'The adulteration experiment reported here was repeated during revision with a balanced design \(6–7 November 2026\); '
     r'the original experiment is documented in Supplementary Methods S1 and Tables S6–S9\.',
     'The adulteration experiment used a balanced design and was carried out on 6–7 November 2026 (Supplementary Methods S3).'),
    (r'The code, candidate grid, and settings were those of the original submission and were not changed for the repeated '
     r'experiment \(', 'The code, candidate grid, and settings were fixed before the measurements ('),
    (r'All results below are from the repeated experiment \(', 'All results below are from the adulteration experiment ('),
    (r' These windows coincide with or adjoin two of the regions identified in the original experiment \(6851–6666 and '
     r'7152–7057 cm⁻¹\)\.', ''),
    (r'For the 88 mixtures analyzed alone, which corresponds to the primary analysis of the original experiment, R² was 0\.935 '
     r'and RMSE 4\.32 percentage points \(3\.44–5\.15\), compared with 0\.929 and 4\.68 percentage points in the original '
     r'experiment \(Table S6\)\. The original result was therefore reproduced on new measurements with an unchanged pipeline\.',
     'For the 88 mixtures analyzed alone, R² was 0.935 and RMSE 4.32 percentage points (3.44–5.15).'),
    (r'; it lay outside the NIR model domain by Hotelling T² and Q, and it had also been the most over-predicted parent in the '
     r'original experiment\.', '; it lay outside the NIR model domain by Hotelling T² and Q.'),
    (r'For adulteration, the repeated experiment used the same 22 parent powders,',
     'For adulteration, the experiment used 22 parent powders,'),
    (r'In a repeated adulteration experiment, decision-level fusion', 'In the adulteration experiment, decision-level fusion'),
    (r' over 10 random partitions\), reproducing the original result\.', ' over 10 random partitions).'),
    (r'Starch estimation in the repeated adulteration experiment by', 'Starch estimation in the adulteration experiment by'),
    (r'decision-level fusion model in the repeated experiment\. Windows', 'decision-level fusion model. Windows'),
]


def main():
    if not BAK.exists():
        shutil.copy(MS, BAK)
    doc = docx.Document(str(MS))
    done = {k: 0 for k in range(len(EDITS))}
    for p in SN.all_paragraphs(doc):
        for k, (pat, new) in enumerate(EDITS):
            done[k] += SN.para_sub(p, pat, new.replace('\\', r'\\'), colour=RED)
    missing = [EDITS[k][0][:70] for k, n in done.items() if n != 1]
    assert not missing, ('edits not applied exactly once', missing)
    text = '\n'.join(p.text for p in SN.all_paragraphs(doc))
    bad = SN.removed_references(text)
    assert not bad, ('references to removed items', bad)
    n = 0
    for p in SN.all_paragraphs(doc):
        n += SN.para_sub(p, SN.SPAN.pattern, lambda m: SN.renumber(m.group(0)), colour=RED)
    doc.save(str(MS))
    left = [text[max(0, m.start() - 70):m.end() + 50].replace('\n', ' ') for m in re.finditer(
        r'(?i)original (dataset|experiment|submission|result)|repeated (adulteration )?experiment|reproducing the original',
        '\n'.join(p.text for p in SN.all_paragraphs(docx.Document(str(MS)))))]
    print('edits applied:', len(EDITS), '| references renumbered:', n, '| remaining mentions:', left)


if __name__ == '__main__':
    main()
