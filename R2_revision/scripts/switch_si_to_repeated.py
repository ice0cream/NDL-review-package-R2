# -*- coding: utf-8 -*-
"""Supplementary Material: remove the analyses of the first adulteration experiment, describe only the repeated
experiment, and renumber tables and figures (si_numbering.py).

    python switch_si_to_repeated.py --dry-run     list the blocks that would be removed
    python switch_si_to_repeated.py               apply (a copy of the input is kept in backup_manuscript_20261119)
"""
from pathlib import Path
import re
import shutil
import sys

import docx
from docx.oxml.ns import qn

import si_numbering as SN

ROOT = Path(__file__).resolve().parents[2]
SI = ROOT / 'manuscript' / 'REVISE2-MANUSCRIPT' / 'supplementary_material_R2.docx'
BAK = ROOT / 'R2_revision' / 'backup_manuscript_20261119' / 'supplementary_material_R2_before_switch.docx'

REMOVE_TABLE_CAPTIONS = [r'^Table S6 Decision', r'^Table S7 Mixture', r'^Table S8 Concentration', r'^Table S9 Exploratory detection',
                         r'^Table S12 Exploratory modality', r'^Table S17 Adulteration', r'^Table S12 \(revised\)',
                         r'^Table S19\.', r'^Table S20\.', r'^Table S22\.']
REMOVE_FIG_CAPTIONS = [r'^Fig\. S13\.', r'^Fig\. S14\.', r'^Fig\. S15\.', r'^Fig\. S19\.']
CAPTION = re.compile(r'^(Table S\d+|Fig\. S\d+|Supplementary Methods S\d|Graphical abstract$|Material added in Revision 2|Tables and figure added)')

TEXT_EDITS = [
    # Supplementary Methods S2: the configurations of the starch pipeline are now in the table of the repeated experiment
    (r'Full settings, fold sample sizes, selected configurations and bands are provided in Tables S10 and S17\.',
     'Full settings are given in Table S10, and the configuration and windows selected in each outer fold in Table S27.'),
    # Table S10
    (r'SIMCA 14\.1 for exploratory original analyses\.', 'SIMCA 14.1 for the exploratory multivariate analyses.'),
    (r'22 parents; 83 primary states or 105 sensitivity states\. Shared parent outer plan; seed 2026\.',
     '22 parents; all 110 states (the 88 mixtures were also analysed alone). Parent outer plan fixed before modelling; seed 2026.'),
    (r'Threshold detection: 5,000 parent draws plus Wilson intervals\.', 'Detection by level: Wilson intervals.'),
    # Supplementary Methods S3
    (r'^Supplementary Methods S3 Repeated adulteration experiment', 'Supplementary Methods S3 Adulteration experiment'),
    (r'The decision-level fusion pipeline of the original submission was run unchanged on all 110 states:',
     'The decision-level fusion pipeline of Supplementary Methods S2, with its code, candidate grid and settings fixed '
     'before the measurements, was run on all 110 states:'),
    # headings and captions
    (r'botanical, phytochemical and original-dataset analyses', 'botanical and phytochemical analyses'),
    (r'^(Table S\d+\.|Fig\. S\d+\.) Repeated experiment: ', r'\1 Adulteration experiment: '),
]
FEATURE_ROW = ('Four NIR windows selected within the training data and the CIELAB coordinates, concatenated and '
               'block-scaled, with one base model selected as above; no PCA.')
S20_CAPTION = ('Fig. S20. Graphical abstract: schematic overview of the workflow for prepared Nanhaia speciosa powders. '
               'The panels in each branch are schematic; the results are those of the nested validations in Figs. 3, 5 '
               'and 6, which are the primary analyses, whereas exploratory and fixed-configuration analyses are secondary. '
               'Spectral preprocessing was applied to the NIR spectra only; CIELAB coordinates and reference values were '
               'not preprocessed. Branch 1 refers to discrimination of the four recorded material categories. In the '
               'adulteration experiment, starch was added stepwise to one aliquot of each authentic powder (0, 5, 15, 30 '
               'and 50%), and every state was repacked and measured three times by NIR and colorimetry.')


def body_items(doc):
    out = []
    for el in doc.element.body.iterchildren():
        tag = el.tag.split('}')[1]
        if tag == 'p':
            text = ''.join(t.text or '' for t in el.iter(qn('w:t'))).strip()
            out.append({'el': el, 'kind': 'p', 'text': text, 'img': bool(list(el.iter(qn('w:drawing'))))})
        elif tag == 'tbl':
            out.append({'el': el, 'kind': 't', 'text': '', 'img': False})
        else:
            out.append({'el': el, 'kind': tag, 'text': '', 'img': False})
    return out


def blocks_to_remove(items):
    caps = [k for k, it in enumerate(items) if it['kind'] == 'p' and CAPTION.match(it['text'])]
    rm = []
    for k in caps:
        t = items[k]['text']
        nxt = next((c for c in caps if c > k), len(items))
        if any(re.match(p, t) for p in REMOVE_TABLE_CAPTIONS):
            end = nxt
            # leave the image of a following figure (it precedes its caption)
            while end - 1 > k and (items[end - 1]['img'] or (items[end - 1]['kind'] == 'p' and not items[end - 1]['text'])):
                end -= 1
            rm.append((t[:60], list(range(k, end))))
        elif any(re.match(p, t) for p in REMOVE_FIG_CAPTIONS):
            start = k
            while start - 1 >= 0 and (items[start - 1]['img'] or (items[start - 1]['kind'] == 'p' and not items[start - 1]['text'])):
                start -= 1
            end = nxt
            while end - 1 > k and (items[end - 1]['img'] or (items[end - 1]['kind'] == 'p' and not items[end - 1]['text'])):
                end -= 1
            rm.append((t[:60], list(range(start, end))))
    return rm


def main(dry):
    doc = docx.Document(str(SI))
    items = body_items(doc)
    rm = blocks_to_remove(items)
    for name, idx in rm:
        kinds = ''.join('T' if items[i]['kind'] == 't' else ('I' if items[i]['img'] else 'p') for i in idx)
        print('remove {:62s} {} elements [{}]'.format(name, len(idx), kinds))
    if dry:
        return
    if not BAK.exists():
        shutil.copy(SI, BAK)
    for _, idx in rm:
        for i in idx:
            el = items[i]['el']
            el.getparent().remove(el)
    # Supplementary Methods S1: the mixture preparation of the first experiment; S3: the reason for repeating it
    for p in list(doc.paragraphs):
        t = p.text.strip()
        if t.startswith('Each preparation combined the required masses of FTR powder and starch to 20.00 g') or t.startswith('Reason. In the original experiment'):
            p._p.getparent().remove(p._p)
    for p in SN.all_paragraphs(doc):
        for pat, new in TEXT_EDITS:
            SN.para_sub(p, pat, new)
    for t in doc.tables:
        for row in t.rows:
            if row.cells[0].text.strip() == 'Fixed-configuration feature PCA':
                SN.para_sub(row.cells[0].paragraphs[0], r'.+', 'Feature fusion')
                SN.para_sub(row.cells[-1].paragraphs[0], r'.+', FEATURE_ROW)
                for extra in row.cells[-1].paragraphs[1:]:
                    extra._p.getparent().remove(extra._p)
    for p in doc.paragraphs:
        if p.text.strip().startswith('Fig. S20.'):
            SN.para_sub(p, r'.+', S20_CAPTION)
    # renumber every reference, then report what still mentions the first experiment
    for p in SN.all_paragraphs(doc):
        SN.para_sub(p, SN.SPAN.pattern, lambda m: SN.renumber(m.group(0)))
    doc.save(str(SI))
    text = '\n'.join(p.text for p in SN.all_paragraphs(docx.Document(str(SI))))
    left = [text[max(0, m.start() - 80):m.end() + 60].replace('\n', ' ') for m in
            re.finditer(r'(?i)original (dataset|experiment|submission|adulteration)|first experiment|repeated experiment|\bR1\b|83[- ]mixture|105-state|64-mixture', text)]
    print('remaining mentions:', len(left))
    for x in left:
        print('   ·', x)


if __name__ == '__main__':
    main('--dry-run' in sys.argv)
