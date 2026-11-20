# -*- coding: utf-8 -*-
"""Numbering of the Supplementary Material after the analyses of the first adulteration experiment were removed.

Removed: Tables S6–S9, S12 (and its revised version), S17, S19, S20, S22; Figs. S13, S14, S15, S19.
Every remaining table and figure is renumbered in order of appearance. renumber() rewrites references in any text
("Table S24", "Tables S24 and S25", "Figs. S23–S25", "Fig. S16A") and raises an error if a reference points to a
removed item, so that no citation of a removed item can survive unnoticed.
"""
import re

_T_KEEP = ['1', '2', '3', '4', '5', '10', '11', '13', '14', '15', '16', '18', '21'] + [str(n) for n in range(23, 39)]
_F_KEEP = [str(n) for n in range(1, 13)] + ['16', '17', '18', '20', '21', '22', '23', '24', '25', '26']
TABLE_MAP = {old: str(i + 1) for i, old in enumerate(_T_KEEP)}
FIG_MAP = {old: str(i + 1) for i, old in enumerate(_F_KEEP)}
REMOVED_TABLES = {'6', '7', '8', '9', '12', '17', '19', '20', '22'}
REMOVED_FIGS = {'13', '14', '15', '19'}

_SEP = r'(?:\s*(?:,\s*and|,|and|–|-|to)\s*)'
SPAN = re.compile(r'\b(?P<kind>Tables?|Figs?\.|Figures?)\s+(?P<body>S\d+[A-Za-z]?(?:' + _SEP + r'S?\d+[A-Za-z]?)*)')
TOK = re.compile(r'(S?)(\d+)([A-Za-z]?)')


class RemovedReference(ValueError):
    pass


def _map_body(kind, body, context):
    table = kind.lower().startswith('table')
    mp, removed = (TABLE_MAP, REMOVED_TABLES) if table else (FIG_MAP, REMOVED_FIGS)
    nums = []

    def sub(m):
        num = m.group(2)
        if num in removed or num not in mp:
            raise RemovedReference('{} S{} is removed or unknown: ...{}...'.format(kind, num, context))
        nums.append(int(num))
        return m.group(1) + mp[num] + m.group(3)

    out = TOK.sub(sub, body)
    # a range must not span a removed item
    for a, b in re.findall(r'S?(\d+)[A-Za-z]?\s*[–-]\s*S?(\d+)', body):
        inside = {str(n) for n in range(int(a), int(b) + 1)}
        if inside & removed:
            raise RemovedReference('range {} spans a removed item: ...{}...'.format(body, context))
    return out


def renumber(text):
    def repl(m):
        ctx = text[max(0, m.start() - 60):m.end() + 60].replace('\n', ' ')
        return m.group('kind') + m.group(0)[len(m.group('kind')):m.start('body') - m.start()] + _map_body(m.group('kind'), m.group('body'), ctx)
    return SPAN.sub(repl, text)


def removed_references(text):
    """List the references to removed items (for checking text before renumbering)."""
    out = []
    for m in SPAN.finditer(text):
        try:
            _map_body(m.group('kind'), m.group('body'), '')
        except RemovedReference:
            out.append(text[max(0, m.start() - 80):m.end() + 40].replace('\n', ' '))
    return out


# ------------------------------------------------------------------ Word helpers (python-docx paragraphs)
def para_sub(p, pattern, repl, colour=None):
    """Regex substitution over the text of a paragraph whose match may span runs; formatting of the first run of
    each match is kept. colour: an RGBColor applied to the run that receives a changed match (marked copy)."""
    runs = list(p.runs)
    if not runs:
        return 0
    texts = [r.text for r in runs]
    full = ''.join(texts)
    matches = list(re.finditer(pattern, full))
    n = 0
    for m in reversed(matches):
        new = m.expand(repl) if isinstance(repl, str) else repl(m)
        if new == m.group(0):
            continue
        starts, pos = [], 0
        for t in texts:
            starts.append(pos)
            pos += len(t)
        i = max(k for k in range(len(runs)) if starts[k] <= m.start())
        j = max(k for k in range(len(runs)) if starts[k] < m.end()) if m.end() > m.start() else i
        head = texts[i][:m.start() - starts[i]]
        tail = texts[j][m.end() - starts[j]:]
        if i == j:
            texts[i] = head + new + tail
        else:
            texts[i] = head + new
            for k in range(i + 1, j):
                texts[k] = ''
            texts[j] = tail
        for k in range(len(runs)):
            if runs[k].text != texts[k]:
                runs[k].text = texts[k]
        if colour is not None:
            runs[i].font.color.rgb = colour
        n += 1
    return n


def all_paragraphs(doc):
    for p in doc.paragraphs:
        yield p
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    yield p
