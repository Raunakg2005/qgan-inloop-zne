"""
audit_numbers.py — Cross-check every number in the manuscript text
===================================================================
Collects all numbers that appear in the generated tables (../tables/*.tex)
and in results/*.json (rounded to 0-4 decimals, plus scientific-notation
mantissas), then lists every number in the manuscript body that is NOT
found there.  Each listed number must be checked by hand (it may be a
setting such as 2000 iterations or a quantity defined in the text).

    python audit_numbers.py [--tex PAPER.tex] [--tables DIR]
"""

import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--tex", default=os.path.join(HERE, "..", "QGenNet_TQE_resubmission.tex"))
_ap.add_argument("--tables", default=os.path.join(HERE, "..", "tables"))
_args = _ap.parse_args()
TEX = _args.tex

known = set()


def add(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return
    for d in range(0, 5):
        known.add(f"{round(x, d):.{d}f}")
    m = f"{x:.1e}"
    known.add(m.split("e")[0])
    if x != 0:
        known.add(f"{x:.0e}".split("e")[0])


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    else:
        add(o)


for f in glob.glob(os.path.join(HERE, "results", "*.json")):
    walk(json.load(open(f)))
for f in glob.glob(os.path.join(_args.tables, "*.tex")):
    for n in re.findall(r"\d+\.\d+|\d+", open(f).read().replace("{,}", "")):
        known.add(n)
for f in glob.glob(os.path.join(HERE, "runs", "*", "*", "seed*", "metrics.json")):
    walk(json.load(open(f)).get("train_seconds", 0) / 60)

s = open(TEX, encoding="utf-8").read()
body = s[s.index(r"\begin{abstract}"):s.index(r"\bibliographystyle")]
body = re.sub(r"\\(label|ref|cite|eqref|input|tableorplaceholder|figorplaceholder)\{[^}]*\}", "", body)
body = re.sub(r"%.*", "", body)
missing = {}
for ln, line in enumerate(body.splitlines(), 1):
    for n in re.findall(r"(?<![\w^_])\d+\.\d+|(?<![\w^_.])\d{2,}(?![\d.])", line):
        if n not in known:
            missing.setdefault(n, []).append(line.strip()[:110])
for n, lines in sorted(missing.items(), key=lambda kv: kv[0]):
    print(f"{n:>8}  {lines[0]}")
print(f"\n{len(missing)} distinct numbers not found in generated outputs (check each by hand)")
