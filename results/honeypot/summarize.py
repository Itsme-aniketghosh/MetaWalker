"""Table of the real-honeypot searches (per-search summaries written by experiments/honeypot/search.py, one folder per round).

Score = certified exploit rate: the best of the top candidates re-measured on fresh PPO seeds (higher is better).
Usage: python summarize.py
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
rows = defaultdict(list)
for f in sorted(HERE.glob('round_*/*.json')):
    s = json.load(open(f))
    rows[(f.parent.name, s['space'], s['budget'], s['method'])].append(s['cert_exploit'])
print(f'{"round":12s} {"space":6s} {"budget":>6s}  {"method":28s} {"n":>2s}  {"score":>5s}  {"se":>4s}')
for (rd, space, budget, method), v in sorted(rows.items(), key=lambda kv: (kv[0][0], -st.mean(kv[1]))):
    se = st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float('nan')
    print(f'{rd:12s} {space:6s} {budget:6d}  {method:28s} {len(v):2d}  {st.mean(v):5.2f}  {se:4.2f}')
