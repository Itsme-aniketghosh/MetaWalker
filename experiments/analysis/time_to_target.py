"""Per search: PPO-costing evaluations used, and how many it took to first reach a valid level at exploit >= 0.8 / 1.0."""
import json, glob, os
out = {}
for d in ('honeypot', 'honeypotE', 'honeypotF'):
    for f in glob.glob(os.path.expanduser(f'~/{d}/runs/manyh_*.jsonl')):
        n_ppo, n, first = 0, 0, {}
        with open(f) as fh:
            for line in fh:
                l = json.loads(line)
                if l.get('free'):
                    continue
                n += 1
                n_ppo += bool(l['valid'] or l.get('reason') == 'honest path not learnable')
                for thr in (0.8, 1.0):
                    if l['valid'] and l['exploit'] >= thr and str(thr) not in first:
                        first[str(thr)] = (n_ppo, n)
        out[d + '/' + os.path.basename(f)[6:-6]] = dict(ppo=n_ppo, evals=n, first=first)
json.dump(out, open(os.path.expanduser('~/ttt.json'), 'w'))
print(len(out), 'searches')
