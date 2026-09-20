import os, json, subprocess, statistics, math
from array import array
from pathlib import Path
root = Path(__file__).resolve().parent
records = []
for threads in (1, 4):
    for pair in range(5):
        order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
        for variant in order:
            env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
            env.pop('CASE_FILTER', None)
            if pair == 0:
                out = root / f'{variant}_outputs_t{threads}'
                out.mkdir(exist_ok=True)
                env['OUTPUT_DIR'] = str(out)
            command = ['taskset', '-c', '0' if threads == 1 else '0-3', str(root / f'{variant}_probe')]
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            for line in result.stdout.splitlines():
                record = json.loads(line)
                record.update(threads=threads, pair=pair, variant=variant)
                records.append(record)
            (root / 'paired.json').write_text(json.dumps(records, indent=2) + '\n')
            print(f'threads={threads} pair={pair} variant={variant}', flush=True)
summary = []
for threads in (1, 4):
    names = [r['case'] for r in records if r['threads'] == threads and r['pair'] == 0 and r['variant'] == 'baseline']
    for name in names:
        group = [r for r in records if r['threads'] == threads and r['case'] == name]
        base = statistics.median(r['median_ms'] for r in group if r['variant'] == 'baseline')
        after = statistics.median(r['median_ms'] for r in group if r['variant'] == 'candidate')
        ratios = [next(r['median_ms'] for r in group if r['variant'] == 'candidate' and r['pair'] == p) / next(r['median_ms'] for r in group if r['variant'] == 'baseline' and r['pair'] == p) for p in range(5)]
        a,b = [array('f', (root / f'{v}_outputs_t{threads}' / f'{name}.f32').read_bytes()) for v in ['baseline', 'candidate']]
        assert len(a) == len(b)
        assert all(math.isfinite(x) for x in a) and all(math.isfinite(x) for x in b)
        max_abs = max(abs(x-y) for x,y in zip(a,b))
        max_ref = max(map(abs, a))
        relative_l2 = math.sqrt(sum((x-y)**2 for x,y in zip(a,b)) / sum(x*x for x in a))
        assert relative_l2 < 1e-5, (name, relative_l2)
        assert max_abs < 1e-5 * max_ref, (name, max_abs, max_ref)
        summary.append(dict(case=name, threads=threads, baseline_ms=base, candidate_ms=after,
            reduction_percent=100*(1-after/base), speedup=base/after,
            paired_reduction_range_percent=[100*(1-max(ratios)),100*(1-min(ratios))],
            output_count=len(a), max_abs=max_abs, max_ref=max_ref, relative_l2=relative_l2))
(root / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
