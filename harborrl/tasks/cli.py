"""CPU-only inspection of locally fetched Harbor datasets."""
import argparse
from collections import Counter
import json
from pathlib import Path
from uuid import uuid4

from harborrl.data.harbor.inspector import inspect


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    probe = sub.add_parser('inspect', help='inspect local tasks without Docker or model dependencies')
    probe.add_argument('--tasks-dir', type=Path, required=True)
    probe.add_argument('--dataset', required=True)
    probe.add_argument('--revision', required=True, help='source revision; recorded alongside content digest')
    probe.add_argument('--derived', action='store_true')
    probe.add_argument('--run-dir', type=Path, required=True)
    report = sub.add_parser('report')
    report.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'inspect':
        if not args.tasks_dir.is_dir():
            parser.error('tasks-dir must exist')
        tasks = sorted(p.parent for p in args.tasks_dir.rglob('task.toml'))
        if not tasks:
            parser.error('no task.toml found')
        rows = []
        for task in tasks:
            result = inspect(task, args.dataset).to_dict()
            rows.append({**result, 'task_path': str(task.resolve()), 'revision': args.revision,
                         'derived': args.derived, 'execution_backend': 'lightrl_interactive',
                         'stage': 'inspect', 'evaluable': False, 'trainable': False,
                         'later_stages': 'not_attempted'})
        args.run_dir.mkdir(parents=True, exist_ok=True)
        # Each attempt is immutable; a new result never overwrites old evidence.
        with (args.run_dir / (uuid4().hex + '.jsonl')).open('x') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    rows = [json.loads(line) for path in sorted(args.run_dir.glob('*.jsonl'))
            for line in path.read_text().splitlines() if line.strip()]
    counts = Counter(row['status'] for row in rows)
    summary = {'attempts': len(rows), 'status_counts': dict(counts),
               'scope': 'local static inspection only; no execution or training evidence'}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.run_dir.is_dir():
        (args.run_dir / 'report.md').write_text(
            '# Harbor 任务静态检查\n\n仅证明本地格式检查结果；执行、评测与训练均未验收。\n\n'
            + '\n'.join(f'- {name}: {count} 次检查' for name, count in sorted(counts.items())) + '\n')
    return 0
