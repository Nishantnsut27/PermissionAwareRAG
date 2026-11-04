"""Run the pilot then the full baseline. Repeat the same command to resume.

Ctrl+C pauses safely. Fresh cases run before earlier failures on each explicit
restart, up to four total attempts per case. All attempts remain in the logs.
--plan is offline.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import dataset as ds, runner
from answering.src import config

PILOT_IDS = ['E001', 'E002', 'E013', 'E022', 'E027', 'E033', 'E035', 'E043',
             'E049', 'E053', 'E056', 'E061', 'E073', 'E079', 'E083', 'E087']


@contextmanager
def run_lock(path):
    """An OS lock is released even if the terminal or process is killed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('This evaluation is already running in another terminal.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def execute(name: str, *, plan: bool = False, phase: str = 'all') -> int:
    dataset = ds.load()
    catalog = ds.PermissionCatalog()
    problems = ds.validate(dataset, catalog)
    if problems:
        raise ds.DatasetError('\n'.join(problems))
    paths = [(stage, ROOT / 'evaluation' / 'runs' / f'{name}-{stage}.json', only)
             for stage, only in [('pilot', PILOT_IDS), ('full', None)]
             if phase == 'all' or stage == phase]
    if plan:
        for phase, path, only in paths:
            cases = runner.select_cases(dataset, only=only)
            previous = runner.load_results(path) if path.exists() else None
            if previous:
                if previous.get('selected_ids') != [case.eval_id for case in cases]:
                    raise ds.DatasetError('The saved report has a different case selection.')
                limits = runner.attempt_limits(previous)
                controls = {**previous['configuration'], 'groq_request_policy': runner.GROQ_REQUEST_POLICY,
                            'case_attempt_limits': limits}
                signature = runner.fingerprint(dataset, catalog, controls)
                if previous['fingerprint'] != signature and not runner.compatible_groq_checkpoint(previous, {**controls, 'fingerprint': signature}):
                    raise ds.DatasetError('The saved report does not match the current evaluation inputs.')
                rows = {row['eval_id']: row for row in previous['results']}
                pending = [case.eval_id for case in runner.execution_order(cases, rows, limits)
                           if runner.attempt_count(rows.get(case.eval_id, {})) < runner.attempt_limit(case.eval_id, limits)
                           and (not runner.case_finished(rows.get(case.eval_id, {}))
                                or runner.retry_candidate(rows.get(case.eval_id, {})))]
                print(f'{phase}: keep all {len(cases)} cases in {path}')
                print(f'Retry once this invocation ({len(pending)}): {", ".join(pending) or "none"}')
                print(f'Preserve {sum(row.get("status") == "PASS" for row in rows.values())} passed cases.')
                if limits:
                    print('One final fifth attempt only for: ' + ', '.join(limits) + '. All other cases stay capped at four.')
                print(f'Existing response log: {previous["response_log"]}')
            else:
                print(f'{phase}: {len(cases)} cases; {max(0, len(cases)-1)*1.5:.1f} minutes pacing; {path}')
        print(f'Each restart: one attempt per queued case; default cap {runner.MAX_CASE_ATTEMPTS}, with only the listed report-specific exceptions.')
        print('Offline validation passed. No provider or index calls made.')
        return 0
    config.validate_env()
    with run_lock(ROOT / '.cache' / 'evaluation.lock'):
        for phase, path, only in paths:
            exists = path.exists()
            print(f'{"Resuming" if exists else "Starting"} {phase}: {path}', flush=True)
            print('Ctrl+C once to pause. Repeat this command to resume: fresh cases first, '
                  f'previous failures at the end; default maximum {runner.MAX_CASE_ATTEMPTS} total attempts per case.', flush=True)
            if exists:
                limits = runner.attempt_limits(runner.load_results(path))
                if limits:
                    print('One final fifth attempt is permitted only for: ' + ', '.join(limits) + '.', flush=True)
            print('Jina requests alternate between configured keys. A failed Jina request pauses '
                  'evaluation; check/change the key named in the error, then rerun.', flush=True)
            print(f'Groq requests rotate across {config.groq_keys.pool().size} unique configured keys; '
                  'rate-limited keys wait for provider cooldowns. Keys in the same organization may share daily quota.', flush=True)
            result = runner.run(dataset, catalog, only=only, delay_s=90,
                                retry_wait_s=90, max_retries=0, resume=exists, output=path)
            print(f'Report: {path}\nResponse log: {result["response_log"]}', flush=True)
            print(f'Queued for next restart: {len(result["retry_queue"])}; '
                  f'attempt limit reached: {len(result["exhausted_cases"])}.', flush=True)
            if not result['selection_complete'] or result.get('stop_reason'):
                print(f'Paused: {result.get("stop_reason") or "incomplete dependencies"}. '
                      'Inspect the failure reasons and logs before restarting.', flush=True)
                if result.get('stop_detail'):
                    print(f'Why: {result["stop_detail"]}', flush=True)
                if result['exhausted_cases']:
                    print('Cases at their attempt limit remain unresolved and cannot be '
                          'retried in this report.', flush=True)
                return 130 if result.get('stop_reason') == 'interrupted' else 2
            print(f'{phase.title()} complete: {result["summary"]["passed"]} passed, '
                  f'{result["summary"]["failed"]} quality/security failures. Saved scores are retained.', flush=True)
        print('Full baseline complete. Review the saved failures and response logs.', flush=True)
        return 1 if result['summary']['failed'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='baseline-v11-20261104',
                        help='Stable checkpoint name; keep it unchanged to resume, use a new name for a new baseline')
    parser.add_argument('--plan', action='store_true', help='offline validation and preview only')
    parser.add_argument('--phase', choices=('all', 'pilot', 'full'), default='all',
                        help='Use full to resume the existing 90-case report directly, without visiting the pilot')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}', args.name):
        parser.error('--name must be a simple filename, without a path')
    logging.basicConfig(level=logging.ERROR)
    try:
        return execute(args.name, plan=args.plan, phase=args.phase)
    except KeyboardInterrupt:
        print('Paused. Run the same command to resume.', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'Evaluation could not continue: {exc}', file=sys.stderr)
        print('Existing checkpoints are preserved. Run the same command after resolving the issue.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
