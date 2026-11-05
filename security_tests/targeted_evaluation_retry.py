"""Offline checks for selective retries with complete, durable attempt history."""
import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import dataset as ds, evaluator, runner
from scripts import run_evaluation


class TargetedRetryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.root / 'full.json'
        self.data = ds.GoldenDataset('offline', '1.1', '', [
            ds.EvalCase(f'E{i:03}', 'U-001', f'question {i}', 'authorized_factual', 'answer', '')
            for i in range(1, 7)])
        self.enterContext(patch.object(ds, 'PROJECT_ROOT', self.root))
        self.enterContext(patch.object(ds, 'validate', return_value=[]))
        self.enterContext(patch.object(runner, 'resolve_index_snapshot', return_value={'collection_name': 'offline'}))
        self.enterContext(patch.object(runner, 'fingerprint', return_value='same'))
        self.enterContext(patch.object(runner.index_retrieval, 'pin_revision', return_value=nullcontext()))
        self.args = dict(dataset=self.data, catalog=SimpleNamespace(documents=[]),
                         output=self.path, delay_s=0, retry_wait_s=0, max_retries=0, verbose=False)
        self.before = runner.run(**self.args, checkpoint=False, initialize_only=True)
        for case, status, count in zip(self.data.cases, ['ERROR', 'FAIL', 'FAIL', 'PASS', 'FAIL', 'ERROR'], [5, 2, 3, 1, 1, 1]):
            row = self.result(case, status).to_dict()
            row['attempts'] = [{'attempt': n + 1, 'fingerprint': 'original',
                                'result': copy.deepcopy(row)} for n in range(count)]
            self.before['results'].append(row)
        runner.save_checkpoint(self.before, self.path)

    def result(self, case, status='PASS'):
        return evaluator.CaseResult(case.eval_id, case.user_id, case.user_name, case.question,
                                    case.category, case.expected_behavior, status=status,
                                    error='provider error' if status == 'ERROR' else None,
                                    outcome='answer', answered=True, answer='answer-' + case.eval_id)

    def run_batch(self):
        return runner.run(**self.args, resume=True, targeted_retry=True)

    def test_exact_selection_one_extra_attempt_and_all_history_in_same_report_and_log(self):
        def execute(case, *_):
            return self.result(case, 'FAIL' if case.eval_id == 'E002' else 'PASS')
        with patch.object(evaluator, 'run_case', side_effect=execute) as run:
            result = self.run_batch()
            self.assertEqual([c.args[0].eval_id for c in run.call_args_list], ['E001', 'E002', 'E005', 'E006'])
            run.reset_mock()
            again = self.run_batch()
            run.assert_not_called()
        self.assertEqual(result['selected_ids'], self.before['selected_ids'])
        self.assertEqual(result['run_id'], self.before['run_id'])
        self.assertEqual(result['response_log'], self.before['response_log'])
        self.assertEqual(result['results'], again['results'])
        self.assertEqual([len(r['attempts']) for r in result['results']], [6, 3, 3, 1, 2, 2])
        for old, new in zip(self.before['results'], result['results']):
            self.assertEqual(new['attempts'][:len(old['attempts'])], old['attempts'])
        for index in (2, 3):
            self.assertEqual(result['results'][index], self.before['results'][index])
        log = [json.loads(line) for line in (self.root / result['response_log']).read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(log) - 1, sum(len(r['attempts']) for r in result['results']))
        self.assertEqual(result['results'][0]['attempts'][-1]['targeted_attempt_cap'], 6)
        self.assertEqual(result['first_attempt_summary']['failed'], 3)
        self.assertEqual(result['targeted_retry'], again['targeted_retry'])

    def test_provider_stop_resumes_remaining_batch_without_renewing_spent_allowance(self):
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: self.result(c, 'ERROR')) as run:
            stopped = self.run_batch()
            self.assertEqual(run.call_count, 1)
        self.assertEqual(stopped['stop_reason'], 'execution_error')
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: self.result(c)) as run:
            result = self.run_batch()
        self.assertEqual([c.args[0].eval_id for c in run.call_args_list], ['E002', 'E005', 'E006'])
        self.assertEqual(len(result['results'][0]['attempts']), 6)
        self.assertEqual(result['targeted_retry'], stopped['targeted_retry'])

    def test_killed_inflight_reservation_cannot_gain_another_attempt(self):
        with patch.object(runner, '_execute_case', side_effect=KeyboardInterrupt()):
            stopped = self.run_batch()
        self.assertEqual(stopped['stop_reason'], 'interrupted')
        self.assertIsNone(stopped['results'][0]['attempts'][-1]['finished_at'])
        with patch.object(evaluator, 'run_case', side_effect=lambda c, *_: self.result(c)) as run:
            result = self.run_batch()
        self.assertNotIn('E001', [c.args[0].eval_id for c in run.call_args_list])
        self.assertEqual(len(result['results'][0]['attempts']), 6)

    def test_selected_followup_uses_saved_parent_without_rerunning_pass(self):
        self.data.cases[4] = replace(self.data.cases[4], conversation_from='E004')
        histories = {}
        def execute(case, catalog, history):
            histories[case.eval_id] = history
            return self.result(case)
        with patch.object(evaluator, 'run_case', side_effect=execute):
            self.run_batch()
        self.assertNotIn('E004', histories)
        self.assertEqual(histories['E005'][-1]['content'], 'answer-E004')

    def test_error_field_overrides_fail_count_and_fresh_cases_are_not_selected(self):
        old = copy.deepcopy(self.before)
        old['results'][2]['error'] = 'LLM generation failed'
        old['results'].pop()
        batch = runner.targeted_retry_plan(old)
        self.assertIn('E003', batch['error_ids'])
        self.assertNotIn('E006', batch['attempt_caps'])
        self.assertNotIn('E004', batch['attempt_caps'])

    def test_mismatched_checkpoint_is_never_written(self):
        before = self.path.read_bytes()
        with patch.object(runner, 'fingerprint', return_value='changed'), self.assertRaises(ds.DatasetError):
            self.run_batch()
        self.assertEqual(self.path.read_bytes(), before)

    def test_no_new_report_or_pilot_allowed(self):
        with self.assertRaises(ds.DatasetError):
            runner.run(**self.args, targeted_retry=True)
        with self.assertRaises(ds.DatasetError):
            run_evaluation.execute('existing', phase='all', targeted_retry=True)


class TargetedMigrationTests(unittest.TestCase):
    def test_only_scheduler_change_preserves_checkpoint_compatibility(self):
        dataset, catalog = ds.load(), ds.PermissionCatalog()
        source = (ds.PACKAGE_ROOT / 'fixtures/runner_before_targeted_retries.py.txt').read_bytes()
        controls = {'index_snapshot': {'collection_name': 'offline'}}
        previous = {'configuration': controls, 'fingerprint': runner.fingerprint(
            dataset, catalog, controls, _source_overrides={'evaluation/runner.py': source})}
        self.assertTrue(runner.compatible_targeted_checkpoint(previous, dataset, catalog, controls))
        changed = copy.deepcopy(dataset)
        changed.cases[0] = replace(changed.cases[0], question='changed input')
        self.assertFalse(runner.compatible_targeted_checkpoint(previous, changed, catalog, controls))
        with patch.object(runner.config, 'GROQ_MODEL', 'changed-model'):
            self.assertFalse(runner.compatible_targeted_checkpoint(previous, dataset, catalog, controls))
        self.assertFalse(runner.compatible_targeted_checkpoint(previous, dataset, catalog,
                                                              {**controls, 'delay_s': 0}))


if __name__ == '__main__':
    unittest.main()
