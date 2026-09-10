"""Offline publication checks only: no SSH, Docker, or model inference."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import launch_node
import verify_saved_results as verifier

class PublishedRecipeTests(unittest.TestCase):
    def test_frozen_sources_unchanged(self):
        for row in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text()):
            self.assertEqual(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(), row['sha256'], row['path'])

    def test_complete_raw_evidence_matrix(self):
        report = verifier.verify_campaign(ROOT/'evidence/text128-dflash-resume')
        self.assertEqual(report['verified_speed_samples'], 12)
        self.assertFalse(report['missing_speed_samples'])
        self.assertTrue(all(arm['long_context_verified'] for arm in report['arms']))

    def test_commands_preserve_context_and_acceleration(self):
        for rank, ip in launch_node.HOSTS.values():
            for depth in (0, 3, 7):
                command = launch_node.command(rank, ip, depth)
                self.assertEqual(command[command.index('--tensor-parallel-size')+1], '2')
                self.assertEqual(command[command.index('--max-model-len')+1], '131072')
                self.assertIn('--enforce-eager', command)
                self.assertIn('--language-model-only', command)
                self.assertIn('MAX_JOBS=1', command)
                if depth:
                    spec = json.loads(command[command.index('--speculative-config')+1])
                    self.assertEqual(spec['method'], 'dflash')
                    self.assertEqual(spec['num_speculative_tokens'], depth)
                else:
                    self.assertNotIn('--speculative-config', command)

    def test_quality_failures_retained(self):
        records = json.loads((ROOT/'evidence/text128-dflash-resume/comparison.json').read_text())
        self.assertEqual(len(records), 3)
        self.assertTrue(all(not row['quality_passed'] for row in records))
        self.assertTrue(next(r for r in records if r['variant']=='dflash2-7')['semantic_checks_passed'])
        self.assertFalse(next(r for r in records if r['variant']=='dflash2-3')['semantic_checks_passed'])

    def test_altered_receipts_and_requests_rejected(self):
        arm = ROOT/'evidence/text128-dflash-resume/dflash2-7'
        label = 'speed-prose-0'
        expected = json.loads((arm/f'{label}-request.json').read_text())
        original_load = verifier.load
        for field in ('content', 'usage', 'decode_est_tps', 'request'):
            def altered(path):
                value = original_load(path)
                if path.name == f'{label}-result.json':
                    if field == 'content': value['content'] += 'test-only alteration'
                    elif field == 'usage': value['usage']['completion_tokens'] += 1
                    elif field == 'decode_est_tps': value['decode_est_tps'] *= 2
                if field == 'request' and path.name == f'{label}-request.json':
                    value['seed'] += 1
                return value
            with self.subTest(field=field), patch.object(verifier, 'load', side_effect=altered):
                with self.assertRaises(AssertionError):
                    verifier.verify_sample(arm, label, expected)

if __name__ == '__main__':
    unittest.main()
