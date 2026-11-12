import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('npu_access', Path(__file__).parents[1] / 'tools/npu_access.py')
access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access)


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stage = self.root / 'stage'
        self.rule = self.root / 'live.rules'
        self.patchers = [patch.object(access, 'RULE', self.rule), patch.object(access, 'HEAP', Path('/dev/null'))]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)

    def test_only_exact_heap_read_permission(self):
        text = access.rule_for(1001)
        self.assertIn('KERNEL=="system"', text)
        self.assertIn('SUBSYSTEM=="dma_heap"', text)
        self.assertIn('MODE:="0400"', text)
        self.assertNotIn('uaccess', text)
        for uid in [0, -1, '1001']:
            with self.assertRaises(ValueError):
                access.rule_for(uid)

    def test_staging_has_no_privileged_mutation(self):
        with patch.object(access, 'sudo') as privileged:
            access.stage_at(self.stage)
            privileged.assert_not_called()
        m = access.check_stage(self.stage)
        self.assertIsNone(m['old_sha256'])
        self.assertEqual(self.stage.stat().st_mode & 0o777, 0o700)

    def test_changed_staged_grant_refused(self):
        access.stage_at(self.stage)
        (self.stage / 'rule').write_text('KERNEL=="*", MODE="0666"\n')
        with patch.object(access, 'sudo') as privileged:
            with self.assertRaisesRegex(ValueError, 'staged rule changed'):
                access.apply(self.stage)
            privileged.assert_not_called()

    def test_live_preimage_change_refused(self):
        access.stage_at(self.stage)
        self.rule.write_text('another release\n')
        with patch.object(access, 'sudo') as privileged:
            with self.assertRaisesRegex(ValueError, 'preimage changed'):
                access.apply(self.stage)
            privileged.assert_not_called()

    def test_stale_rollback_refused(self):
        access.stage_at(self.stage)
        self.rule.write_text('newer installed release\n')
        with patch.object(access, 'sudo') as privileged:
            with self.assertRaisesRegex(ValueError, 'stale rollback'):
                access.rollback(self.stage)
            privileged.assert_not_called()

    def test_saved_acl_integrity_checked(self):
        access.stage_at(self.stage)
        self.rule.write_bytes((self.stage / 'rule').read_bytes())
        (self.stage / 'heap.original.acl').write_text('changed\n')
        with patch.object(access, 'sudo') as privileged:
            with self.assertRaisesRegex(ValueError, 'original ACL changed'):
                access.rollback(self.stage)
            privileged.assert_not_called()


if __name__ == '__main__':
    unittest.main()
