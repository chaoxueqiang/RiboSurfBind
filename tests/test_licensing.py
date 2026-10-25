"""Standard-library-only tests of the advisory redistribution checker."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('redistribution_check', ROOT / 'scripts/check_redistribution.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class LicensingTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'docs').mkdir()
        # Temporary fixtures are test outputs, not project/license edits.
        (root / 'notice.txt').write_text('fixture notice\n', encoding='utf-8')
        inventory = dict(
            schema='ribosurfbind_third_party_review_v1', release_ready=False,
            license_documents=[dict(path='notice.txt', retained_sha256=hashlib.sha256(b'fixture notice\n').hexdigest())],
            components=[dict(id='pending', blocking=True, status='needs_permission', scope=['file'],
                             reason='No grant recorded.', action='Obtain applicable permission.')],
            restricted_history_paths=['restricted/'],
        )
        self.save(root, inventory)
        return inventory

    def save(self, root, inventory):
        (root / 'docs/third_party_licenses.json').write_text(json.dumps(inventory), encoding='utf-8')

    def test_actual_inventory_has_preserved_notices_and_unresolved_items(self):
        errors, blockers = CHECK.audit(ROOT)
        self.assertEqual(errors, [])
        inventory = json.loads((ROOT / 'docs/third_party_licenses.json').read_text(encoding='utf-8'))
        expected = {item['id'] for item in inventory['components'] if item['blocking']}
        self.assertEqual({message.split(':', 1)[0] for message in blockers}, expected)

    def test_missing_and_altered_notice_fail(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            inventory = self.fixture(root)
            (root / 'notice.txt').write_text('altered', encoding='utf-8')
            self.assertEqual(len(CHECK.audit(root)[0]), 1)
            inventory['license_documents'][0]['path'] = 'missing.txt'
            self.save(root, inventory)
            self.assertEqual(len(CHECK.audit(root)[0]), 1)

    def test_author_approved_mit_is_scoped_to_original_code(self):
        text = (ROOT / 'LICENSE').read_text(encoding='utf-8')
        self.assertTrue(text.startswith('MIT License\n'))
        self.assertIn('Copyright (c) 2026 Xueqiang Chao', text)
        inventory = json.loads((ROOT / 'docs/third_party_licenses.json').read_text(encoding='utf-8'))
        components = {item['id']: item for item in inventory['components']}
        self.assertFalse(components['original_project']['blocking'])
        self.assertEqual(components['original_project']['status'], 'author_approved_original_code')
        self.assertEqual(components['original_project']['copyright'],
                         'Copyright (c) 2026 Xueqiang Chao')
        self.assertEqual(components['original_project']['license'],
                         'MIT for original code and associated documentation only')
        # This approval does not itself change the recorded third-party grants.
        self.assertNotEqual(components['msms_binary']['license'], 'MIT')
        self.assertNotEqual(components['dmasif_lineage']['license'], 'MIT')

    def test_clearance_cannot_contradict_pending_review(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            inventory = self.fixture(root)
            inventory['release_ready'] = True
            self.save(root, inventory)
            with self.assertRaises(ValueError):
                CHECK.audit(root)

    def test_evidence_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            inventory = self.fixture(root)
            inventory['license_documents'][0]['path'] = '../notice.txt'
            self.save(root, inventory)
            with self.assertRaises(ValueError):
                CHECK.audit(root)

    def test_history_remains_blocking_even_after_current_file_is_absent(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            result = subprocess.CompletedProcess([], 0, stdout='abc123\n', stderr='')
            with patch.object(CHECK.subprocess, 'run', return_value=result):
                errors, blockers = CHECK.audit(root, check_history=True)
            self.assertEqual(errors, [])
            self.assertEqual(len(blockers), 2)
            self.assertIn('Git history', blockers[-1])


if __name__ == '__main__':
    unittest.main()
