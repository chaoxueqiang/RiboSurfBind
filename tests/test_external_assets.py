"""Guards for external-download assets and non-destructive publication exports."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_assets
from scripts.check_external_assets import verify_asset
from scripts.export_public_snapshot import excluded, export_snapshot


class ExternalAssetTests(unittest.TestCase):
    def fixture(self, root):
        (root/'docs').mkdir()
        (root/'data.bin').write_bytes(b'data')
        normal = dict(path='data.bin', bytes=4, sha256=hashlib.sha256(b'data').hexdigest())
        optional = dict(path='external.bin', bytes=8, sha256=hashlib.sha256(b'external').hexdigest())
        with (root/'docs/release_asset_checksums.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['path','bytes','sha256'])
            writer.writeheader()
            writer.writerows([normal, optional])
        (root/'docs/asset_aliases.json').write_text('[]')
        (root/'docs/external_assets.json').write_text(json.dumps(dict(assets=dict(ernie=optional))))

    def run_verify(self, root, *arguments):
        with patch.object(verify_assets, 'ROOT', root), patch.object(sys, 'argv', ['verify_assets', *arguments]):
            with contextlib.redirect_stdout(io.StringIO()):
                verify_assets.main()

    def test_only_declared_missing_external_is_optional(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            self.run_verify(root)
            with self.assertRaises(SystemExit):
                self.run_verify(root, '--include-external')
            (root/'data.bin').unlink()
            with self.assertRaises(SystemExit):
                self.run_verify(root)

    def test_present_but_wrong_optional_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            (root/'external.bin').write_bytes(b'wrong')
            with self.assertRaises(SystemExit):
                self.run_verify(root)

    def test_explicit_manifest_stays_strict(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            with self.assertRaises(SystemExit):
                self.run_verify(root, '--manifest', str(root/'docs/release_asset_checksums.csv'))

    def test_external_hash_check_and_export_no_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root/'asset'
            path.write_bytes(b'asset')
            specification = dict(bytes=5, sha256=hashlib.sha256(b'asset').hexdigest())
            verify_asset(path, specification)
            specification['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                verify_asset(path, specification)
            with self.assertRaises(ValueError):
                export_snapshot(root/'source', root)
            self.assertEqual(path.read_bytes(), b'asset')

    def test_publication_policy_has_no_tracked_external_files(self):
        policy = json.loads((ROOT/'docs/external_assets.json').read_text())
        rules = policy['release_exclusions']
        self.assertTrue(excluded('third_party/surface_tools/msms-2.6.1/msms.html', rules))
        self.assertTrue(excluded('checkpoints/ernie_rna/ERNIE-RNA_pretrain.pt', rules))
        self.assertFalse(excluded('checkpoints/train60/seed1/best_model.pt', rules))
        self.assertFalse(excluded('data/processed/hariboss.pt', rules))
        if (ROOT/'.git').exists():
            tracked = subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z','--',*rules])
            self.assertEqual(tracked, b'')
        if (ROOT/'RELEASE_SNAPSHOT.json').exists():
            # A fresh publication Git repository is allowed. The exporter must
            # not have copied the experiment history or its external assets.
            report = json.loads((ROOT/'RELEASE_SNAPSHOT.json').read_text())
            self.assertFalse(report['git_history_copied'])
            if (ROOT/'.git').exists():
                history = subprocess.check_output(
                    ['git','-C',str(ROOT),'rev-list','--all','--',*rules])
                self.assertEqual(history, b'')
            for relative in rules:
                self.assertFalse((ROOT/relative.rstrip('/')).exists())


if __name__ == '__main__':
    unittest.main()
