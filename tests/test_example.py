"""Fast, compilation-free checks for the supplied real-RNA example."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from data_process.packed import read_package
from utils.io import sha256_file

EXAMPLE = ROOT/'examples/2JUK_A'


class ExampleTests(unittest.TestCase):
    def test_example_tensors_match_frozen_test18_input(self):
        manifest = json.loads((EXAMPLE/'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            self.assertEqual(sha256_file(EXAMPLE/name), digest)
        self.assertEqual(sha256_file(ROOT/manifest['checkpoint']), manifest['checkpoint_sha256'])
        source = next(s for s in read_package(ROOT/'data/processed/test18.pt')['samples']
                      if s['record_id'] == manifest['source_record_id'])
        demo = read_package(EXAMPLE/'2JUK_A.pt', require_labels=False)['samples'][0]
        self.assertNotIn('y', demo)
        self.assertEqual(demo['record_id'], '2JUK_A')
        self.assertEqual(demo['sequence'], source['sequence'])
        self.assertEqual(demo['residue_ids'], source['residue_ids'])
        for key in ['pos', 'embedding', 'attention']:
            self.assertTrue(torch.equal(demo[key], source[key]), key)
        for key in ['xyz', 'nuv', 'chemistry', 'atom_name_index', 'owner']:
            self.assertTrue(torch.equal(demo['surface'][key], source['surface'][key]), key)
        self.assertEqual(len(demo['sequence']), 22)

    def test_reference_checker_rejects_modified_predictions(self):
        spec = importlib.util.spec_from_file_location('demo_check', EXAMPLE/'run_example.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference = EXAMPLE/'expected_predictions.csv'
        self.assertEqual(module.compare_predictions(reference, reference, 1e-4)['status'], 'PASS')
        with reference.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        rows[0]['residue_id'] = 'A:999:'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'bad.csv'
            with output.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                module.compare_predictions(output, reference, 1e-4)


if __name__ == '__main__':
    unittest.main()
