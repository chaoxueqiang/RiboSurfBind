"""Guards for preprocessing simplification and lossless asset sharing."""
import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from data_process.attention import make_sequence_key
from data_process.dataset import sequence_key
from data_process.geometry import _align_record
from scripts.verify_assets import verify_aliases


class CleanupTests(unittest.TestCase):
    def test_shared_sequence_key_preserves_cache_names(self):
        self.assertIs(sequence_key, make_sequence_key)
        for sequence in ['', 'A', 'AUGC', 'auGC', 'AUGCN']:
            expected = 'seq_'+hashlib.sha256(sequence.encode('utf-8')).hexdigest()
            self.assertEqual(make_sequence_key(sequence), expected)

    def test_alignment_needs_only_current_geometry(self):
        record = dict(sequence='AGU', coordinates=np.arange(9, dtype=np.float32).reshape(3, 3),
                      residue_atoms=[['A'], ['G'], ['U']])
        for sequence, status, count in [
            ('AGU', 'exact', 3),
            ('CGU', 'explicit_terminal_substitution_index0', 3),
            ('AG', 'explicit_extra_terminal_residue_removed', 2),
        ]:
            xyz, observed_status, atoms = _align_record(record, sequence, 'fixture')
            self.assertEqual(observed_status, status)
            np.testing.assert_array_equal(xyz.numpy(), record['coordinates'][:count])
            self.assertEqual(atoms, record['residue_atoms'][:count])
        with self.assertRaises(ValueError):
            _align_record(record, 'ACU', 'fixture')
        with self.assertRaises(ValueError):
            _align_record(record, 'A', 'fixture')

    def test_shared_assets_and_checkpoint_provenance(self):
        self.assertEqual(verify_aliases(ROOT), 8)
        aliases = {r['path']:r for r in json.loads((ROOT/'docs/asset_aliases.json').read_text())}
        rows = json.loads((ROOT/'docs/artifact_manifest.json').read_text())
        self.assertEqual(len({(ROOT/r['target']).resolve() for r in rows}), 53)
        for row in rows:
            if row['target'] in aliases:
                self.assertEqual(row['shared_with'], aliases[row['target']]['target'])
                self.assertEqual(row['sha256'], aliases[row['target']]['sha256'])

    def test_generated_figures_are_not_frozen_assets(self):
        with (ROOT/'docs/result_asset_checksums.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertFalse(any(r['path'].startswith('results/reproduced_figures/') for r in rows))


if __name__ == '__main__':
    unittest.main()
