"""Regression tests for preparation and evaluation guards; no GPU required."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from data_process.prepare_records import write_entries, read_table, contact_labels
from data_process.dataset import load_protocol
from utils.checkpoints import evaluation_set


def chain_record(chain, offset=0):
    return dict(chain_id=chain, sequence='AG', coordinates=np.asarray([[offset, 0, 0], [offset+1, 0, 0]], dtype=np.float32),
                residue_identifiers=[dict(component_number=i, insertion_code=None) for i in [1, 2]],
                residue_atoms=[[dict(coordinate=np.asarray([offset+i, 0, 0], dtype=np.float32))] for i in [0, 1]])


class WorkflowTests(unittest.TestCase):
    def test_append_keeps_all_samples_and_mapping(self):
        with tempfile.TemporaryDirectory() as td, patch('data_process.prepare_records._chain_records', return_value=[chain_record('A')]):
            root = Path(td)
            for rid in ['first', 'second']:
                write_entries([dict(record_id=rid, pdb='rna.pdb', chains='A')], root)
            self.assertEqual([r['sample_id'] for r in read_table(root/'surface_manifest/samples.tsv')], ['first', 'second'])
            self.assertEqual(len(read_table(root/'surface_manifest/residue_mapping.tsv')), 4)
            before = (root/'records/first.npz').read_bytes()
            with self.assertRaises(FileExistsError):
                write_entries([dict(record_id='first', pdb='rna.pdb', chains='A')], root)
            self.assertEqual(before, (root/'records/first.npz').read_bytes())

    def test_multichain_order_and_identity_labels(self):
        with tempfile.TemporaryDirectory() as td, patch('data_process.prepare_records._chain_records', return_value=[chain_record('A'), chain_record('B', 10)]):
            root = Path(td)
            labels = root/'labels.tsv'
            labels.write_text('residue_id\tlabel\nA:2:\t1\nB:1:\t0\nA:1:\t0\nB:2:\t1\n')
            write_entries([dict(record_id='multi', pdb='rna.pdb', chains='B,A', labels=str(labels), split='train')], root/'out')
            with np.load(root/'out/records/multi.npz') as data:
                self.assertEqual(data['residue_ids'].tolist(), ['B:1:', 'B:2:', 'A:1:', 'A:2:'])
                np.testing.assert_array_equal(data['y'], [0, 1, 0, 1])
                np.testing.assert_array_equal(data['xyz'][:, 0], [10, 11, 0, 1])

    def test_incomplete_labels_and_invalid_batch_do_not_commit(self):
        with tempfile.TemporaryDirectory() as td, patch('data_process.prepare_records._chain_records', return_value=[chain_record('A')]):
            root = Path(td)
            labels = root/'labels.csv'
            labels.write_text('residue_id,label\nA:1:,1\n')
            with self.assertRaises(ValueError):
                write_entries([dict(record_id='bad', pdb='rna.pdb', chains='A', labels=str(labels))], root/'out')
            self.assertFalse((root/'out/records').exists())
            with self.assertRaises(ValueError):
                write_entries([dict(record_id='same', pdb='rna.pdb', chains='A')]*2, root/'out')

    def test_stale_surface_and_broken_existing_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as td, patch('data_process.prepare_records._chain_records', return_value=[chain_record('A')]):
            root = Path(td)
            entry = dict(record_id='one', pdb='rna.pdb', chains='A')
            write_entries([entry], root)
            (root/'surface').mkdir()
            (root/'surface/one.npz').touch()
            with self.assertRaises(FileExistsError):
                write_entries([entry], root, overwrite=True)
            (root/'records/orphan.npz').touch()
            with self.assertRaises(ValueError):
                write_entries([dict(record_id='new', pdb='rna.pdb', chains='A')], root)

    def test_contact_cutoff_boundary_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            ligand = Path(td)/'ligand.pdb'
            ligand.write_text('HETATM    1  C1  LIG Z   1       4.000   0.000   0.000  1.00  0.00           C\nEND\n')
            atoms = [[dict(coordinate=np.zeros(3))]]
            np.testing.assert_array_equal(contact_labels(ligand, atoms, 4.0, 'inclusive'), [1])
            np.testing.assert_array_equal(contact_labels(ligand, atoms, 4.0, 'exclusive'), [0])

    def test_evaluation_protocol_and_set_guards(self):
        config = dict(protocol='hariboss', set_id=2, seed=1)
        self.assertEqual(evaluation_set(config, 'hariboss'), 2)
        with self.assertRaises(ValueError):
            evaluation_set(config, 'hariboss', 1)
        with self.assertRaises(ValueError):
            evaluation_set(config, 'test18')
        with self.assertRaises(ValueError):
            evaluation_set(config, 'hariboss', 2, dict(config, seed=5))
        with self.assertRaises(ValueError):
            evaluation_set(dict(protocol='external', set_id=1), 'external', 2)

    def test_custom_split_guard_and_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'splits').mkdir()
            path = root/'splits/custom.json'
            split = dict(train=['b'], validation=['a'], test=['c'])
            path.write_text(json.dumps(split))
            graphs = [SimpleNamespace(node_feature_record_id=i) for i in ['a', 'b', 'c']]
            with patch('data_process.dataset.load_group', return_value=(graphs, {})):
                data, _, _ = load_protocol('custom', data_root=root)
                self.assertEqual(data['train'][0].node_feature_record_id, 'b')
                path.write_text(json.dumps(dict(split, test=['a'])))
                with self.assertRaises(ValueError):
                    load_protocol('custom', data_root=root)

    def test_hariboss_results_follow_current_seeds(self):
        seeds = set(json.loads((ROOT/'configs/protocols.json').read_text())['hariboss']['paper_seeds'])
        with (ROOT/'results/published_metrics.csv').open() as f:
            rows = list(csv.DictReader(f))
        for fold in range(1, 5):
            selected = [r for r in rows if r['experiment'] == 'hariboss_set%d' % fold]
            self.assertEqual(len(selected), len(seeds))
            self.assertEqual({int(r['seed']) for r in selected}, seeds)
            for row in selected:
                path = ROOT/'results/raw/hariboss'/('set%d' % fold)/('seed'+row['seed'])/'result.json'
                metrics = json.loads(path.read_text())['evaluations']['validation_loss']['test']['fixed_0_5']
                for metric in ['auc', 'precision', 'recall', 'mcc']:
                    self.assertAlmostEqual(float(row[metric]), metrics[metric], places=14)


if __name__ == '__main__':
    unittest.main()
