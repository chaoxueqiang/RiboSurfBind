#!/usr/bin/env python3
"""Prepare single/batch RNA records with optional labels and append-safe manifests.

This command prepares new inputs only; it never rebuilds paper benchmark inputs.
"""
import argparse
import csv
import fcntl
import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from data_process.geometry import _chain_records

SAMPLE_FIELDS = ['sample_id', 'pdb_path', 'model_index', 'chain_id', 'sequence', 'length', 'split']
MAPPING_FIELDS = ['sample_id', 'sequence_index', 'pdb_chain_id', 'pdb_residue_number',
                  'pdb_insertion_code', 'sequence_base']


def read_table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        sample = handle.read(4096)
        if not sample.strip():
            raise ValueError('Empty input table: ' + str(path))
        handle.seek(0)
        delimiter = '\t' if '\t' in sample.splitlines()[0] else ','
        return list(csv.DictReader(handle, delimiter=delimiter))


def atomic_table(path, rows, fields):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_chains(value):
    chains = [c.strip() for c in value.split(',')]
    if not chains or any(not c for c in chains) or len(set(chains)) != len(chains):
        raise ValueError('chains must be an ordered comma-separated list without duplicates')
    return chains


def imported_labels(path, record_id, residue_ids):
    rows = read_table(path)
    if not rows or not {'residue_id', 'label'}.issubset(rows[0]):
        raise ValueError('Label table requires residue_id,label (and optional record_id)')
    if 'record_id' in rows[0]:
        rows = [r for r in rows if r['record_id'] == record_id]
    values = {}
    for row in rows:
        rid = row['residue_id']
        if rid in values or row['label'] not in ('0', '1'):
            raise ValueError('Duplicate residue identity or non-binary label: ' + rid)
        values[rid] = int(row['label'])
    if set(values) != set(residue_ids):
        raise ValueError('Labels must cover every selected RNA residue exactly: ' + record_id)
    return np.asarray([values[rid] for rid in residue_ids], dtype=np.float32)


def contact_labels(path, residue_atoms, cutoff, boundary):
    from Bio.PDB import MMCIFParser, PDBParser
    suffixes = [s.lower() for s in path.suffixes]
    parser = MMCIFParser(QUIET=True) if any(s in ('.cif', '.mmcif') for s in suffixes) else PDBParser(QUIET=True)
    if suffixes[-1:] == ['.gz']:
        with gzip.open(path, 'rt') as handle:
            structure = parser.get_structure('ligand', handle)
    else:
        structure = parser.get_structure('ligand', str(path))
    models = list(structure)
    if len(models) != 1:
        raise ValueError('The explicit ligand file must contain exactly one structural model')
    atoms = [np.asarray(a.coord, dtype=np.float64) for a in models[0].get_atoms()
             if (a.element or a.name[:1]).upper() not in ('H', 'D')]
    if not atoms:
        raise ValueError('No ligand heavy atoms found')
    ligand = np.stack(atoms)
    labels = []
    for residue in residue_atoms:
        rna = np.stack([a['coordinate'] for a in residue]).astype(np.float64)
        distance = np.sqrt(((rna[:, None] - ligand[None]) ** 2).sum(axis=-1)).min()
        labels.append(distance <= cutoff if boundary == 'inclusive' else distance < cutoff)
    return np.asarray(labels, dtype=np.float32)


def prepare_entry(entry):
    rid = entry['record_id']
    if not rid or not rid.replace('_', '').replace('-', '').isalnum():
        raise ValueError('record_id must contain only letters, digits, underscores and hyphens')
    model_index = int(entry.get('model_index') or 0)
    if model_index < 0:
        raise ValueError('model_index cannot be negative')
    chains = parse_chains(entry['chains'])
    if any(len(c) != 1 for c in chains):
        raise ValueError('Legacy surface export requires single-character chain IDs; normalize chain IDs first')
    available = {r['chain_id']: r for r in _chain_records(Path(entry['pdb']), model_index)}
    if any(c not in available for c in chains):
        raise ValueError('Requested RNA chain is absent: ' + rid)
    excluded = set(filter(None, (entry.get('exclude_residues') or '').split(';')))
    sequence, coordinates, identifiers, atoms = [], [], [], []
    observed = set()
    for chain in chains:
        record = available[chain]
        for base, xyz, ident, residue_atoms in zip(record['sequence'], record['coordinates'],
                                                  record['residue_identifiers'], record['residue_atoms']):
            key = f"{chain}:{ident['component_number']}:{ident['insertion_code'] or ''}"
            observed.add(key)
            if key in excluded:
                continue
            sequence.append(base)
            coordinates.append(xyz)
            identifiers.append((key, chain, ident))
            atoms.append(residue_atoms)
    if excluded - observed or not sequence:
        raise ValueError('Unknown excluded residue or empty RNA selection: ' + rid)
    residue_ids = [r[0] for r in identifiers]
    if len(set(residue_ids)) != len(residue_ids):
        raise ValueError('Ambiguous structural residue identities: ' + rid)
    labels_path, ligand_path = entry.get('labels'), entry.get('ligand_pdb')
    if labels_path and ligand_path:
        raise ValueError('Choose imported labels OR an explicit ligand file, not both')
    payload = dict(record_id=np.asarray(rid), sequence=np.asarray(''.join(sequence)),
                   xyz=np.asarray(coordinates, dtype=np.float32), residue_ids=np.asarray(residue_ids, dtype=str))
    if labels_path:
        payload['y'] = imported_labels(Path(labels_path), rid, residue_ids)
        payload['label_source'] = np.asarray('identity_matched_annotation')
    elif ligand_path:
        if Path(ligand_path).resolve() == Path(entry['pdb']).resolve():
            raise ValueError('Provide a separate file containing only the selected ligand')
        raw_cutoff = entry.get('cutoff')
        cutoff = float(4.0 if raw_cutoff in (None, '') else raw_cutoff)
        boundary = entry.get('contact_boundary') or 'inclusive'
        if not np.isfinite(cutoff) or cutoff <= 0 or boundary not in ('inclusive', 'exclusive'):
            raise ValueError('Invalid contact cutoff or boundary')
        payload['y'] = contact_labels(Path(ligand_path), atoms, cutoff, boundary)
        payload['label_source'] = np.asarray('explicit_ligand_heavy_atom_contacts')
        payload['label_cutoff_angstrom'] = np.asarray(cutoff)
        payload['label_contact_boundary'] = np.asarray(boundary)
    split = entry.get('split') or 'prediction'
    if split not in ('prediction', 'train', 'validation', 'test'):
        raise ValueError('split must be prediction, train, validation, or test')
    if split != 'prediction' and 'y' not in payload:
        raise ValueError('Labelled train/validation/test records require labels: ' + rid)
    sample = dict(sample_id=rid, pdb_path=str(Path(entry['pdb']).resolve()), model_index=model_index,
                  chain_id=','.join(chains), sequence=''.join(sequence), length=len(sequence), split=split)
    mappings = [dict(sample_id=rid, sequence_index=i, pdb_chain_id=chain,
                     pdb_residue_number=ident['component_number'],
                     pdb_insertion_code=ident['insertion_code'] or '', sequence_base=sequence[i])
                for i, (_, chain, ident) in enumerate(identifiers)]
    return payload, sample, mappings


def write_entries(entries, output, overwrite=False):
    prepared = [prepare_entry(entry) for entry in entries]
    ids = [p[1]['sample_id'] for p in prepared]
    if not prepared or len(set(ids)) != len(ids):
        raise ValueError('Empty batch or duplicate record_id')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.prepare_records.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, records = output / 'surface_manifest', output / 'records'
        paths = [manifest / name for name in ('samples.tsv', 'residue_mapping.tsv')]
        if paths[0].exists() != paths[1].exists():
            raise ValueError('Incomplete existing surface manifest; repair it before adding records')
        samples = read_table(paths[0]) if paths[0].exists() else []
        mappings = read_table(paths[1]) if paths[1].exists() else []
        sample_ids = [r['sample_id'] for r in samples]
        record_ids = {p.stem for p in records.glob('*.npz')}
        if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != record_ids:
            raise ValueError('Existing records and manifest disagree; use a clean output or repair the manifest')
        if {r['sample_id'] for r in mappings} != set(sample_ids):
            raise ValueError('Existing residue mapping and samples disagree')
        for rid in ids:
            if rid in record_ids and not overwrite:
                raise FileExistsError('Record exists; use a new ID or explicit --overwrite: ' + rid)
            if (output / 'surface' / (rid + '.npz')).exists():
                raise FileExistsError('Surface cache exists; use a fresh output to avoid stale features: ' + rid)
        records.mkdir(exist_ok=True)
        manifest.mkdir(exist_ok=True)
        for payload, sample, mapping in prepared:
            rid = sample['sample_id']
            temporary = records / (rid + '.npz.tmp')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, **payload)
            temporary.replace(records / (rid + '.npz'))
            samples = [r for r in samples if r['sample_id'] != rid] + [sample]
            mappings = [r for r in mappings if r['sample_id'] != rid] + mapping
        atomic_table(paths[0], samples, SAMPLE_FIELDS)
        atomic_table(paths[1], mappings, MAPPING_FIELDS)
    for payload, sample, _ in prepared:
        print('Prepared', sample['sample_id'], sample['length'], 'nucleotides;',
              'labels included' if 'y' in payload else 'unlabelled prediction', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='CSV/TSV batch: record_id,pdb,chains plus optional fields')
    parser.add_argument('--pdb', type=Path)
    chain = parser.add_mutually_exclusive_group()
    chain.add_argument('--chain', help='One chain; legacy CLI remains supported')
    chain.add_argument('--chains', help='Ordered comma-separated chains, e.g. A,B')
    parser.add_argument('--record-id')
    parser.add_argument('--model-index', type=int, default=0)
    parser.add_argument('--split', choices=['prediction', 'train', 'validation', 'test'], default='prediction')
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument('--labels', type=Path, help='CSV/TSV residue_id,label; exact structural identity matching')
    labels.add_argument('--ligand-pdb', type=Path, help='Separate selected-ligand-only structure; for labels only')
    parser.add_argument('--cutoff', type=float, default=4.0)
    parser.add_argument('--contact-boundary', choices=['inclusive', 'exclusive'], default='inclusive')
    parser.add_argument('--exclude-residues', default='', help='Semicolon-separated chain:number:insertion IDs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.manifest:
        if any(v is not None for v in (args.pdb, args.chain, args.chains, args.record_id, args.labels, args.ligand_pdb)):
            parser.error('Batch --manifest cannot be combined with single-record input options')
        entries = read_table(args.manifest)
        if not entries or not {'record_id', 'pdb', 'chains'}.issubset(entries[0]):
            parser.error('Batch manifest requires record_id,pdb,chains')
        for entry in entries:
            for name in ('pdb', 'labels', 'ligand_pdb'):
                if entry.get(name):
                    path = Path(entry[name])
                    entry[name] = str(path if path.is_absolute() else args.manifest.parent / path)
    else:
        if not args.pdb or not (args.chain or args.chains) or not args.record_id:
            parser.error('Provide --manifest OR --pdb, --chain/--chains, --record-id')
        entries = [dict(record_id=args.record_id, pdb=str(args.pdb), chains=args.chains or args.chain,
                        model_index=args.model_index, split=args.split,
                        labels=str(args.labels) if args.labels else None,
                        ligand_pdb=str(args.ligand_pdb) if args.ligand_pdb else None,
                        cutoff=args.cutoff, contact_boundary=args.contact_boundary,
                        exclude_residues=args.exclude_residues)]
    try:
        write_entries(entries, args.output, args.overwrite)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
