#!/usr/bin/env python3
"""Losslessly package existing records/ERNIE/surface caches; never re-extract.

For a custom RNA: --dataset-root data/raw/custom --name custom
--output data/processed/custom.pt. For benchmark migration: --source-root
PATH --output-root data [--include-raw]. Existing outputs are never overwritten.
"""
import argparse
import csv
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from data_process.attention import load_attention_record
from data_process.dataset import load_record, sequence_key
from data_process.packed import SCHEMA, validate_sample
from utils.io import read_json, write_json, sha256_file


def _metadata(value):
    """Keep provenance readable and weights_only-loadable without NumPy pickles."""
    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return {str(k): _metadata(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_metadata(v) for v in value]
    if isinstance(value, np.ndarray):
        return _metadata(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def sample_from_record(path, dataset_root):
    root = Path(dataset_root)
    graph = load_record(path, root/'embeddings', root/'attention', root/'surface', require_labels=False)
    rid = graph.node_feature_record_id
    key = sequence_key(graph.rna_sequence)
    attention = load_attention_record(root/'attention', graph.rna_sequence, int(graph.num_nodes))
    embedding = torch.load(root/'embeddings'/f'{key}.pt', map_location='cpu', weights_only=False)
    with np.load(root/'surface'/f'{rid}.npz', allow_pickle=False) as raw:
        surface = dict(record_id=rid, node_count=int(raw['node_count']),
                       xyz=torch.from_numpy(np.asarray(raw['surface_xyz'], dtype=np.float32)),
                       nuv=torch.from_numpy(np.asarray(raw['surface_nuv'], dtype=np.float32)),
                       chemistry=torch.from_numpy(np.asarray(raw['chemical_features'], dtype=np.float32)),
                       atom_name_index=torch.from_numpy(np.asarray(raw['atom_name_index'], dtype=np.int64)),
                       owner=torch.from_numpy(np.asarray(raw['owner_residue'], dtype=np.int64)),
                       assignment=str(raw['surface_atom_assignment'].item()))
        # Geometry/chemistry definitions remain documented by the original
        # preprocessing source. Retain small per-cache metadata when present.
        surface_metadata = {k: _metadata(raw[k]) for k in raw.files
                            if raw[k].ndim == 0 and k != 'node_count'}
    sample = dict(record_id=rid, display_name=graph.display_name, sequence=graph.rna_sequence,
                  residue_ids=list(graph.residue_ids), pos=graph.pos,
                  embedding=graph.ernie_rna_embedding, attention=attention['attention'],
                  surface=surface, provenance=dict(
                      attention=_metadata({k:v for k,v in attention.items() if k != 'attention'}),
                      embedding=_metadata({k:v for k,v in embedding.items() if k != 'embedding'})
                                if isinstance(embedding,dict) else {'metadata_available':False},
                      surface=surface_metadata))
    if getattr(graph, 'y', None) is not None:
        sample['y'] = graph.y
    validate_sample(sample, require_labels=False)
    return sample


def save_package(name, samples, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite RNA package: ' + str(output))
    if not samples or len({s['record_id'] for s in samples}) != len(samples):
        raise ValueError('Empty package or duplicate IDs: ' + name)
    counts = dict(records=len(samples), nucleotides=sum(len(s['sequence']) for s in samples),
                  surface_points=sum(s['surface']['xyz'].shape[0] for s in samples))
    if all('y' in s for s in samples):
        counts['positives'] = sum(int(s['y'].sum().item()) for s in samples)
        counts['negatives'] = counts['nucleotides']-counts['positives']
    payload = dict(schema=SCHEMA, name=name, counts=counts, samples=samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(output)
    print(name, counts, flush=True)
    return dict(file=str(output), sha256=sha256_file(output), bytes=output.stat().st_size, **counts)


def migrate_benchmarks(source, output, include_raw=False):
    source = Path(source).resolve()
    output = Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and destination must be independent directories')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('processed','splits','raw'):
        if (output/name).exists():
            raise FileExistsError('Migration destination already exists: ' + str(output/name))
    report = dict(schema=SCHEMA, feature_extraction_performed=False, packages={},
                  source_directory_name=source.name, raw_assets_copied=include_raw)
    inventory = []
    for group in ('train60_test18','hariboss','hariboss_tr60','tl12','jl10'):
        root = source/group
        split = read_json(root/'splits.json')
        samples = [sample_from_record(path,root) for path in sorted((root/'records').glob('*.npz'))]
        if group == 'train60_test18':
            train_ids = set(split['train']+split['validation'])
            test_ids = set(split['test18'])
            parts = [('train60',[s for s in samples if s['record_id'] in train_ids]),
                     ('test18',[s for s in samples if s['record_id'] in test_ids])]
            if len(train_ids | test_ids) != len(samples) or train_ids & test_ids:
                raise ValueError('Train60/Test18 sample mismatch')
        else:
            parts = [(group,samples)]
        for name, subset in parts:
            report['packages'][name] = save_package(name,subset,output/'processed'/f'{name}.pt')
            report['packages'][name]['file'] = f'processed/{name}.pt'
        if group == 'hariboss':
            for set_name, ids in split.items():
                write_json(output/'splits'/f'hariboss_{set_name}.json',ids)
        else:
            filename = 'train60' if group=='train60_test18' else group
            write_json(output/'splits'/f'{filename}.json',split)
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            inventory.append(dict(path=str(path.relative_to(source)),bytes=path.stat().st_size,
                                  sha256=sha256_file(path)))
            if include_raw and relative.parts[0] not in ('records','embeddings','attention','surface') \
                    and path.name != 'splits.json':
                if relative.parts[0] == 'structures':
                    target = output/'raw/structures'/group/Path(*relative.parts[1:])
                elif relative.parts[0] in ('labels','source'):
                    target = output/'raw/annotations'/group/relative
                else:
                    target = output/'raw/provenance'/group/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
    with (output/'source_checksums.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['path','bytes','sha256'])
        writer.writeheader();writer.writerows(inventory)
    report['original_files'] = len(inventory)
    report['original_bytes'] = sum(row['bytes'] for row in inventory)
    write_json(output/'manifest.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--source-root',type=Path,help='Original five-group benchmark directory')
    sources.add_argument('--dataset-root',type=Path,help='One prepared records/embeddings/attention/surface directory')
    parser.add_argument('--output-root',type=Path,default=ROOT/'data')
    parser.add_argument('--include-raw',action='store_true')
    parser.add_argument('--name',help='Custom package name')
    parser.add_argument('--output',type=Path,help='Custom .pt output file')
    parser.add_argument('--split-output',type=Path,help='Write explicit train/validation/test IDs from the surface manifest')
    args = parser.parse_args()
    if args.source_root:
        if args.output or args.name or args.split_output:
            parser.error('--name/--output belong to --dataset-root')
        migrate_benchmarks(args.source_root,args.output_root,args.include_raw)
    else:
        if not args.name or not args.output:
            parser.error('--dataset-root requires --name and --output')
        samples = [sample_from_record(p,args.dataset_root) for p in sorted((args.dataset_root/'records').glob('*.npz'))]
        splits = None
        if args.split_output:
            if args.split_output.resolve() == args.output.resolve():
                parser.error('Package and split manifest must be different files')
            if args.split_output.exists():
                raise FileExistsError('Refusing to overwrite split manifest: '+str(args.split_output))
            if args.name != 'custom' or args.output.name != 'custom.pt':
                parser.error('Custom training expects --name custom --output .../processed/custom.pt')
            with (args.dataset_root/'surface_manifest/samples.tsv').open(newline='') as handle:
                manifest = list(csv.DictReader(handle,delimiter='\t'))
            splits = {'train':[],'validation':[],'test':[]}
            for row in manifest:
                if row['split'] not in splits:
                    parser.error('Training split export requires explicit train/validation/test assignments')
                splits[row['split']].append(row['sample_id'])
            flat = [rid for ids in splits.values() for rid in ids]
            if any(not ids for ids in splits.values()) or len(flat) != len(set(flat)) \
                    or set(flat) != {s['record_id'] for s in samples} or any('y' not in s for s in samples):
                parser.error('Split manifest must cover every labelled sample exactly, with three nonempty splits')
        save_package(args.name,samples,args.output)
        if splits is not None:
            write_json(args.split_output,splits)


if __name__ == '__main__':
    main()
