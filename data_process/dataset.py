"""Published packed benchmarks and optional raw preparation records."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from data_process.attention import load_attention_record, make_sequence_key as sequence_key
from data_process.packed import RNAInputs, read_package
from utils.io import ROOT, read_json, resolve_path

@lru_cache(maxsize=512)
def read_embedding(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return torch.as_tensor(payload['embedding'] if isinstance(payload,dict) else payload, dtype=torch.float32)

def load_record(path, embedding_dir, attention_dir, surface_dir, require_labels=True):
    with np.load(path, allow_pickle=False) as record:
        rid = str(record['record_id'].item())
        sequence = str(record['sequence'].item())
        xyz = np.asarray(record['xyz'], dtype=np.float32)
        labels = np.asarray(record['y'], dtype=np.float32) if 'y' in record else None
        residue_ids = record['residue_ids'].astype(str).tolist() if 'residue_ids' in record else [str(i+1) for i in range(len(sequence))]
        display = str(record['display_name'].item()) if 'display_name' in record else rid
    n = len(sequence)
    if not n or xyz.shape != (n,3) or not np.isfinite(xyz).all() or len(residue_ids) != n:
        raise ValueError('Invalid residue record: '+rid)
    if labels is None and require_labels:
        raise ValueError('Missing benchmark labels: '+rid)
    if labels is not None and (labels.shape != (n,) or not np.isin(labels,[0,1]).all()):
        raise ValueError('Invalid binding labels: '+rid)
    embedding = read_embedding(str(Path(embedding_dir)/f'{sequence_key(sequence)}.pt'))
    if tuple(embedding.shape) != (n,768) or not torch.isfinite(embedding).all():
        raise ValueError('Invalid ERNIE embedding: '+rid)
    load_attention_record(Path(attention_dir), sequence, n)
    with np.load(Path(surface_dir)/f'{rid}.npz', allow_pickle=False) as surface:
        if int(surface['node_count']) != n:
            raise ValueError('Surface/residue mismatch: '+rid)
    graph = Data(x=torch.zeros(n,1), edge_index=torch.empty(2,0,dtype=torch.long),
                 pos=torch.from_numpy(xyz), ernie_rna_embedding=embedding,
                 rna_sequence=sequence, node_feature_record_id=rid,
                 residue_ids=residue_ids, display_name=display)
    if labels is not None:
        graph.y = torch.from_numpy(labels)
    return graph

def dataset_paths(group, data_root=None):
    root = resolve_path(data_root or 'data')
    names = ('train60', 'test18') if group == 'train60_test18' else (group,)
    return {'root': root, 'packages': [root/'processed'/f'{name}.pt' for name in names]}

def graph_from_sample(sample):
    n = len(sample['sequence'])
    graph = Data(x=torch.zeros(n,1), edge_index=torch.empty(2,0,dtype=torch.long),
                 pos=sample['pos'], ernie_rna_embedding=sample['embedding'],
                 rna_sequence=sample['sequence'], node_feature_record_id=sample['record_id'],
                 residue_ids=list(sample['residue_ids']), display_name=sample['display_name'],
                 rna_inputs=RNAInputs(sample['attention'],sample['surface']))
    if sample.get('y') is not None:
        graph.y = sample['y']
    return graph

def load_package_graphs(path, require_labels=False):
    return [graph_from_sample(sample) for sample in read_package(path,require_labels)['samples']]

def load_group(group, ids=None, data_root=None, require_labels=True):
    paths = dataset_paths(group,data_root)
    samples = [sample for path in paths['packages']
               for sample in read_package(path,require_labels)['samples']]
    lookup = {sample['record_id']: sample for sample in samples}
    if len(lookup) != len(samples):
        raise ValueError('Duplicate record identifiers')
    selected = samples if ids is None else [lookup[rid] for rid in ids]
    graphs = [graph_from_sample(sample) for sample in selected]
    return graphs, paths

def load_protocol(protocol, set_id=1, data_root=None):
    if protocol == 'custom':
        if data_root is None or set_id != 1:
            raise ValueError('Custom protocol requires --data-root and set_id=1')
        paths = dataset_paths('custom',data_root)
        splits = read_json(paths['root']/'splits/custom.json')
        if set(splits) != {'train','validation','test'} or any(not ids for ids in splits.values()):
            raise ValueError('Custom splits require nonempty train, validation, and test lists')
        flat = [rid for ids in splits.values() for rid in ids]
        if len(flat) != len(set(flat)):
            raise ValueError('Custom train/validation/test record overlap')
        graphs,_ = load_group('custom',data_root=data_root)
        lookup = {g.node_feature_record_id:g for g in graphs}
        if set(flat) != set(lookup):
            raise ValueError('Custom splits must cover the package exactly')
        datasets = {name:[lookup[rid] for rid in ids] for name,ids in splits.items()}
        return datasets,paths,{'test_batch_size':16,'counts':{k:len(v) for k,v in datasets.items()}}
    settings = read_json(ROOT/'configs/protocols.json')[protocol]
    group = settings['group']
    paths = dataset_paths(group,data_root)
    filename = (f'hariboss_set{set_id}.json' if protocol == 'hariboss' else
                'train60.json' if protocol == 'test18' else 'hariboss_tr60.json')
    splits = read_json(paths['root']/'splits'/filename)
    flat = [rid for ids in splits.values() for rid in ids]
    if len(flat) != len(set(flat)):
        raise ValueError('Train/validation/test record overlap')
    graphs, _ = load_group(group,flat,data_root)
    lookup = {g.node_feature_record_id:g for g in graphs}
    datasets = {split:[lookup[rid] for rid in ids] for split,ids in splits.items()}
    expected = settings['counts'][f'set{set_id}'] if protocol=='hariboss' else settings['counts']
    if {k:len(v) for k,v in datasets.items()} != expected:
        raise ValueError('Protocol sample counts changed')
    return datasets, paths, settings
