"""Portable, tensor-only RNA packages and a non-collating runtime carrier.

Disk files contain only dictionaries/lists/scalars/tensors, not custom classes.
RNAInputs is created after loading: PyG keeps one carrier per RNA rather than
concatenating variable-size attention matrices or offsetting point ownership.
The carrier deliberately has no ``to`` method; model input loaders transfer its
CPU tensors at the same point in forward as in the original implementation.
"""
from functools import lru_cache
from pathlib import Path
import torch
from data_process.attention import validate_attention_matrix, ATTENTION_SEMANTICS

SCHEMA = 'ribosurfbind_rna_package_v1'


class RNAInputs:
    def __init__(self, attention, surface):
        self.attention = attention
        self.surface = surface


def validate_sample(sample, require_labels=True):
    rid = sample['record_id']
    n = len(sample['sequence'])
    if not n or len(sample['residue_ids']) != n:
        raise ValueError('Invalid sequence/residue identities: ' + rid)
    for key, shape in [('pos', (n, 3)), ('embedding', (n, 768))]:
        value = sample[key]
        if not torch.is_tensor(value) or tuple(value.shape) != shape \
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid ' + key + ': ' + rid)
    attention = sample['attention']
    if attention.dtype != torch.float32:
        raise ValueError('Attention must remain float32: ' + rid)
    validate_attention_matrix(attention, n, rid)
    if sample['provenance']['attention']['attention_semantics'] != ATTENTION_SEMANTICS:
        raise ValueError('Unsupported attention semantics: ' + rid)
    labels = sample.get('y')
    if labels is None and require_labels:
        raise ValueError('Missing benchmark labels: ' + rid)
    if labels is not None and (labels.shape != (n,) or labels.dtype != torch.float32
                               or not bool(((labels == 0) | (labels == 1)).all())):
        raise ValueError('Invalid binding labels: ' + rid)
    surface = sample['surface']
    m = surface['xyz'].shape[0]
    if m < 1 or surface['node_count'] != n or surface['record_id'] != rid:
        raise ValueError('Invalid surface identity: ' + rid)
    if surface['assignment'] != 'msms_source_atom':
        raise ValueError('Unsupported point ownership: ' + rid)
    for key, shape in [('xyz', (m, 3)), ('nuv', (m, 3, 3)), ('chemistry', (m, 21))]:
        value = surface[key]
        if tuple(value.shape) != shape or value.dtype != torch.float32 \
                or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid surface ' + key + ': ' + rid)
    for key, upper in [('atom_name_index', 35), ('owner', n)]:
        value = surface[key]
        if value.shape != (m,) or value.dtype != torch.int64 \
                or bool(((value < 0) | (value >= upper)).any()):
            raise ValueError('Invalid surface ' + key + ': ' + rid)


@lru_cache(maxsize=8)
def _read_package(path, modified_ns, file_size):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != SCHEMA:
        raise ValueError('Unsupported RNA package: ' + path)
    samples = payload['samples']
    if not samples or len({s['record_id'] for s in samples}) != len(samples):
        raise ValueError('Empty package or duplicate record IDs: ' + path)
    for sample in samples:
        validate_sample(sample, require_labels=False)
    if payload['counts']['records'] != len(samples) \
            or payload['counts']['nucleotides'] != sum(len(s['sequence']) for s in samples):
        raise ValueError('Package counts differ from contents: ' + path)
    return payload


def read_package(path, require_labels=True):
    path = Path(path).resolve()
    stat = path.stat()
    payload = _read_package(str(path), stat.st_mtime_ns, stat.st_size)
    if require_labels and any(sample.get('y') is None for sample in payload['samples']):
        raise ValueError('Unlabelled samples in benchmark package: ' + str(path))
    return payload
