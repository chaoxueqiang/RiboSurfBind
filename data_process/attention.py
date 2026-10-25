import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import torch

CACHE_SCHEMA_VERSION = "ernie_attention_postsoftmax_v2"
ATTENTION_SEMANTICS = "per_head_postsoftmax_rna_renormalized_mean"
RAW_SOURCE_SEMANTICS = (
    "ERNIE pre-softmax attention bias transformed to probability"
)
SPECIAL_TOKEN_POLICY = (
    "softmax_full_then_drop_bos_eos_then_rna_renormalize"
)
MIN_ATTENTION_VALUE_TOL = -1e-6
MAX_ROW_SUM_ERROR_TOL = 1e-4
_ATTENTION_RECORD_CACHE: Dict[Tuple[str, int, int], Dict] = {}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_sequence_key(sequence: str) -> str:
    return f"seq_{sha256_text(sequence)}"


def summarize_attention_matrix(attention: torch.Tensor) -> Dict[str, float]:
    row_sums = attention.sum(dim=-1)
    row_sum_error = torch.abs(row_sums - 1.0)
    return {
        "global_min": float(attention.min().item()),
        "global_max": float(attention.max().item()),
        "row_sum_min": float(row_sums.min().item()),
        "row_sum_max": float(row_sums.max().item()),
        "max_row_sum_error": float(row_sum_error.max().item()),
        "negative_count": int((attention < 0).sum().item()),
    }


def validate_attention_matrix(
    attention: torch.Tensor,
    node_count: int,
    sequence_key: str,
) -> Dict[str, float]:
    if not torch.is_tensor(attention):
        raise TypeError(f"Attention cache for {sequence_key} is not a tensor.")
    if not attention.is_floating_point():
        raise TypeError(
            f"Attention cache for {sequence_key} must be floating point."
        )
    if tuple(attention.shape) != (node_count, node_count):
        raise ValueError(
            f"Attention shape mismatch for {sequence_key}: "
            f"{tuple(attention.shape)} != {(node_count, node_count)}"
        )
    if not torch.isfinite(attention).all():
        raise ValueError(f"Attention cache for {sequence_key} contains NaN/Inf.")
    stats = summarize_attention_matrix(attention)
    if stats["global_min"] < MIN_ATTENTION_VALUE_TOL:
        raise ValueError(
            f"Attention cache for {sequence_key} contains negative values: "
            f"min={stats['global_min']}"
        )
    if stats["max_row_sum_error"] > MAX_ROW_SUM_ERROR_TOL:
        raise ValueError(
            f"Attention cache rows for {sequence_key} are not normalized: "
            f"max_error={stats['max_row_sum_error']}"
        )
    return stats


def validate_attention_record(
    record: Dict,
    sequence_key: str,
    sequence: str,
    node_count: int,
) -> Dict[str, float]:
    required_fields = {
        "cache_schema_version",
        "attention_semantics",
        "raw_source_semantics",
        "special_token_policy",
        "sequence_key",
        "sequence",
        "sequence_sha256",
        "num_nodes",
        "attention",
        "source_model",
        "source_checkpoint",
        "ernie_repo_commit",
        "created_at",
    }
    missing = sorted(required_fields - set(record))
    if missing:
        raise ValueError(
            f"Attention cache for {sequence_key} is missing fields: {missing}"
        )
    if record["cache_schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Cache schema mismatch for {sequence_key}: "
            f"{record['cache_schema_version']} != {CACHE_SCHEMA_VERSION}"
        )
    if record["attention_semantics"] != ATTENTION_SEMANTICS:
        raise ValueError(f"Attention semantics mismatch for {sequence_key}.")
    if record["raw_source_semantics"] != RAW_SOURCE_SEMANTICS:
        raise ValueError(f"Raw source semantics mismatch for {sequence_key}.")
    if record["special_token_policy"] != SPECIAL_TOKEN_POLICY:
        raise ValueError(f"Special-token policy mismatch for {sequence_key}.")
    if record["sequence_key"] != sequence_key:
        raise ValueError(f"Sequence-key mismatch for {sequence_key}.")
    if record["sequence"] != sequence:
        raise ValueError(f"Sequence mismatch for {sequence_key}.")
    if record["sequence_sha256"] != sha256_text(sequence):
        raise ValueError(f"Sequence hash mismatch for {sequence_key}.")
    if int(record["num_nodes"]) != node_count:
        raise ValueError(f"Node-count mismatch for {sequence_key}.")
    return validate_attention_matrix(
        record["attention"],
        node_count=node_count,
        sequence_key=sequence_key,
    )


def load_attention_record(
    cache_dir: Path,
    sequence: str,
    node_count: int,
) -> Dict:
    sequence_key = make_sequence_key(sequence)
    path = Path(cache_dir) / f"{sequence_key}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ERNIE-RNA attention cache for {sequence_key}: {path}"
        )
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    record = _ATTENTION_RECORD_CACHE.get(cache_key)
    if record is None:
        record = torch.load(path, map_location="cpu", weights_only=False)
        validate_attention_record(
            record,
            sequence_key=sequence_key,
            sequence=sequence,
            node_count=node_count,
        )
        _ATTENTION_RECORD_CACHE[cache_key] = record
    return record


def _batch_ptr(batch) -> torch.Tensor:
    ptr = getattr(batch, "ptr", None)
    if ptr is not None:
        return ptr
    return torch.tensor(
        [0, int(batch.x.shape[0])],
        dtype=torch.long,
        device=batch.x.device,
    )


def load_batch_attention(
    batch,
    cache_dir: Path,
    device: torch.device,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    ptr = _batch_ptr(batch)
    ptr_values = ptr.detach().cpu().tolist()
    packed = getattr(batch, 'rna_inputs', None)
    if packed is not None:
        packed = packed if isinstance(packed, list) else [packed]
        if len(packed) != len(ptr_values)-1:
            raise ValueError('Packed attention/RNA batch count mismatch')
        matrices = []
        for inputs, start, end in zip(packed, ptr_values[:-1], ptr_values[1:]):
            if tuple(inputs.attention.shape) != (end-start, end-start):
                raise ValueError('Packed attention/node count mismatch')
            matrices.append(inputs.attention.to(device=device, dtype=torch.float32))
        return matrices, ptr
    if cache_dir is None:
        raise ValueError('RNA has neither packed attention nor an explicit preparation cache')
    explicit_sequences = getattr(batch, "rna_sequence", None)
    if explicit_sequences is not None:
        if isinstance(explicit_sequences, str):
            explicit_sequences = [explicit_sequences]
        else:
            explicit_sequences = list(explicit_sequences)
        if len(explicit_sequences) != len(ptr_values) - 1:
            raise ValueError(
                "Explicit RNA sequence count does not match batch graph count: "
                f"{len(explicit_sequences)} != {len(ptr_values) - 1}"
            )
    matrices: List[torch.Tensor] = []
    for sample_index, (start, end) in enumerate(
        zip(ptr_values[:-1], ptr_values[1:])
    ):
        if explicit_sequences is None:
            raise ValueError("Canonical RNA records must carry rna_sequence")
        else:
            sequence = str(explicit_sequences[sample_index])
            if len(sequence) != end - start:
                raise ValueError(
                    "Explicit RNA sequence/node-count mismatch for graph "
                    f"{sample_index}: {len(sequence)} != {end - start}"
                )
        record = load_attention_record(
            Path(cache_dir),
            sequence=sequence,
            node_count=end - start,
        )
        matrices.append(
            record["attention"].to(device=device, dtype=torch.float32)
        )
    return matrices, ptr
