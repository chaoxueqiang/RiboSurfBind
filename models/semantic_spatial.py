"""Source-preserving numerical kernels; see THIRD_PARTY_NOTICES.md."""
import math
from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F

def _normalize_top_k_per_layer(
    top_k: Union[int, Sequence[int]],
    num_layers: int,
) -> Tuple[int, ...]:
    if isinstance(top_k, bool):
        raise ValueError("top_k must contain positive integers")
    if isinstance(top_k, int):
        values = (int(top_k),) * int(num_layers)
    else:
        values = tuple(top_k)
        if len(values) != int(num_layers):
            raise ValueError(
                "per-layer top_k must have one value per attention layer: "
                f"got {len(values)} values for {num_layers} layers"
            )
    if any(
        isinstance(value, bool)
        or int(value) != value
        or int(value) < 1
        for value in values
    ):
        raise ValueError("top_k must contain positive integers")
    return tuple(int(value) for value in values)

def topk_attention_adjacency(
    attention: torch.Tensor,
    top_k: int,
    include_self: bool = False,
) -> torch.Tensor:
    """Keep the strongest incoming attention neighbours in every row.

    ERNIE attention row ``i`` describes how residue ``i`` attends to source
    residues ``j``.  Therefore top-k selection is performed independently on
    every row before ``adj @ x`` message aggregation.  Selected weights are
    renormalized to sum to one so changing k does not silently change the
    overall message scale.
    """

    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError(
            "attention must be a square [num_residues, num_residues] tensor"
        )
    if isinstance(top_k, bool) or int(top_k) != top_k or int(top_k) < 1:
        raise ValueError("top_k must be a positive integer")
    if not bool(torch.isfinite(attention).all().item()):
        raise ValueError("attention contains NaN or Inf")

    node_count = int(attention.shape[0])
    if node_count == 0:
        return attention
    if node_count == 1:
        return torch.ones_like(attention)

    available = node_count if include_self else node_count - 1
    effective_k = min(int(top_k), available)
    scores = attention
    if not include_self:
        scores = attention.clone()
        scores.fill_diagonal_(-torch.inf)

    values, indices = torch.topk(
        scores,
        k=effective_k,
        dim=1,
        largest=True,
        sorted=False,
    )
    sparse = torch.zeros_like(attention).scatter(1, indices, values)
    row_sum = sparse.sum(dim=1, keepdim=True)
    fallback = torch.zeros_like(attention).scatter(
        1,
        indices,
        torch.ones_like(values) / float(effective_k),
    )
    tiny = torch.finfo(attention.dtype).tiny
    return torch.where(
        row_sum > tiny,
        sparse / row_sum.clamp_min(tiny),
        fallback,
    )

def spatial_knn_adjacency(
    positions: torch.Tensor,
    k: int,
    include_self: bool = False,
) -> torch.Tensor:
    """Build a row-stochastic 3D residue kNN adjacency.

    Rows are receiving residues and columns are source neighbours, matching
    the orientation used by ``adj @ x`` in the attention branch.  Spatial
    neighbours receive uniform mass; distances are used only for selection.
    """

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [num_residues, 3]")
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError("spatial k must be a positive integer")
    if not bool(torch.isfinite(positions).all().item()):
        raise ValueError("positions contain NaN or Inf")
    node_count = int(positions.shape[0])
    if node_count == 0:
        return positions.new_zeros((0, 0))
    if node_count == 1:
        return positions.new_ones((1, 1))

    available = node_count if include_self else node_count - 1
    effective_k = min(int(k), available)
    distances = torch.cdist(positions, positions)
    if not include_self:
        distances.fill_diagonal_(torch.inf)
    indices = torch.topk(
        distances,
        k=effective_k,
        dim=1,
        largest=False,
        sorted=False,
    ).indices
    values = positions.new_full(
        indices.shape, 1.0 / float(effective_k)
    )
    return positions.new_zeros((node_count, node_count)).scatter(
        1, indices, values
    )

class GCNIIConvolution(nn.Module):
    """Published equal-width GCNII update with layer-input residual."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.weight = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(
        self,
        input: torch.Tensor,
        adj: Optional[torch.Tensor],
        h0: torch.Tensor,
        lamda: float,
        alpha: float,
        layer_index: int,
        neighbor_message: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        theta_l = min(1.0, math.log(lamda / layer_index + 1.0))
        if neighbor_message is None:
            hi = torch.matmul(adj, input)
        else:
            if neighbor_message.shape != input.shape:
                raise ValueError(
                    "neighbor_message must have the same shape as input: "
                    f"{tuple(neighbor_message.shape)} != {tuple(input.shape)}"
                )
            hi = neighbor_message
        support = (1 - alpha) * hi + alpha * h0
        transformed = torch.mm(support, self.weight)
        output = theta_l * transformed + (1 - theta_l) * support
        output = output + input
        return output

class DualRelationNodeGate(nn.Module):
    """Residue-specific soft gate over semantic and spatial messages."""

    def __init__(self, hidden_dim: int, gate_hidden_dim: int):
        super().__init__()
        if int(gate_hidden_dim) < 1:
            raise ValueError("gate_hidden_dim must be positive")
        self.state_norm = nn.LayerNorm(int(hidden_dim))
        self.semantic_norm = nn.LayerNorm(int(hidden_dim))
        self.spatial_norm = nn.LayerNorm(int(hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(4 * int(hidden_dim), int(gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), 2),
        )
        # Exact 50/50 mixture at initialization.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        state: torch.Tensor,
        semantic_message: torch.Tensor,
        spatial_message: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state_norm = self.state_norm(state)
        semantic_norm = self.semantic_norm(semantic_message)
        spatial_norm = self.spatial_norm(spatial_message)
        gate_input = torch.cat(
            [
                state_norm,
                semantic_norm,
                spatial_norm,
                torch.abs(semantic_norm - spatial_norm),
            ],
            dim=-1,
        )
        weights = torch.softmax(self.mlp(gate_input), dim=-1)
        message = (
            weights[:, 0:1] * semantic_message
            + weights[:, 1:2] * spatial_message
        )
        return message, weights

class SemanticSpatialBranch(nn.Module):
    """GCNII propagation with learned semantic/spatial message gating.

    Graph construction and GCNII arithmetic are source-preserving ports of
    the experiment implementation (ATMGBs-inspired propagation).
    """
    def __init__(self, config):
        super().__init__()
        self.num_layers = int(config['attention_depth'])
        if not 1 <= self.num_layers <= 6:
            raise ValueError('attention_depth must be in [1, 6]')
        self.graph_mode = config.get('graph_mode', 'topk_spatial_knn')
        if self.graph_mode not in ('topk', 'topk_spatial_knn'):
            raise ValueError('Unsupported graph mode')
        self.top_k_per_layer = _normalize_top_k_per_layer(config['semantic_topk'], self.num_layers)
        self.spatial_knn_k = int(config['spatial_knn'])
        self.spatial_knn_weight = float(config.get('spatial_weight', .5))
        self.output_dim = int(config['hidden_dim'])
        self.input_proj = nn.Linear(768, self.output_dim)
        self.convs = nn.ModuleList([
            GCNIIConvolution(self.output_dim)
            for _ in range(self.num_layers)])
        self.att_dropout = float(config['dropout'])
        self.lamda = float(config['gcnii_lambda'])
        self.alpha = float(config['gcnii_alpha'])
        self.dual_relation_gates = None
        if config.get('relation_gate', True):
            if self.graph_mode != 'topk_spatial_knn':
                raise ValueError('Relation gate requires both relation graphs')
            rng = torch.random.get_rng_state()
            self.dual_relation_gates = nn.ModuleList([
                DualRelationNodeGate(self.output_dim, int(config['gate_hidden_dim']))
                for _ in range(self.num_layers)])
            torch.random.set_rng_state(rng)

    def forward(self, x, attention, positions=None):
        x = F.dropout(x, self.att_dropout, training=self.training)
        h0 = F.relu(self.input_proj(x))
        inner = h0
        for index, conv in enumerate(self.convs, start=1):
            semantic = topk_attention_adjacency(
                attention, self.top_k_per_layer[index - 1], include_self=False)
            spatial = None
            adjacency = semantic
            if self.graph_mode == 'topk_spatial_knn':
                if positions is None:
                    raise ValueError('RNA coordinates are required')
                spatial = spatial_knn_adjacency(positions, self.spatial_knn_k, include_self=False)
                # With a learned gate, GCNII consumes the gated message and
                # does not use a fixed-mixture adjacency at all. The original
                # fixed-mixture arithmetic remains for no-gate ablations.
                adjacency = None
                if self.dual_relation_gates is None:
                    adjacency = (1.0 - self.spatial_knn_weight) * semantic + self.spatial_knn_weight * spatial
            previous = F.dropout(inner, self.att_dropout, training=self.training)
            message = None
            if self.dual_relation_gates is not None:
                message, _ = self.dual_relation_gates[index - 1](
                    previous, torch.matmul(semantic, previous), torch.matmul(spatial, previous))
            inner = F.relu(conv(previous, adjacency, h0, self.lamda,
                                self.alpha, index, neighbor_message=message))
        return inner

def run_batched_attention_branch(
    branch: SemanticSpatialBranch,
    x: torch.Tensor,
    attention_matrices: Sequence[torch.Tensor],
    ptr: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run per-RNA propagation, concatenating only the final residue states."""

    output_parts: List[torch.Tensor] = []
    ptr_values = ptr.detach().cpu().tolist()
    if len(attention_matrices) != len(ptr_values) - 1:
        raise ValueError(
            "Attention matrix count must match the number of RNAs in the batch."
        )
    for sample_idx, (start, end) in enumerate(
        zip(ptr_values[:-1], ptr_values[1:])
    ):
        sample_x = x[start:end]
        sample_attention = attention_matrices[sample_idx]
        sample_positions = None if positions is None else positions[start:end]
        expected_shape = (end - start, end - start)
        if tuple(sample_attention.shape) != expected_shape:
            raise ValueError(
                f"Attention shape mismatch for sample {sample_idx}: "
                f"{tuple(sample_attention.shape)} != {expected_shape}"
            )
        final_state = branch(sample_x, sample_attention, sample_positions)
        if tuple(final_state.shape) != (end - start, branch.output_dim):
            raise RuntimeError("Attention branch returned invalid final residue states")
        output_parts.append(final_state)
    return torch.cat(output_parts, dim=0)
