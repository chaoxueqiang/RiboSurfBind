"""Published two-branch model; no search or unused candidate architectures.

The legacy-named module attributes deliberately preserve checkpoint keys and
parameter initialization/optimizer order. The four compatibility layers are
never used in forward; see docs/REPRODUCIBILITY.md.
"""
from torch import nn
import torch
from .semantic_spatial import SemanticSpatialBranch, run_batched_attention_branch
from .surface import SurfaceBranch, load_batch_surface
from .fusion import AttentionSurfaceLightConcatFusion
from data_process.attention import load_batch_attention


class RiboSurfBind(nn.Module):
    def __init__(self, config, attention_cache_dir=None, surface_cache_dir=None):
        super().__init__()
        self.attention_cache_dir = str(attention_cache_dir) if attention_cache_dir is not None else None
        self.surface_cache_dir = str(surface_cache_dir) if surface_cache_dir is not None else None
        # Checkpoint/RNG compatibility only. Do not remove without migrating
        # published checkpoints and the initialization trajectory.
        self.lin4 = nn.Linear(256, 256)
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.independent_late_fusion = AttentionSurfaceLightConcatFusion(
            config['hidden_dim'], surface_input_dim=128,
            dropout=config['dropout'],
            hidden_dim=config['fusion_hidden_dim'],
            ablate_branch=config.get('ablate_branch', 'none'),
        )
        self.attention_branch = SemanticSpatialBranch(config)
        self.extra_branch_module = SurfaceBranch(radius=config['surface_radius'])
        self.last_branch_aux_logits = None

    def forward(self, data, return_classifier_logits=False):
        embedding = data.ernie_rna_embedding
        if tuple(embedding.shape) != (int(data.num_nodes), 768):
            raise ValueError('Expected one 768D ERNIE embedding per nucleotide')
        matrices, ptr = load_batch_attention(data, self.attention_cache_dir,
                                             embedding.device)
        attention_state = run_batched_attention_branch(
            self.attention_branch, embedding, matrices, ptr,
            positions=data.pos if self.attention_branch.graph_mode == 'topk_spatial_knn' else None,
        )
        _, surfaces = load_batch_surface(data, self.surface_cache_dir, embedding.device)
        surface_state, surface_mask = self.extra_branch_module(surfaces)
        surface_mask = surface_mask.to(dtype=embedding.dtype)
        logits, att_aux, surface_aux = self.independent_late_fusion(
            attention_state, surface_state, surface_mask)
        self.last_branch_aux_logits = {
            'attention': att_aux, 'surface': surface_aux, 'surface_mask': surface_mask}
        probability = torch.sigmoid(logits)
        if return_classifier_logits:
            return probability, logits
        return probability
