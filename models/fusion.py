"""Source-preserving numerical kernels; see THIRD_PARTY_NOTICES.md."""
import torch
from torch import nn
import torch.nn.functional as F

class AttentionSurfaceLightConcatFusion(nn.Module):
    """Separate branch normalization, concatenation and a 128D prediction MLP.

    Auxiliary heads preserve the published training objective and optimizer
    state. Single-branch ablations mask states after their normalization.
    """

    def __init__(
        self, attention_input_dim, surface_input_dim=128, dropout=0.3,
        hidden_dim=128,
        ablate_branch="none",
    ):
        super().__init__()
        attention_input_dim = int(attention_input_dim)
        surface_input_dim = int(surface_input_dim)
        hidden_dim = int(hidden_dim)
        if hidden_dim < 1 or surface_input_dim < 1:
            raise ValueError("fusion dimensions must be positive")
        self.attention_norm = nn.LayerNorm(attention_input_dim)
        self.surface_norm = nn.LayerNorm(surface_input_dim)
        self.hidden = nn.Linear(attention_input_dim + surface_input_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(hidden_dim, 1)
        if ablate_branch not in {"none", "attention", "surface"}:
            raise ValueError("ablate_branch must be none, attention, or surface")
        self.ablate_branch = str(ablate_branch)
        self.attention_head = nn.Linear(attention_input_dim, 1)
        self.surface_head = nn.Linear(surface_input_dim, 1)

    def forward(self, attention_state, surface_state, surface_mask):
        if surface_state is None or surface_mask is None:
            raise ValueError("Light ATT+Surface concat requires Surface state/mask")
        attention = self.attention_norm(attention_state)
        surface = self.surface_norm(surface_state)
        mask = surface_mask.to(dtype=surface.dtype)
        surface = surface * mask
        # Controlled branch ablation: preserve the exact fusion-head shape and
        # parameter count while removing one branch's predictive information.
        if self.ablate_branch == "attention":
            attention = torch.zeros_like(attention)
        elif self.ablate_branch == "surface":
            surface = torch.zeros_like(surface)
        hidden = self.dropout(
            F.gelu(self.hidden(torch.cat([attention, surface], dim=-1)))
        )
        logits = self.output(hidden).squeeze(-1)
        attention_logits = self.attention_head(attention).squeeze(-1)
        surface_logits = self.surface_head(surface).squeeze(-1)
        return logits, attention_logits, surface_logits
