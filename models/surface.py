"""Source-preserving numerical kernels; see THIRD_PARTY_NOTICES.md."""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn

SURFACE_CHEMICAL_FEATURE_DIM = 21
SURFACE_ATOM_NAME_VOCAB_SIZE = 35

def _batch_bounds(batch):
    ptr = getattr(batch, "ptr", None)
    if ptr is None:
        return [0, int(batch.x.size(0))]
    return ptr.detach().cpu().tolist()

def _batch_record_ids(batch, graph_count):
    values = getattr(batch, "node_feature_record_id", None)
    if isinstance(values, str):
        values = [values]
    elif isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list) or len(values) != graph_count:
        raise ValueError(
            "Extra branches require one data.node_feature_record_id per graph; "
            f"expected {graph_count}, got {values!r}"
        )
    return [str(value) for value in values]

def load_batch_surface(
    batch,
    cache_dir,
    device,
):
    """Load ligand-free, real-MSMS RNA surface point clouds."""
    bounds = _batch_bounds(batch)
    record_ids = _batch_record_ids(batch, len(bounds) - 1)
    packed = getattr(batch, 'rna_inputs', None)
    if packed is not None:
        packed = packed if isinstance(packed, list) else [packed]
        if len(packed) != len(record_ids):
            raise ValueError('Packed surface/RNA batch count mismatch')
        surfaces = []
        for inputs, rid, start, end in zip(packed, record_ids, bounds[:-1], bounds[1:]):
            surface = inputs.surface
            if surface['record_id'] != rid or surface['node_count'] != end-start:
                raise ValueError('Packed surface identity/node count mismatch')
            moved = dict(record_id=rid, node_count=end-start)
            moved.update({key: surface[key].to(device=device) for key in
                          ('xyz', 'nuv', 'chemistry', 'atom_name_index', 'owner')})
            surfaces.append(moved)
        return bounds, surfaces
    if cache_dir is None:
        raise ValueError('RNA has neither packed surface nor an explicit preparation cache')
    surfaces = []
    for record_id, start, end in zip(record_ids, bounds[:-1], bounds[1:]):
        path = Path(cache_dir) / f"{record_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing real MSMS surface cache: {path}")
        with np.load(path, allow_pickle=False) as payload:
            xyz = np.asarray(payload["surface_xyz"], dtype=np.float32)
            nuv = np.asarray(payload["surface_nuv"], dtype=np.float32)
            chemistry = np.asarray(payload["chemical_features"], dtype=np.float32)
            atom_name_index = np.asarray(payload["atom_name_index"], dtype=np.int64)
            owner = np.asarray(payload["owner_residue"], dtype=np.int64)
            cached_nodes = int(np.asarray(payload["node_count"]).item())
            assignment = (
                str(np.asarray(payload["surface_atom_assignment"]).item())
                if "surface_atom_assignment" in payload
                else "legacy_nearest_atom"
            )
        node_count = end - start
        point_count = xyz.shape[0]
        if cached_nodes != node_count:
            raise ValueError(
                f"Surface node mismatch for {record_id}: {cached_nodes} != {node_count}"
            )
        if xyz.ndim != 2 or xyz.shape[1] != 3 or point_count == 0:
            raise ValueError(f"Invalid surface coordinates for {record_id}: {xyz.shape}")
        if nuv.shape != (point_count, 3, 3):
            raise ValueError(f"Invalid surface frames for {record_id}: {nuv.shape}")
        if chemistry.shape != (point_count, SURFACE_CHEMICAL_FEATURE_DIM):
            raise ValueError(
                f"Invalid RNA surface chemistry for {record_id}: {chemistry.shape}"
            )
        if assignment != "msms_source_atom":
            raise ValueError(
                f"Surface cache {path} uses {assignment!r}; rebuild it with "
                "prepare_surface.py --overwrite to inherit GeoBind MSMS source atoms"
            )
        if atom_name_index.shape != (point_count,) or np.any(atom_name_index < 0) \
                or np.any(atom_name_index >= SURFACE_ATOM_NAME_VOCAB_SIZE):
            raise ValueError(f"Invalid nearest-atom indices for {record_id}")
        if owner.shape != (point_count,) or np.any(owner < 0) or np.any(owner >= node_count):
            raise ValueError(f"Invalid surface owner-residue mapping for {record_id}")
        if not all(np.isfinite(value).all() for value in (xyz, nuv, chemistry)):
            raise ValueError(f"Non-finite surface cache for {record_id}")
        surface = {
                "record_id": record_id,
                "node_count": node_count,
                "xyz": torch.from_numpy(xyz).to(device=device),
                "nuv": torch.from_numpy(nuv).to(device=device),
                "chemistry": torch.from_numpy(chemistry).to(device=device),
                "atom_name_index": torch.from_numpy(atom_name_index).to(device=device),
                "owner": torch.from_numpy(owner).to(device=device),
        }
        surfaces.append(surface)
    return bounds, surfaces

class _ContiguousBackward(torch.autograd.Function):
    """GeoBind/KeOps compatibility shim retained from the released code."""

    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient.contiguous()

class GeoBindQuasiGeodesicConv(nn.Module):
    """Source-compatible port of released GeoBind ``dMaSIFConv``.

    This is the exact ``cheap=False`` computation used by the released layer
    for one RNA surface at a time. GeoBind's optional KeOps ``ranges`` is
    ``None`` here because every surface is evaluated separately.
    """

    def __init__(self, in_channels, out_channels=64, radius=12.0):
        super().__init__()
        self.input_dim = int(in_channels)
        self.output_dim = int(out_channels)
        self.radius = float(radius)
        self.hidden_dim = self.output_dim
        self.cuts = 8
        self.head_dim = min(8, self.hidden_dim)
        if self.hidden_dim % self.head_dim:
            raise ValueError("GeoBind hidden dimension must be divisible by head dimension")
        self.head_count = self.hidden_dim // self.head_dim
        self.net_in = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.norm_in = nn.GroupNorm(4, self.hidden_dim)
        self.direction_filter = nn.Sequential(
            nn.Linear(3, self.cuts), nn.ReLU(), nn.Linear(self.cuts, self.hidden_dim)
        )
        self.net_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.output_dim, self.output_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.norm_out = nn.GroupNorm(4, self.output_dim)
        with torch.no_grad():
            nn.init.normal_(self.direction_filter[0].weight)
            nn.init.uniform_(self.direction_filter[0].bias)
            self.direction_filter[0].bias *= (
                0.8 * (self.direction_filter[0].weight ** 2).sum(-1).sqrt()
            )
            nn.init.uniform_(
                self.direction_filter[2].weight,
                a=-1.0 / math.sqrt(self.cuts),
                b=1.0 / math.sqrt(self.cuts),
            )
            nn.init.normal_(self.direction_filter[2].bias)
            self.direction_filter[2].bias *= (
                0.5 * (self.direction_filter[2].weight ** 2).sum(-1).sqrt()
            )

    def forward(self, points, local_frames, features):
        # Load KeOps only when a surface convolution is executed.
        from pykeops.torch import LazyTensor

        features = self.net_in(features)
        features = self.norm_in(features.transpose(0, 1)[None])[0].transpose(0, 1)
        points = points / (math.sqrt(2.0) * self.radius)
        x_i = LazyTensor(points[:, None, :])
        x_j = LazyTensor(points[None, :, :])
        frames_i = LazyTensor(local_frames.contiguous().view(-1, 1, 9))
        normal_i = frames_i[:3]
        normal_j = LazyTensor(local_frames[:, 0, :].contiguous().detach()[None, :, :])
        first_weight = LazyTensor(self.direction_filter[0].weight.view(1, 1, -1))
        first_bias = LazyTensor(self.direction_filter[0].bias.view(1, 1, -1))
        delta = x_j - x_i
        distance_squared = (delta ** 2).sum(-1) * ((2 - (normal_i | normal_j)) ** 2)
        window = (-distance_squared).exp()
        local_delta = frames_i.matvecmult(delta)
        hidden_direction = first_weight.matvecmult(local_delta) + first_bias
        hidden_direction = hidden_direction.relu()
        outputs = []
        for head in range(self.head_count):
            begin = head * self.head_dim
            finish = begin + self.head_dim
            second_weight = LazyTensor(
                self.direction_filter[2].weight[begin:finish].contiguous().view(1, 1, -1)
            )
            second_bias = LazyTensor(
                self.direction_filter[2].bias[begin:finish].contiguous().view(1, 1, -1)
            )
            directional = second_weight.matvecmult(hidden_direction) + second_bias
            directional = directional.relu()
            source = LazyTensor(features[:, begin:finish].contiguous()[None, :, :])
            interaction = window * directional * source
            outputs.append(_ContiguousBackward.apply(
                interaction.sum(dim=1, sum_scheme="auto")
            ))
        features = torch.cat(outputs, dim=1)
        features = self.net_out(features)
        return self.norm_out(features.transpose(0, 1)[None])[0].transpose(0, 1)

class SurfaceBranch(nn.Module):
    """Four 64D quasi-geodesic blocks; max+attention residue readout.

    Geometry convolution is ported from the released experiment implementation
    based on GeoBind/dMaSIF. No protein pretrained weights are used here.
    """
    def __init__(self, radius=12.0):
        super().__init__()
        self.atom_name_embedding = nn.Embedding(35, 8)
        self.layers = nn.ModuleList([GeoBindQuasiGeodesicConv(29, 64, radius)] +
                                    [GeoBindQuasiGeodesicConv(64, 64, radius) for _ in range(3)])
        self.post = nn.ModuleList([nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
                                   for _ in range(4)])
        self.residual = nn.ModuleList([nn.Linear(29, 64)] + [nn.Linear(64, 64) for _ in range(3)])
        self.point_pool_attention = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def _encode_one(self, surface):
        x = torch.cat([surface['chemistry'], self.atom_name_embedding(surface['atom_name_index'])], dim=-1)
        for layer, post, residual in zip(self.layers, self.post, self.residual):
            update = post(layer(surface['xyz'], surface['nuv'], x))
            x = residual(x) + update
        return x

    def forward(self, surfaces):
        outputs, masks = [], []
        for surface in surfaces:
            points = self._encode_one(surface)
            pooled, mask = self._pool_points_max_attention(points, surface['owner'], surface['node_count'])
            outputs.append(pooled)
            masks.append(mask)
        if not outputs:
            raise ValueError('Empty RNA surface batch')
        return torch.cat(outputs, dim=0), torch.cat(masks, dim=0)

    def _pool_points_attention_only(self, point_states, owner, node_count):
        """Return only the learned 64D attention pool for each residue."""
        logits = self.point_pool_attention(point_states).squeeze(-1)
        maximum_logits = point_states.new_full((node_count,), -float("inf"))
        maximum_logits.scatter_reduce_(
            0, owner, logits, reduce="amax", include_self=True
        )
        weights = torch.exp(logits - maximum_logits[owner])
        normalizer = point_states.new_zeros((node_count,))
        normalizer.index_add_(0, owner, weights)
        weights = weights / normalizer[owner].clamp_min(1e-12)
        attended = point_states.new_zeros((node_count, point_states.size(1)))
        attended.index_add_(0, owner, weights[:, None] * point_states)
        counts = point_states.new_zeros((node_count, 1))
        counts.index_add_(
            0, owner, point_states.new_ones((point_states.size(0), 1))
        )
        present = counts.squeeze(-1) > 0
        attended = torch.where(
            present[:, None], attended, torch.zeros_like(attended)
        )
        return attended, present[:, None]

    def _pool_points_max_attention(self, point_states, owner, node_count):
        """Concatenate max and learned attention pools for each residue."""
        maxima = point_states.new_full(
            (node_count, point_states.size(1)), -float("inf")
        )
        maxima.scatter_reduce_(
            0, owner[:, None].expand(-1, point_states.size(1)), point_states,
            reduce="amax", include_self=True,
        )
        counts = point_states.new_zeros((node_count, 1))
        counts.index_add_(0, owner, point_states.new_ones((point_states.size(0), 1)))
        present = counts > 0
        maxima = torch.where(present, maxima, torch.zeros_like(maxima))
        attended, _ = self._pool_points_attention_only(
            point_states, owner, node_count
        )
        return torch.cat([maxima, attended], dim=-1), present
