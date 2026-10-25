# GeoBind molecular-surface source audit

Scope: upstream source provenance and its adaptation in the current released
RiboSurfBind implementation. Historical source hashes below identify the
audited upstream files; they are not hashes of the adapted model files.
Installation and runtime validation are documented in `ENVIRONMENTS.md`.

This records source/numerical provenance, **not redistribution clearance**.
GeoBind's MIT notice is retained, but the dMaSIF implementation lineage has
a separate CC BY-NC-ND permission question. MSMS has an archive transfer
restriction. See `REDISTRIBUTION_AUDIT.md` before public distribution;
source hashes and passing equivalence tests do not resolve these rights.

## Primary method and released implementation

- Paper: **GeoBind: segmentation of nucleic acid binding interface on protein
  surface with geometric deep learning**, *Nucleic Acids Research* (2023).
- Author repository: `https://github.com/zpliulab/GeoBind`
- Audited commit: `f8e336b8239fecf655d5a306ae56f887b1cf146e`
- The released defaults are four convolution blocks, 64 channels and radius
  12 Å.  The convolution is not PointNet/GCN: it applies a normal-aware
  Gaussian pseudo-geodesic window and a learned directional filter to
  coordinates expressed in each surface point's `[normal,u,v]` frame.

Selected source SHA-256 values at that commit:

| file | SHA-256 |
|---|---|
| `GeoBindProcessor.py` | `c53856d94ed325c31eea8f0f1b0474fb3a08cee7b0d68d4b409f7a8842745331` |
| `compute_surface/protonate.py` | `091a8af6183d25e0cad023c41594038c51029f978e4ba79ad4190d8c19cb1d43` |
| `compute_surface/extract_xyzrn.py` | `99e0cd7b7c53bec36eb23c65f670a19cdb9f9647949fe22ad8e1c5ff06bbda09` |
| `compute_surface/apply_msms.py` | `bb0c7a53180562b874dc36fdcde34372b5e132b81d5d2f96151de614c4820f2c` |
| `compute_surface/fixmesh.py` | `a874394685fdae208e0c1e972d04f7d70e78f161c3419ca7ffa85b9f0aa3baa6` |
| `compute_surface/compute_normal.py` | `eaa7168736bf948c57eb5b86430feb04c1b2becdcc9b851e7da1f2ddeeca70cc` |
| `models/geometry_processing.py` | `65e593574fa233ad80375bc408ae672a85951b515ae126f50047ded28a098e5e` |
| `models/Segmentation.py` | `3b544c06dc7a7febb0d8e250537bf8b4c70e6c49df0726e396905a487c3e9658` |

## Exact geometry software

### MSMS

- Official Scripps MSMS 2.6.1 Linux archive was used.
- Downloaded archive SHA-256:
  `5f0ca50360b5938e74c538e0399d582abc4a40ef4cf410e66f31a1f91e6e3e1f`.
- Static executable SHA-256:
  `739b88d8195b89ee5a9c00ef7b986f6bb9d32f7d3b200f457781c98b95958088`.
- A real included structure produced `.vert`, `.face`, and `.area` files and
  ended with `MSMS terminated normally`.

The GeoBind invocation is preserved exactly: `density=3`, `hdensity=3`,
`probe=1.5`.  Official element radii are also preserved.

### Reduce

- Exact version: `reduce.3.23.130521`, as required by GeoBind.
- The original publisher endpoint is no longer serving the old file.  The
  binary was recovered from the immutable public `edraizen/masif:latest`
  container layer whose recorded build command downloaded that exact Duke
  release.
- Container blob digest:
  `sha256:57aff346306dcc320ddcca807564d7e211e538b0ca082c54e96473a53c4f3c55`.
- Recovered publisher gzip SHA-256:
  `3cad24fe5f10534f8f60c1e9e00f420cbc3a291dca4f0881a4960bd278ab37ea`;
  this equals the independently published Reduce-3.23 PKGBUILD checksum.
- Executable SHA-256:
  `6d066f88bff740627d7c1d2fb0200d326978fe70a0c041a2528c8853e682b1ce`.
- Runtime self-report: `reduce: version 3.23 05/21/2013`.

The released GeoBind calls are preserved: `-Trim`, then `-HIS`.  On a real
Train60 RNA, Reduce added 326 hydrogens and MSMS generated 13,233 raw vertices
and 26,462 faces without any ligand coordinates.

### PyMesh

- Official PyMesh release wheel: `pymesh2-0.3-cp37-cp37m-linux_x86_64.whl`.
- Wheel SHA-256:
  `da1f2df6c6856f58cf537cb877aae420108d4ac34d2a78bb47769cdcf2a85980`.
- It is kept in a Python-3.7-only preprocessing environment and communicates
  with RiboSurfBind's preparation process through NPZ arrays. The released repair operation order and
  constants are copied verbatim in `pymesh_fix_official.py`, including the
  1.2 Å target edge length.

## RNA task adaptation (explicitly not claimed as original GeoBind)

GeoBind predicts nucleic-acid interfaces on **protein** surfaces.  Here its
surface geometry core is transferred to the RNA receptor surface for
ligand-free RNA small-molecule site prediction:

1. only explicitly aligned RNA residues are written as the receptor; ligand,
   water, ions and unrelated HETATM records never enter Reduce/MSMS;
2. every repaired surface point inherits the source atom ID of its nearest
   raw MSMS surface vertex, exactly following the released GeoBind
   `fix_mesh` provenance transfer;
3. protein HMM/chemistry features are replaced with deterministic RNA point
   chemistry: element, actual nearest-atom identity, owner base,
   base/sugar/phosphate region, the project's FR3D-derived donor/acceptor
   definitions, nearest-atom distance and PyMesh curvature;
4. the point input concatenates 21 cached chemical/geometric descriptors and
   a learned 8D atom-name embedding. Four 64-channel quasi-geodesic convolution
   blocks encode each RNA surface separately, using the surface-point local
   frames and residual updates implemented in `models/surface.py`;
5. points are assigned to residues by inherited source-atom ownership. For
   each residue, 64D channel-wise maximum pooling and a 64D learned attention
   pool are concatenated into a 128D vector. There is no mean-pooling readout
   or separate 256D surface projection in the current model;
6. the 128D residue-relation state and 128D surface state receive separate
   LayerNorm operations and are concatenated into a 256D joint representation.
   A 256-to-128-to-1 MLP with GELU and dropout predicts each residue's logit;
   sigmoid gives its binding probability. This late fusion is concatenation,
   not a learned residual gate (`models/fusion.py`);
7. the relation gate belongs to the other branch: it weights semantic-graph
   and spatial-graph messages within residue propagation. It must not be
   confused with the surface readout or the final two-branch fusion.

The surface branch is trained for the RNA task; no protein-surface pretrained
weights are loaded. Auxiliary prediction heads and historical compatibility
layers are retained as described in `REPRODUCIBILITY.md`.

No binding label or ligand coordinate is used in surface preprocessing or
forward inference.
