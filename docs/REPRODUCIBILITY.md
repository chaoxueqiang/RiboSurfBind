# Reproduction contract

## Git asset storage

Current benchmark data and published checkpoints are tracked in this repository;
the old migration backup and newly generated runs remain ignored. Only the two
large HARIBOSS data packages use Git LFS, as listed in `.gitattributes`. MSMS and
the ERNIE pretrained checkpoint are local-only official-download assets; their
versions/hashes are in `external_assets.json` and instructions in
[EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md). Full local input/weight files,
their hashes, model arithmetic and random-state handling are unchanged. Preserve
both `.git/objects` and `.git/lfs/objects` when backing up an unpublished local
repository. Those historical experiment objects must not be copied into a
publication repository: use the fresh snapshot exporter to omit external
assets and old Git history. A publication clone needs Git LFS and both real
HARIBOSS objects before asset checks; a bare pointer is not a valid input package.

The present server uses a verified official Git LFS binary in `.git/local-tools/`
with repository-local filters, hooks and a `git lfs` alias. No system/global Git
configuration or training environment is changed. This local setup is not cloned;
on another machine install Git LFS normally and run `git lfs install --local`.

The preprocessing/asset cleanup removes discarded nucleotide interaction-frame
and legacy centroid calculations, unused imports, and the duplicate sequence-key
implementation. It does not edit model files, training/evaluation arithmetic,
random-state handling, auxiliary heads or the four compatibility layers.
Surface-point local frames remain active and unchanged. Paired verification is
recorded in `preprocessing_asset_cleanup_20261024.json`.

`asset_aliases.json` records eight lossless, relative internal links: two default
sensitivity weight copies and six byte-identical training-log copies. Original
paths, configuration files, checkpoint source hashes, and the 55 logical
checkpoint configurations are preserved; 53 physical weight files remain.
Preserve links in release archives and copies. Frozen manuscript figures remain
in `results/figure_sources/`; disposable plot exports are no longer checksum
dependencies. Existing historical audit reports describe their original layouts
and are not rewritten as if those audits had run after this cleanup.

The later idle-code cleanup removes the unused model config copy, unreachable
canonical-reduction/sorting alternatives, and stale surface comments. The
default automatic reduction and point order are unchanged. The legacy
environment-variable rejection remains as a fail-fast guard, not an alternate
algorithm. See `model_idle_cleanup_verification_20261024.json`: initialization,
RNG and strict loading agree across 55 checkpoints; paired CPU single-RNA
training steps agree in losses, gradients, clipping, Adam state/parameters and
RNG for all three paper protocols; all 610 Test18 predictions are bitwise equal.
Compatibility layers, auxiliary heads and ablation paths are retained.

The 2026-10-24 workflow update adds input/evaluation guards and synchronizes
current HARIBOSS result assets, without changing model or training arithmetic.
See [WORKFLOW_FIXES_20261024.md](WORKFLOW_FIXES_20261024.md) for its 19-test
validation scope, same-context Test18 comparison, and installation boundaries.

## Scientific computation retained

The final model consumes frozen 768D ERNIE-RNA embeddings, post-softmax
RNA-renormalized attention, RNA residue coordinates, and ligand-free surface
caches. No ligand coordinates or experimental labels enter prediction inputs.
The canonical prepared records replace the inactive historical 136D tensor
carrier. They and the precomputed ERNIE/surface inputs are now stored together
in six `data/processed/*.pt` packages; no new feature definition is introduced.
Coordinates, labels, sequence order, surface ownership and cache tensors are
copied from the actual experimental inputs, not rebuilt. Ordered manifests are
in `data/splits/`; raw structures and source annotations are inspection-only.
The sibling `data-原始的形式` is preserved unchanged and is not loaded by any
benchmark command. Packages use only ordinary tensor/dictionary/list/scalar
objects, loadable with `weights_only=True`. Runtime carriers preserve separate
RNA attention matrices and local point-owner indices during PyG batching.

`data_package_verification_cpu.json` and `data_package_verification_cuda.json`
record the paired migration checks. All 885 original record inputs and all
ordered split lists were compared; original source-file checksums remain
unchanged. CPU/CUDA checks compare initialization/RNG and 55 strict checkpoint
loads, eight two-RNA molecular training scenarios (loss/gradients/clipping/one
Adam update), full Test18/TL12/JL10 prediction arrays, and two test RNAs from
each HARIBOSS split. The 11 fast tests include packed batching and the existing
epoch-boundary resume check. CLI training was exercised with `--smoke-only`;
no complete 100-epoch experiment was repeated for this data-layout migration.

* Train60: the original ordered 60 records are split 54/6 with split seed 42.
  Training seed and split seed are distinct. Test18 has 18 RNAs / 610 residues.
* HARIBOSS: four official structure-based split manifests are retained. The
  353 multichain graph records contain 35,114 residues / 4,958 positives.
* External protocol: the official 432-record HARIBOSS+TR60 manifest is used
  (345 training / 43 validation / 44 internal test). The extra 37 candidate
  records are not silently included. TL12 contains 515 residues / 183 positives;
  current JL10 contains 821 residues / 264 positives with the established GTP
  removal/alignment processing preserved.

Published HARIBOSS table seeds are 1,5,12,20,23; published TL12/JL10 table seeds
are 15,20,24,28,46. The latter are the actual runs underlying the manuscript,
not a newly substituted seed list. The seed1 external model retained for the
7L0Z_G case is separate from those five table runs. Later ongoing experimental
runs are not automatically incorporated into a frozen paper release.

## Model and optimizer compatibility

The final model retains the old `state_dict` module names for strict loading.
`lin4`, `lin1`, `lin2`, `bn1` are explicitly unused compatibility layers. Their
construction order is retained because removing them changes initialization
of later active layers. No original candidate graph branches are executed.

The final-state-only cleanup removes unused GCNII variant/dimension-changing
branches, unused intermediate-state concatenations, the identity surface
projection and the unused classifier-embedding return option. GCNII keeps the
same equal-width arithmetic and residual addition order. Fixed-mixture
adjacency is computed only when the learned relation gate is absent; all
single-graph and no-gate ablation computations remain unchanged. Dropout calls,
initialization, parameter order, loss and optimizer settings are retained.

`slim_verification_cpu.json` and `slim_verification_cuda.json` record paired
before/after checks on the tested CPU and RTX 4090 / torch 2.2.0 environment:
55 checkpoint architectures, 24 attention-branch settings, eight molecular
training scenarios, RNG states, gradients, clipping, one Adam update including
optimizer moments, and all 610 Test18 probabilities. The reports compare old
and new **within each device**; they do not claim CPU and CUDA are bitwise equal
to each other, or that all 100-epoch runs were repeated.

Train60 historically includes zero times the ATT auxiliary BCE in the loss;
HARIBOSS/external omit that term. Zero gradients and missing gradients differ
for Adam/weight decay. The shared engine preserves this protocol distinction.
The actual surface auxiliary BCE coefficient is 0.05. This implementation does
not change the objective to GRASSP's weighted BCE or its AdamW optimizer.

Single-branch ablations mask branch states after normalization while retaining
the fusion head. The no-gate variant retains fixed equal-weight semantic/spatial
messages. Neither is replaced with a different classifier or a no-graph model.

## Checkpoints and metrics

`best_model.pt` in the published checkpoint folders is the original
`lowest_validation_loss.pt`, without tensor conversion. Names and SHA-256
hashes are recorded in `artifact_manifest.json`. Some original result/config
files name validation MCC as a historical default; the raw loss-selected
result block, not that top-level string, is the source of the paper numbers.
The normalized runtime defaults explicitly select minimum mean per-RNA
validation BCE. The validation loader has batch size 1.
The `optimizer`, `threshold` and `checkpoint_selection` configuration fields
document the fixed published protocol, not supported alternative switches.

Test metrics pool all nucleotide labels/scores within a split. Across seeds,
metrics are computed for each run and then averaged; concatenating all seeds'
predictions is not equivalent. The decision is probability **strictly greater
than** 0.5; AUC uses continuous scores. No test-threshold scan or test-based
checkpoint choice is performed by the released engine. Raw metrics are stored
at full precision and formatted once for manuscript tables.

Historical raw results and prediction arrays live in `results/raw`. They may
contain obsolete source-server paths as provenance only; no runtime uses those
paths. Historical scripts and the private `.archive` copy have been removed
from this cleaned release; the unchanged sibling `paper_code` retains them.

## Determinism and resume

The engine restores Python/NumPy/PyTorch/CUDA RNG state, the shuffled training
loader generator and Adam state from `resume.pt` at an epoch boundary. It keeps
only the latest full resume state instead of all epoch weights. Old published
weights-only files are not complete training-resume checkpoints.

Strict deterministic algorithms are enabled, CuDNN benchmarking and TF32 are
disabled, and CUBLAS workspace configuration is `:4096:8`. Automatic KeOps
reduction mode is retained. Exact equality across GPU types, drivers or library
versions is not promised. The included equivalence audit records the environment
and scope actually tested; it is not a claim that every 100-epoch experiment
was retrained during code cleanup.

## Environments

`environment.yml` specifies the actual Python 3.8 / torch 2.2 training stack.
Training/evaluation with existing caches does not require fairseq or PyMesh.
ERNIE extraction uses a separate Python 3.9/fairseq environment. The original
upstream environment file and source are retained with ERNIE-RNA; the observed
local extraction stack is recorded in `PREPROCESSING.md`.

Raw surface reconstruction invokes Reduce 3.23, MSMS 2.6.1, then a separate
Python 3.7 / PyMesh 0.3 process. Do not attempt to install that legacy PyMesh
wheel into the training environment. PyMOL rendering is optional and independent
of model training. Ligands in figures are visualization references only.
