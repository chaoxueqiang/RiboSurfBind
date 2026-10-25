# Reproducibility

This guide records the inputs, computation and evaluation conventions needed
to reproduce the released model. Installation commands are in
[ENVIRONMENTS.md](ENVIRONMENTS.md); current smoke checks and expected results
are in [REMOTE_CLONE_VALIDATION.md](REMOTE_CLONE_VALIDATION.md).

## Frozen inputs and benchmark protocols

Use the six checksummed `data/processed/*.pt` packages for exact paper-input
reproduction. Each contains the actual experimental RNA records, coordinates,
labels, frozen 768D ERNIE-RNA embeddings, post-softmax RNA-renormalized
attention, and ligand-free molecular-surface inputs. Coordinates, residue order,
surface ownership and feature tensors are not regenerated during training.
No ligand coordinates or binding labels enter forward prediction inputs.

Ordered splits are in `data/splits/`. Benchmark commands do not read raw PDBs
or intermediate feature directories. Package tensors and dictionaries are
loadable with `weights_only=True`; batching retains separate RNA attention
matrices and local surface-point ownership indices.

- **Train60/Test18:** the ordered 60 training records use the original 54/6
  split with split seed 42. Training seed and split seed are distinct. Test18
  contains 18 RNAs and 610 nucleotides.
- **HARIBOSS:** retain the four official structure-based split manifests. The
  shared 353-record package contains 35,114 nucleotides and 4,958 positives;
  do not split multichain records into separate samples.
- **External protocol:** the 432-record HARIBOSS+TR60 manifest contains 345
  training, 43 validation and 44 internal-test records. TL12 contains 515
  nucleotides and 183 positives; JL10 contains 821 nucleotides and 264 positives.
  The supplied alignment and nucleotide-like-ligand removal are already
  reflected in these packages and must not be silently changed.

Published HARIBOSS table seeds are **1, 5, 12, 20, 23**; TL12/JL10 table seeds
are **15, 20, 24, 28, 46**. The external seed1 model retained for the 7L0Z_G
case study is separate from those five table runs. New runs do not automatically
replace the frozen paper results.

## Model and optimizer compatibility

The released model keeps the historical `state_dict` names for strict loading.
`lin4`, `lin1`, `lin2` and `bn1` are unused compatibility layers, but their
construction order is intentional: deleting them changes initialization of
later active layers. Auxiliary heads and the order of parameter construction,
dropout calls and residual arithmetic are also retained.

Trial20 settings are in `configs/trial20.json`; the checkpoint's sibling
`config.json` identifies its actual architecture and protocol. Do not remove
compatibility parameters or substitute different initialization in an attempt
to reproduce a saved seed.

Train60 historically includes zero times the ATT auxiliary BCE in the loss;
HARIBOSS/external omit that term. Zero gradients and absent gradients can
produce different Adam/weight-decay behavior. The shared engine preserves this
distinction. The surface auxiliary BCE coefficient is 0.05. The optimizer is
Adam, not AdamW, and the released objective is not a replacement weighted BCE.

Single-branch ablations mask branch states after normalization while retaining
the fusion head. The no-relation-gate ablation uses fixed equal-weight semantic
and spatial messages. These are not alternative classifiers or no-graph models.
Surface-point local frames and automatic KeOps reduction order remain active.

## Checkpoints, metrics and decision threshold

Published `best_model.pt` files are the original minimum-validation-loss
weights, not converted tensors. Their SHA-256 values and source provenance are
recorded in [artifact_manifest.json](artifact_manifest.json). The validation
criterion is minimum mean per-RNA BCE, with validation batch size 1. Test data
does not select the checkpoint.

Historical raw result/config files can name validation MCC as an old default;
the original loss-selected result block is the source of the released paper
metrics. The `optimizer`, `threshold` and `checkpoint_selection` configuration
fields document the fixed protocol, not supported alternative switches.

Metrics pool all nucleotide labels and scores within a test split. For a
multi-seed result, calculate each run's metrics first, then average them;
concatenating predictions across seeds is not equivalent. AUC uses continuous
scores. Binary decisions use **`probability > 0.5`**, without a test-threshold
scan. Store full precision and round only when formatting tables.

`results/raw/` preserves available historical metrics, logs and predictions.
Old server paths in these files are provenance only, not runtime dependencies.
`results/published_metrics.csv` contains unrounded paper values.

## Determinism and epoch-boundary resume

Strict deterministic algorithms are enabled; CuDNN benchmarking and TF32 are
disabled. CUBLAS workspace configuration is `:4096:8`. Keep the pinned software
stack and current computation order. Exact equality across GPU types, drivers,
library versions or CPU/CUDA execution is not guaranteed.

At a completed epoch boundary, `resume.pt` restores model and Adam states,
Python/NumPy/PyTorch/CUDA RNG states, and the shuffled training-loader generator.
Resume with the same configuration, seed and output directory. Only the latest
full resume state is kept. A published weights-only `best_model.pt` cannot
restore an optimizer trajectory.

The regression suite checks initialization and RNG golden hashes, strict
checkpoint loading, ordered data splits, metrics, packed batching and
epoch-boundary resume with a lightweight dropout model. GPU smoke and full
Test18 checks are summarized in
[REMOTE_CLONE_VALIDATION.md](REMOTE_CLONE_VALIDATION.md). These checks are not
a rerun of all full 100-epoch, multi-seed experiments.

## Asset storage and verification

Benchmark data, RiboSurfBind weights and retained results are versioned in this
repository. The two large HARIBOSS packages use Git LFS; run `git lfs pull`
before verification. Pointer files are not usable data packages.

[asset_aliases.json](asset_aliases.json) describes eight relative, internal
links sharing two default-sensitivity weight copies and six identical log
copies. There are 55 checkpoint configurations backed by 53 physical weight
files. Preserve these links when copying or archiving the repository.

```bash
python scripts/verify_assets.py
python -m unittest discover -s tests -v
```

The verifier checks inputs/weights using
[release_asset_checksums.csv](release_asset_checksums.csv), results using
[result_asset_checksums.csv](result_asset_checksums.csv), and all shared-asset
links. Disposable outputs in `results/new_runs/` and
`results/reproduced_figures/` are not frozen assets.

MSMS and the ERNIE pretrained weight are separately obtained preprocessing
assets; see [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md). They are not required for
training/evaluation with frozen packages. New feature extraction is not
promised to match the historical caches bitwise. Prepare new data separately,
following [PREPROCESSING.md](PREPROCESSING.md), rather than overwriting the
benchmark packages. Optional PyMOL display surfaces are not model inputs.
