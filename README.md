# RiboSurfBind

RNA–small molecule binding-site prediction with semantic–spatial residue
learning and molecular-surface learning.

The public interface consists of `train.py`, `evaluate.py`, and `predict.py`.
The architecture is implemented in `models/`; benchmark protocols and paper
experiment settings are explicit in `configs/`.

## License and public-release status

Original RiboSurfBind code and associated documentation are licensed under
the [MIT License](LICENSE), Copyright (c) 2026 Xueqiang Chao. This grant covers
original contributions only: third-party code keeps its upstream terms, and
datasets, model weights and result assets are not automatically relicensed
under MIT by this code-license choice.

This reproducible working repository is **not yet a license-cleared public
release** because separate third-party permissions remain unresolved.
[Third-party notices](THIRD_PARTY_NOTICES.md) and the
[redistribution audit](docs/REDISTRIBUTION_AUDIT.md) record outstanding
code/data/weight permissions. MSMS and the ERNIE pretrained weight are now
external-download assets, not bundled in the publication snapshot. Do not push
the experiment checkout's old Git history, which still contains those files;
use `scripts/export_public_snapshot.py` to create a fresh, history-free copy.
GeoBind/dMaSIF implementation lineage also requires clarification.

Run `python scripts/check_redistribution.py --check-history` before release.
This advisory check does not change files or enforce Git pushes. Data/weight
versioning below describes the local working repository, not blanket permission
to redistribute every asset.

## Installation

Paper training uses GPU. For the freshly validated GPU pip/venv setup, follow
the pinned recipe in [ENVIRONMENTS.md](docs/ENVIRONMENTS.md). The original
Conda GPU manifest is also retained:

```
conda env create -f environment.yml
conda activate ribosurfbind
```

The training environment uses Python 3.8 and PyTorch 2.2.0. KeOps requires a
working C++ compiler and a writable user cache. CUDA 12.1 is specified for GPU
use; CPU inference is supported but surface convolutions are slower. This
environment specification reflects the experiment stack; fresh GPU pip/venv
validation passed, while the separate Conda solver attempt was blocked by
channel access. Validation scope, extraction/PyMesh recipes and checks are in
[ENVIRONMENTS.md](docs/ENVIRONMENTS.md).

Frozen benchmark features in `data/` and RiboSurfBind-trained weights in
`checkpoints/` are versioned in the same repository. Two files use Git Large
File Storage (Git LFS): `data/processed/hariboss.pt` and
`data/processed/hariboss_tr60.pt`. Their paths and contents are unchanged.
MSMS and `checkpoints/ernie_rna/ERNIE-RNA_pretrain.pt` are separately obtained
preprocessing assets; see [official download and verification instructions](docs/EXTERNAL_ASSETS.md).
Neither is needed for packed training/evaluation or the real-RNA quick example.
Install [Git LFS](https://git-lfs.com/) when obtaining
the repository, then run the following from its root before tests/evaluation:

```
git lfs install --local
git lfs pull
python scripts/verify_assets.py
python -m unittest discover -s tests -v
```

A source ZIP can contain LFS pointer files instead of the actual large assets;
use an LFS-enabled clone or ensure both real data files have been downloaded.
The asset SHA-256 index is in `docs/release_asset_checksums.csv`; model
provenance is in `docs/artifact_manifest.json`. New training/plot outputs and
the old `data-原始的形式/` migration backup are not versioned.
Result/figure hashes are in `docs/result_asset_checksums.csv`;
`python scripts/verify_assets.py` checks both input/weight and result manifests.
Byte-identical default-sensitivity weights and training logs share canonical
files through relative, repository-internal symbolic links documented in
`docs/asset_aliases.json`. There are still 55 checkpoint configurations, backed
by 53 physical RiboSurfBind weight files. Preserve links when copying or creating
release archives (for example, `cp -a` or ordinary `tar` without dereferencing).
All links remain inside the repository.

## Quick example: one real RNA

In the GPU training environment:

```
python examples/2JUK_A/run_example.py --device cuda:0
```

For CPU-only quick inference, omit `--device cuda:0`. Paper training uses GPU;
the CPU default is a convenience for testing installation, not a training change.

This predicts the 22 nucleotides of PDB **2JUK, chain A**, using supplied
label-free frozen inputs and the published Train60 seed1 checkpoint. It checks
residue identities, binary predictions and reference probabilities, then writes
`2JUK_A.csv` and a `PASS` report under `results/new_runs/example_2JUK_A/`.
No ERNIE extraction or PyMesh installation is needed for this quick path.
See [the example guide](examples/2JUK_A/README.md) for the structure, FASTA,
expected output and optional feature-reconstruction commands.

## Data organization

`data/processed/` contains six self-contained tensor packages: `train60.pt`,
`test18.pt`, `hariboss.pt`, `hariboss_tr60.pt`, `tl12.pt`, and `jl10.pt`.
Each RNA includes residue identities/coordinates, frozen ERNIE embeddings and
attention, labels when available, and all molecular-surface inputs. Benchmark
loading does not read intermediate feature directories or original PDB files.
`data/splits/` preserves the original ordered lists; HARIBOSS Set1–Set4 share
one 353-RNA package. `data/raw/` retains structures, annotations and provenance
for inspection/reprocessing, not for training. See [data/README.md](data/README.md)
for the schema. `--data-root` denotes the directory containing `processed/`
and `splits/`, not a single dataset subdirectory.

The supplied benchmark packages retain the exact cached features used in the
paper; they are not regenerated during installation or training.

## Quick evaluation

```
python evaluate.py --protocol test18 \
  --checkpoint checkpoints/train60/seed1/best_model.pt \
  --device cpu --output results/new_runs/test18_evaluation

python evaluate.py --protocol hariboss --set 2 \
  --checkpoint checkpoints/hariboss/set2/seed1/best_model.pt \
  --device cuda:0 --output results/new_runs/hariboss_set2_evaluation

python evaluate.py --protocol external --split jl10 \
  --checkpoint checkpoints/hariboss_tr60/seed15/best_model.pt \
  --device cuda:0 --output results/new_runs/jl10_seed15_evaluation
```

Evaluation exports pooled nucleotide-level metrics and complete prediction
arrays. Predictions use `probability > 0.5`, not a test-optimized threshold.
`best_model.pt` is the original minimum-validation-loss checkpoint.
The evaluator checks the checkpoint protocol and HARIBOSS set before loading
data. If --set is omitted it is taken from the checkpoint metadata; an explicit
incompatible --set or --protocol is rejected, including configuration overrides.

## Training

```
python train.py --protocol test18 --seed 1 --device cuda:0 \
  --output results/new_runs/test18_seed1

python train.py --protocol hariboss --set 1 --seed 1 --device cuda:0 \
  --output results/new_runs/hariboss_set1_seed1

python train.py --protocol external --seed 15 --device cuda:0 \
  --output results/new_runs/external_seed15
```

The external protocol trains on the official HARIBOSS+TR60 345/43/44 split,
then evaluates the same loss-selected model on current TL12 and JL10 inputs.
HARIBOSS preserves each official Set1–Set4 split. Train60 uses its original
54/6 train/validation split; Test18 does not influence checkpoint selection.

`--smoke-only` performs one training update without starting a formal run.
Use a separate output directory for smoke tests. The normal schedule is 100
epochs, batch size 16, Adam, linear warmup followed by cosine decay. See
`configs/trial20.json` for all active parameters.

To resume at a completed epoch boundary, keep the original output directory,
configuration and command arguments, and add:

```
--resume results/new_runs/test18_seed1/resume.pt
```

Old published weights alone cannot resume an optimizer trajectory. New runs
save the latest full `resume.pt` plus the validation-selected `best_model.pt`;
they do not generate 100 redundant weights per run.

Historical checkpoint/RNG compatibility layers are deliberately retained; see
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before altering model construction.

## Reproduce paper experiments

Commands print the complete job list by default. Add `--execute` to run it.

```
python scripts/run_experiments.py --suite benchmarks --device cuda:0
python scripts/run_experiments.py --suite ablations --device cuda:0
python scripts/run_experiments.py --suite sensitivity --device cuda:0
```

The frozen paper HARIBOSS seeds are 1,5,12,20,23. The exact external table seeds are
15,20,24,28,46. Existing runs are retained, not silently replaced by new
training results. Optional `--seeds` changes the requested benchmark jobs;
it does not rewrite the published tables.

Five controlled ablations are configured in `configs/ablations.json`; three
one-factor sensitivity grids are in `configs/sensitivity.json`.

```
python train.py --protocol test18 --ablation no_relation_gate --seed 1 \
  --device cuda:0 --output results/new_runs/no_relation_gate
python train.py --protocol test18 --attention-depth 6 --seed 1 \
  --device cuda:0 --output results/new_runs/depth6
```

## Predict a new RNA

Prepare records, frozen ERNIE features and ligand-free surface features using
`data_process/`. Detailed commands and separate preprocessing environments
are documented in [PREPROCESSING.md](docs/PREPROCESSING.md).
The record preparer safely appends batches, supports ordered multichain entities,
and accepts identity-matched labels or explicit ligand-contact label generation.
New labelled datasets can use the separate --protocol custom path; see the same
guide for explicit train/validation/test manifests. Published inputs are untouched.

```
python data_process/pack_dataset.py --dataset-root data/raw/custom \
  --name custom --output data/processed/custom.pt

python predict.py --data data/processed/custom.pt \
  --checkpoint checkpoints/train60/seed1/best_model.pt \
  --device cuda:0 --output results/new_runs/custom_predictions
```

RNA sequences **and 3D structure-derived inputs** are required. This is not a
sequence-only predictor. The ligand is not required as a model input. Output
CSV gives nucleotide identity, sequence position, structural residue identifier,
probability and binary label. Unlabelled records are accepted.
For debugging preprocessing only, the explicit `--records`, `--embedding-dir`,
`--attention-dir`, `--surface-dir` interface remains available; benchmark jobs
and the normal packaged prediction path do not need it.

## Results and figures

`results/published_metrics.csv` stores unrounded paper metrics.
`results/raw/` retains original per-run metrics/logs and available predictions.
`results/figure_sources/` contains manuscript figure assets and plotting data.

```
python scripts/reproduce_figures.py
```

This regenerates HARIBOSS AUC/MCC comparisons, sensitivity curves and a
three-decimal summary table from numerical files. Baseline numbers are the
retained literature comparison values, not newly rerun baseline models.
Frozen manuscript figures are retained only in `results/figure_sources/`.
`results/reproduced_figures/` is disposable output, excluded from the frozen
asset checksums; regenerating plots does not invalidate release verification.

Four cases are configured in `configs/cases.json`: 1NEM_A, 1Q8N_A, 1AKX and
7L0Z_G. Their original audited predictions and native molecular scenes are
included. In an optional PyMOL environment with NumPy and Matplotlib:

```
python scripts/render_cases.py
```

The script retains original cameras, TP/FP/FN residue colors and yellow ligand
display. PyMOL display surfaces are not the model's MSMS/PyMesh input surface.

## Validation and scope

The regression suite checks model initialization/RNG, strict checkpoint loading,
benchmark splits, metrics, batching and epoch-boundary training resume. Asset
checks cover frozen inputs, weights and results. Detailed equivalence records
and their CPU/CUDA scope are linked from
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Use the frozen benchmark inputs for exact paper-input reproduction. Fresh
ERNIE extraction is not promised to match historical caches bitwise, and exact
agreement across hardware/library versions is not guaranteed. Full 100-epoch
experiments were not repeated during code organization. See
[PREPROCESSING.md](docs/PREPROCESSING.md) and
[ENVIRONMENTS.md](docs/ENVIRONMENTS.md) for the corresponding boundaries.

## Attribution and release checklist

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Before public upload, the
authors must finalize the project license, citation metadata and redistribution
permissions for bundled third-party tools, data and pretrained weights.
