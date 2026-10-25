# RiboSurfBind

RNA–small molecule binding-site prediction integrating nucleotide-relation
learning and molecular-surface learning.

RiboSurfBind requires an RNA sequence and structure-derived inputs; it is not
a sequence-only predictor. The ligand is not a model input. Frozen benchmark
features and RiboSurfBind-trained checkpoints are included in this repository,
so the example and benchmark workflows do not require feature regeneration.

## 1. Download the repository and data

Install [Git LFS](https://git-lfs.com/), then run:

```bash
git clone https://github.com/chaoxueqiang/RiboSurfBind.git
cd RiboSurfBind
git lfs install --local
git lfs pull
```

Two data files use LFS: `data/processed/hariboss.pt` and
`data/processed/hariboss_tr60.pt` (approximately 545 MB combined). A source ZIP
may contain only their small pointer files. Use an LFS-enabled checkout and
complete `git lfs pull` before running tests or HARIBOSS workflows.

## 2. Install the main environment

The tested paper-training stack is Linux x86-64, Python 3.8 and PyTorch
2.2.0+cu121. GPU execution also requires a compatible NVIDIA driver, CUDA
toolkit/NVRTC and a C++ compiler for KeOps.

Follow the pinned [GPU pip/venv installation recipe](docs/ENVIRONMENTS.md#tested-gpu-installation-primary-paper-training-route).
The same guide includes optional CPU and Conda routes, separate preprocessing
environments, and the scope of installation validation. CPU inference is
supported, but the paper training uses GPU; Conda resolution was not validated
successfully on the test server.

Only the main environment is needed for the frozen-input example, benchmark
training and evaluation. The ERNIE and PyMesh environments are needed only
when rebuilding inputs from structures and sequences. MSMS and the ERNIE-RNA
pretrained weight are not bundled; obtain them from their official sources
only if needed for preprocessing, following [EXTERNAL_ASSETS.md](docs/EXTERNAL_ASSETS.md).

## 3. Check the installation and run one RNA

In the activated main environment, from the repository root:

```bash
python scripts/verify_assets.py
python -m unittest discover -s tests -v
python examples/2JUK_A/run_example.py --device cuda:0
```

The example predicts all 22 nucleotides of PDB **2JUK, chain A**, using the
included label-free inputs and Train60 seed1 checkpoint. It checks residue
identities, binary decisions and reference probabilities, then writes
`2JUK_A.csv` and `verification.json` under `results/new_runs/example_2JUK_A/`.
Successful verification reports `PASS`. Omit `--device cuda:0` for CPU inference.
See the [example guide](examples/2JUK_A/README.md) for input files, expected
output and optional feature-reconstruction commands.

To check one GPU training update without starting a full experiment:

```bash
python train.py --protocol test18 --seed 1 --device cuda:0 --smoke-only \
  --output results/new_runs/training_smoke
```

Use a new output directory for each check. See the
[remote-clone validation record](docs/REMOTE_CLONE_VALIDATION.md) for the tested
snapshot, asset checks, 36-test suite and GPU results.

## Data and checkpoints

`data/processed/` contains six self-contained packages: `train60.pt`,
`test18.pt`, `hariboss.pt`, `hariboss_tr60.pt`, `tl12.pt` and `jl10.pt`.
They include nucleotide identities/coordinates, frozen ERNIE embeddings and
attention, labels, and molecular-surface inputs. Benchmark loading does not
read original PDB files or intermediate feature directories.

`data/splits/` preserves ordered protocol splits. `data/raw/` contains source
structures, annotations and provenance for inspection or reprocessing. See
[data/README.md](data/README.md) for the package schema. `--data-root` means the
directory containing both `processed/` and `splits/`.

| Protocol | Training / validation | Evaluation | Checkpoint directory |
| --- | --- | --- | --- |
| `test18` | Train60: 54 / 6 RNAs | Test18: 18 RNAs | `checkpoints/train60/seed1/` |
| `hariboss` | Official Set1–Set4 splits | Corresponding held-out set | `checkpoints/hariboss/set<1-4>/seed<seed>/` |
| `external` | HARIBOSS+TR60: 345 / 43 RNAs; 44 internal-test RNAs retained separately | TL12 and JL10 | `checkpoints/hariboss_tr60/seed<seed>/` |

Each published checkpoint has `best_model.pt` and a sibling `config.json`.
The selected checkpoint minimizes validation loss; the test set is not used
by the checkpoint selector. The saved HARIBOSS seeds are **1, 5, 12, 20, 23**;
the external benchmark seeds are **15, 20, 24, 28, 46**. Exact settings are in
the checkpoint configurations and [configs/protocols.json](configs/protocols.json).

Input/weight and result SHA-256 indexes are in
[release_asset_checksums.csv](docs/release_asset_checksums.csv) and
[result_asset_checksums.csv](docs/result_asset_checksums.csv).
[artifact_manifest.json](docs/artifact_manifest.json) records model provenance.
There are 55 checkpoint configurations backed by 53 physical weight files;
duplicate assets use relative, repository-internal symbolic links listed in
[asset_aliases.json](docs/asset_aliases.json). Preserve symbolic links when
copying or archiving the repository.

## Evaluate a checkpoint

```bash
python evaluate.py --protocol test18 \
  --checkpoint checkpoints/train60/seed1/best_model.pt \
  --device cuda:0 --output results/new_runs/test18_evaluation

python evaluate.py --protocol hariboss --set 2 \
  --checkpoint checkpoints/hariboss/set2/seed1/best_model.pt \
  --device cuda:0 --output results/new_runs/hariboss_set2_evaluation

python evaluate.py --protocol external --split tl12 \
  --checkpoint checkpoints/hariboss_tr60/seed15/best_model.pt \
  --device cuda:0 --output results/new_runs/tl12_seed15_evaluation
```

For JL10, change `--split tl12` to `--split jl10` and choose a different output
directory. `--device cpu` is available for CPU evaluation.

Evaluation exports pooled nucleotide-level metrics and prediction arrays.
Decisions use **`probability > 0.5`**, not a threshold optimized on the test
set. The evaluator checks checkpoint protocol and HARIBOSS set compatibility;
if `--set` is omitted, it is read from the checkpoint configuration.

## Train and resume

```bash
python train.py --protocol test18 --seed 1 --device cuda:0 \
  --output results/new_runs/test18_seed1

python train.py --protocol hariboss --set 1 --seed 1 --device cuda:0 \
  --output results/new_runs/hariboss_set1_seed1

python train.py --protocol external --seed 15 --device cuda:0 \
  --output results/new_runs/external_seed15
```

The default [Trial20 configuration](configs/trial20.json) uses 100 epochs,
batch size 16, Adam, linear warmup followed by cosine decay, and strict
determinism controls. Each run saves the validation-selected `best_model.pt`
and latest full `resume.pt` state.

To resume at a completed epoch boundary, keep the same configuration, seed,
output directory and other arguments, and add:

```bash
--resume results/new_runs/test18_seed1/resume.pt
```

Published `best_model.pt` weights alone do not contain the optimizer and RNG
states needed to resume a training trajectory. Compatibility layers and model
initialization order are intentional; see [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
before changing model construction. Exact agreement across different hardware
or library versions is not guaranteed.

## Paper experiments and figures

The experiment launcher prints the job list without running it:

```bash
python scripts/run_experiments.py --suite benchmarks --device cuda:0
python scripts/run_experiments.py --suite ablations --device cuda:0
python scripts/run_experiments.py --suite sensitivity --device cuda:0
```

Add `--execute` to start the requested jobs. This can launch full training;
it is not an installation check. Five ablations and three one-factor
sensitivity grids are defined in [ablations.json](configs/ablations.json)
and [sensitivity.json](configs/sensitivity.json).

Unrounded paper metrics are stored in `results/published_metrics.csv`;
`results/raw/` contains retained per-run metrics/logs and available predictions.
Regenerate numerical plots and the three-decimal summary table with:

```bash
python scripts/reproduce_figures.py
```

Baseline values are retained literature comparison results, not newly rerun
baseline models. Frozen figure assets are in `results/figure_sources/`;
generated figures under `results/reproduced_figures/` are not frozen inputs.

Four case studies (1NEM_A, 1Q8N_A, 1AKX and 7L0Z_G) are configured in
[cases.json](configs/cases.json), with retained predictions and molecular
scenes. In an optional PyMOL environment with NumPy and Matplotlib, run
`python scripts/render_cases.py`. The rendering preserves the original
cameras, TP/FP/FN colors and ligand display. PyMOL display surfaces are not
the MSMS/PyMesh surfaces used as model inputs.

## Prepare and predict a new RNA

Follow [PREPROCESSING.md](docs/PREPROCESSING.md) to prepare nucleotide records,
frozen ERNIE features and ligand-free molecular-surface features in their
respective environments. Once those inputs have been prepared:

```bash
python data_process/pack_dataset.py --dataset-root data/raw/custom \
  --name custom --output data/processed/custom.pt

python predict.py --data data/processed/custom.pt \
  --checkpoint checkpoints/train60/seed1/best_model.pt \
  --device cuda:0 --output results/new_runs/custom_predictions
```

Unlabelled inputs are supported. Prediction CSVs include nucleotide identity,
sequence position, structural residue identifier, probability and binary
label. The preprocessing guide also covers ordered multichain inputs and
training on custom labelled splits.

Use the supplied frozen packages for exact paper-input reproduction. Newly
extracted ERNIE or surface features are not promised to match historical
caches bitwise. Neither the remote-clone checks nor the code-organization
checks repeated all full 100-epoch, multi-seed experiments.

## Citation and questions

Use [CITATION.cff](CITATION.cff) to cite this software repository. It currently
uses the confirmed software maintainer's name, Xueqiang Chao; it is not the
manuscript author list. The manuscript citation will be added when its author
list and bibliographic information are confirmed. No paper DOI or publication
details are asserted here.

For reproducibility questions or bug reports, open a
[GitHub issue](https://github.com/chaoxueqiang/RiboSurfBind/issues) and include
the command, environment, checkpoint and error message, without credentials
or private data.

## License and third-party assets

Original RiboSurfBind code and associated documentation are licensed under
the [MIT License](LICENSE), Copyright (c) 2026 Xueqiang Chao. Third-party code
retains its own terms; this code-license choice does not automatically
relicense datasets, weights or result assets.

**Third-party redistribution clearance is incomplete.** The
[third-party notices](THIRD_PARTY_NOTICES.md) and
[redistribution audit](docs/REDISTRIBUTION_AUDIT.md) describe unresolved
permissions, including GeoBind/dMaSIF lineage and source data/weights.
Technical validation is not a license clearance. MSMS and the ERNIE-RNA
pretrained weight are external-download assets, not included in this checkout.
