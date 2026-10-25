# Real RNA example: 2JUK, chain A

This is a 22-nucleotide RNA from the independent Test18 benchmark:

```
GGCCUUCCCACAAGGGAAGGCC
```

## Quick prediction

From the repository root, in the training environment:

```
python examples/2JUK_A/run_example.py
```

For the paper GPU environment, run the same example explicitly on CUDA:

```
python examples/2JUK_A/run_example.py --device cuda:0
```

Both routes are tested; the default CPU path is only a convenient quick check,
not a claim that the paper model was trained on CPU.

The command calls the unmodified `predict.py`, using the published Train60
seed1 checkpoint. It writes `2JUK_A.csv` and `verification.json` under
`results/new_runs/example_2JUK_A/`. The final report must say `PASS` with
`nucleotides: 22`. The CSV has one row per residue, with sequence position,
structural residue ID, nucleotide, probability, and binary prediction.

The reference starts as follows (probabilities rounded here for display only):

| Position | Residue | Base | Probability | Prediction |
|---|---|---|---:|---:|
| 1 | A:1: | G | 0.129777 | 0 |
| 2 | A:2: | G | 0.135255 | 0 |
| 3 | A:3: | C | 0.095691 | 0 |

`expected_predictions.csv` retains all 22 full-precision reference scores.
Verification requires exact residue identities and binary decisions, with a
maximum absolute probability difference of 1e-4 to allow small numerical
differences between supported environments. The threshold is strictly >0.5.
This demo checks installation/inference, not aggregate scientific performance.

For the ordinary prediction entry point without the comparison wrapper:

```
python predict.py --data examples/2JUK_A/2JUK_A.pt \
  --checkpoint checkpoints/train60/seed1/best_model.pt \
  --device cpu --output results/new_runs/2JUK_A_prediction
```

The quick path needs neither ERNIE extraction nor PyMesh, and does not require
the three large LFS objects. It uses the small supplied example package and
the ordinary Git-tracked RiboSurfBind checkpoint. The model still requires its
structure-derived surface inputs; this is not sequence-only inference.

## Included files and provenance

- `2JUK_A.pdb`: ligand-free RNA chain A, heavy atoms only, exported from the
  retained benchmark structure without changing the residue order/coordinates.
- `sequence.fasta`: the matching RNA sequence.
- `2JUK_A.pt`: frozen embeddings, post-softmax attention, residue coordinates,
  surface point coordinates/frames, chemistry, atom indices and ownership.
- `expected_predictions.csv`: CPU reference from the published checkpoint.
- `manifest.json`: source IDs, file hashes, checkpoint hash, counts and tolerance.

Input tensors are copied from `test18_0005` in `data/processed/test18.pt`, not
recomputed. Only the display record ID is changed to `2JUK_A`; the label vector
is omitted from the demonstration package. Original benchmark files are not
modified. No ligand coordinates or binding labels are required for prediction.

## Rebuild features from this structure (optional)

First install the three environments in `docs/ENVIRONMENTS.md`. From the
repository root, prepare the exact chain identity:

```
python data_process/prepare_records.py --pdb examples/2JUK_A/2JUK_A.pdb \
  --chain A --record-id 2JUK_A --output results/new_runs/2JUK_A_prepared

conda run -n ribosurfbind-ernie python data_process/extract_ernie.py \
  --records results/new_runs/2JUK_A_prepared/records \
  --output results/new_runs/2JUK_A_prepared \
  --checkpoint checkpoints/ernie_rna/ERNIE-RNA_pretrain.pt --device cpu

python data_process/prepare_surface.py \
  --manifest results/new_runs/2JUK_A_prepared/surface_manifest \
  --output results/new_runs/2JUK_A_prepared/surface \
  --reduce-bin third_party/surface_tools/reduce-3.23/reduce \
  --msms-bin third_party/surface_tools/msms-2.6.1/msms.x86_64Linux2.2.6.1.staticgcc \
  --surface-python /absolute/path/to/envs/ribosurfbind-surface/bin/python

python data_process/pack_dataset.py --dataset-root results/new_runs/2JUK_A_prepared \
  --name 2JUK_A --output results/new_runs/2JUK_A_rebuilt.pt

python predict.py --data results/new_runs/2JUK_A_rebuilt.pt \
  --checkpoint checkpoints/train60/seed1/best_model.pt \
  --device cpu --output results/new_runs/2JUK_A_rebuilt_prediction
```

The fresh extraction path is distinct from the frozen-input quick check.
Fresh ERNIE/geometry computations are not promised to reproduce historical
caches bitwise. Do not overwrite benchmark packages or the supplied reference
CSV with regenerated inputs/results.
