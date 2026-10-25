# Input preparation

The included benchmark caches are the exact inputs used for the manuscript.
They are now packaged in `data/processed/*.pt`, with ordered manifests in
`data/splits/`. They permit training and evaluation without reading raw PDBs,
downloading or rerunning legacy surface software. The scripts below support
new RNA preparation; intermediate caches live under `data/raw/custom/`.

## 1. Residue records

```
python data_process/prepare_records.py --pdb RNA.pdb --chain A \
  --record-id RNA_A --output data/raw/custom
```

This appends `records/RNA_A.npz` and both surface-manifest tables without
discarding earlier samples. Duplicate IDs are rejected by default. Explicit
`--overwrite` is permitted only before a surface cache exists for that ID;
otherwise use a fresh output to prevent stale structure-derived inputs.
Inconsistent old records/manifests are rejected, not silently repaired.
NPZ fields are
`record_id`, `sequence`, `xyz[N,3]`, `residue_ids[N]`; optional `y[N]` is accepted
for labelled benchmarks but is not required for prediction. Residue positions
use the retained experimental anchor/alignment implementation. For a multichain
benchmark record, keep the official concatenation order and explicit chain/
residue identifiers; never split a published multichain record silently.

### Batches, multichain entities, and labels for new data

For one multichain RNA use `--chains A,B` instead of `--chain A`. Chain order
is explicit and preserved. Each residue keeps its chain:number:insertion ID.
The legacy surface PDB export requires single-character chain IDs; normalize
long mmCIF chain IDs explicitly before preparation rather than truncating them.
For batches use a CSV/TSV table and run:

```
python data_process/prepare_records.py --manifest samples.tsv --output data/raw/custom
```

Required columns are `record_id`, `pdb`, `chains`; optional columns are
`model_index`, `split`, `labels`, `ligand_pdb`, `cutoff`, `contact_boundary`,
and `exclude_residues`. File paths in the table are relative to the table's
directory unless absolute. A comma-separated chain list must be quoted in CSV.
Use unique IDs and explicit train/validation/test assignments for new labelled
datasets; prediction is the default. No random re-splitting is performed.

Labels may be imported with `--labels labels.tsv` (or the batch column).
The table has `residue_id,label`, optionally `record_id`; identities must cover
the selected residues exactly once, with labels 0 or 1. Row order is irrelevant.
Alternatively use `--ligand-pdb selected_ligand.pdb` with a separate file
containing only the intended ligand, in the same coordinate frame as the RNA.
For new labels only, the default is any RNA heavy atom within **≤4 Å** of any
selected ligand heavy atom. `--cutoff` and `--contact-boundary exclusive` make
the rule explicit. These options do not regenerate or redefine paper labels.
Ligand coordinates are used only in label generation and never packed as model
inputs. Users must select the correct ligand and remove unrelated species from
that explicit ligand file. If a nucleotide-like ligand is recognized as an RNA
residue, exclude it by exact identity, e.g. `--exclude-residues 'A:101:'`;
multiple exclusions are separated by semicolons. Retain the official target
entity and residue mapping for benchmark reprocessing rather than guessing it.

## 2. Frozen ERNIE features

The official ERNIE pretrained weight is not included in the publication copy.
Obtain and verify it using [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md). The default
path below is a user-installed local file, not a bundled checkpoint. For an
alternate location pass its actual path to `--checkpoint`.

Use a separate upstream-compatible extraction environment (not the training
environment). The observed local stack is Python 3.9.23, torch 1.10.0,
fairseq 0.12.2, NumPy 1.26.4, OmegaConf 2.0.6, hydra-core 1.0.7.
The upstream `third_party/ERNIE-RNA-official/environment.yml` is also retained.
The project installation recipes and executable checks for all three
environments are in [ENVIRONMENTS.md](ENVIRONMENTS.md).

```
python data_process/extract_ernie.py --records data/raw/custom/records \
  --output data/raw/custom \
  --checkpoint checkpoints/ernie_rna/ERNIE-RNA_pretrain.pt --device cpu
```

Embeddings use the final encoder layer. Attention bias is softmax-normalized
over the full token set, BOS/EOS are removed, each RNA-only head is renormalized,
heads are averaged and the final RNA rows are renormalized. Do not apply a
second softmax to cached probabilities. Cache identity and row sums are checked.

The new extraction entry point was exercised on a 30-nucleotide RNA. CPU
extraction and the historical cached values are not bitwise identical: maximum
absolute differences were approximately 9.76e-4 for embeddings and 8.94e-8 for
attention. This observation does not establish the cause of the difference.
For exact reproduction of the paper inputs, use the supplied, checksummed
benchmark caches; the cleanup did not regenerate or replace them. See
`feature_extraction_validation.json` for the measured comparison.

## 3. Ligand-free surface features

MSMS must likewise be obtained directly from Scripps and verified, as described
in [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md); no executable is mirrored here.
These two external assets are not needed when using the supplied packed inputs.

```
python data_process/prepare_surface.py --manifest data/raw/custom/surface_manifest \
  --output data/raw/custom/surface --reduce-bin /path/to/reduce \
  --msms-bin /path/to/msms --surface-python /path/to/pymesh37/python
```

For disconnected RNA components, use the retained `--msms-all-components` option
where appropriate; `prepare_surface_fallback.py` retains the audited grouped-chain
fallback for MSMS failures. It is not a replacement for normal joint processing
of contacting chains. See the surface source/environment audit documents.
`validate_cache.py` can audit the final inputs and explicitly repair the known
non-finite-curvature case in a new output cache. Published caches are already
the finite runtime versions; no repair is performed while training.

The 21 chemical/geometric descriptors plus a learned 8D atom-name embedding
form the model's 29D point input. Source-atom residue ownership is retained.
Local frames and mesh repair are generated by the ported surface pipeline, not
by the RNA display surface used in PyMOL figures.

## 4. Pack once, then predict

```
python data_process/pack_dataset.py --dataset-root data/raw/custom \
  --name custom --output data/processed/custom.pt

python predict.py --data data/processed/custom.pt \
  --checkpoint checkpoints/train60/seed1/best_model.pt --output results/custom
```

CSV contains sequence positions, original residue identities, probabilities and
binary predictions. No ligand is needed by the inference model.

The packer checks dimensions, finite values and RNA/surface alignment. It does
not re-extract features, renormalize attention, reorder points or freeze the
learned 8D atom-name embedding. Existing `.pt` outputs are never overwritten.
`data_process/validate_cache.py` still audits optional *intermediate* cache
directories and supports the established explicit curvature-repair operation.
For a packed data file, use `python data_process/validate_cache.py --package
data/processed/custom.pt --output /path/to/validation.json`; validation itself
does not modify the package or its source caches.

## 5. Train/evaluate an explicitly split new dataset

After all labelled records and features are prepared, export their explicit
splits together with the package (all three splits must be nonempty):

```
python data_process/pack_dataset.py --dataset-root data/raw/custom --name custom \
  --output /path/to/new_dataset/processed/custom.pt \
  --split-output /path/to/new_dataset/splits/custom.json
python train.py --protocol custom --data-root /path/to/new_dataset --seed 1 \
  --device cuda:0 --output results/new_runs/custom_seed1
python evaluate.py --protocol custom --data-root /path/to/new_dataset \
  --checkpoint results/new_runs/custom_seed1/best_model.pt \
  --device cuda:0 --output results/new_runs/custom_evaluation
```

This separate custom protocol does not alter any paper benchmark settings or
splits. It checks exact record-ID disjointness and package coverage, not sequence
or structural homology. Design biologically independent splits before preparing
the manifest. The training engine/architecture is the same, with no Test18-only
historical zero-weight ATT auxiliary term. Never label custom results as a
reproduction of the original benchmark.
