# RiboSurfBind benchmark data

Training and evaluation use self-contained packages in `processed/`. Each
package is a plain dictionary containing schema, dataset name, counts and an
ordered list of RNA samples. It can be inspected with
`torch.load(path, map_location='cpu', weights_only=True)` in PyTorch 2.2.
No custom Python classes are pickled in a package.

| Package | RNAs | Role |
| --- | ---: | --- |
| train60.pt | 60 | Fixed 54/6 training/validation split |
| test18.pt | 18 | Independent Test18 benchmark |
| hariboss.pt | 353 | One shared data pool, four official splits |
| hariboss_tr60.pt | 432 | Official 345/43/44 combined-data split |
| tl12.pt | 12 | Independent external benchmark |
| jl10.pt | 10 | Current aligned external benchmark |

## Samples

Each sample includes `record_id`, `display_name`, `sequence`, `residue_ids`,
`pos[N,3]`, `embedding[N,768]`, `attention[N,N]`, optional `y[N]`, and a
`surface` dictionary: `xyz[M,3]`, `nuv[M,3,3]`, `chemistry[M,21]`,
`atom_name_index[M]`, `owner[M]`, `node_count`, `record_id`, `assignment`.
The 8D atom-name embedding is still trained inside the model, not cached here.
Attention provenance and available embedding/surface metadata are retained.
Original source paths in provenance are historical descriptions, never runtime
dependencies. Labels do not enter forward prediction.

Variable-size attention matrices and surfaces are kept separate per RNA at
runtime. Point ownership is local to each RNA and is never batch-offset.
Float32 inputs, int64 indices, residue order and point order are unchanged.
Graphs are constructed by the existing model, not precomputed during packing.

## Splits and raw assets

`splits/train60.json` stores the original ordered train, validation and Test18
ID lists. `hariboss_set1.json` through `hariboss_set4.json` refer to the shared
353-RNA package. `hariboss_tr60.json` stores the combined protocol's ordered
lists. TL12/JL10 manifests retain their original evaluation order.

`raw/structures/` preserves the existing structure files (including ligand
reference files where originally present). `raw/annotations/` holds original
annotations/split source tables. `raw/provenance/` preserves source-release
files and alignment history. None is read during packed training/evaluation.
Raw ligand coordinates are not model inputs.

`manifest.json` lists counts and package SHA-256 values. `source_checksums.csv`
records all files in the untouched pre-migration data directory. That sibling
directory is retained as a backup, but is not required to run any benchmark.

For new RNA preparation, write intermediate caches under `raw/custom/`, then
use `data_process/pack_dataset.py --dataset-root ... --name ... --output ...`.
Prediction accepts `predict.py --data processed/custom.pt` (with the desired
path relative to your working directory); see the repository README for a full
command. Existing benchmark features were packed, not recomputed.
