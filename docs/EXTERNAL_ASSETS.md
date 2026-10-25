# Separately obtained preprocessing assets

MSMS 2.6.1 and the ERNIE-RNA pretrained checkpoint are **not distributed in
the publication snapshot**. Obtain them directly from the official publishers
under their terms. No automatic downloader accepts terms for you.

Packed benchmark training/evaluation and `examples/2JUK_A/run_example.py` do
not need these two assets: frozen attention, embeddings and surfaces are already
in the supplied packages. All RiboSurfBind-trained checkpoints remain included.

## ERNIE-RNA pretrained checkpoint

1. Follow the author's [fixed-version checkpoint link file](https://github.com/Bruce-ywj/ERNIE-RNA/blob/43bc06de1088ed03ffd7de918ad4b2c2a3346a43/checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_checkpoint.txt).
   It points to [this official Drive download](https://drive.google.com/file/d/1CmNxJxgjDhRoBdlDODFFjDNNzJMHVzmO/view).
2. Place the actual pretrained file at
   `checkpoints/ernie_rna/ERNIE-RNA_pretrain.pt`, or choose an external local
   path and pass `--checkpoint /your/path/ERNIE-RNA_pretrain.pt` to
   `data_process/extract_ernie.py`. Do not use a secondary-structure finetuned
   checkpoint or a different pretrained release.
3. Verify before extraction:

```
python scripts/check_external_assets.py --asset ernie
# If stored elsewhere:
python scripts/check_external_assets.py --asset ernie --path /your/path/ERNIE-RNA_pretrain.pt
```

Expected size: **1,035,221,383 bytes**. SHA-256:
`50ac42d4a2af361b8a7d9fb62546e3f380c32893b179554609440339a6676851`.

Drive access/quotas and future link availability are controlled by the official
publisher. If inaccessible or the checksum changes, contact the publisher;
do not silently replace the weight or claim the new features match the paper.
No source-code MIT license is asserted as a license for this external weight.

## MSMS 2.6.1

1. Read the Scripps [terms](https://ccsb.scripps.edu/msms/license/) and obtain
   the Linux 64-bit 2.6.1 archive from the [official download page](https://ccsb.scripps.edu/msms/downloads/).
   Expected archive name: `msms_i86_64Linux2_2.6.1.tar.gz`.
2. Check the archive SHA-256 before extracting:
   `5f0ca50360b5938e74c538e0399d582abc4a40ef4cf410e66f31a1f91e6e3e1f`.
   Extract into your chosen local tools directory and keep the publisher's
   accompanying terms there. The expected executable is
   `msms.x86_64Linux2.2.6.1.staticgcc`.
3. Verify the executable and provide its path explicitly:

```
python scripts/check_external_assets.py --asset msms --path /your/tools/msms.x86_64Linux2.2.6.1.staticgcc

python data_process/prepare_surface.py --manifest data/raw/custom/surface_manifest \
  --output data/raw/custom/surface \
  --reduce-bin third_party/surface_tools/reduce-3.23/reduce \
  --msms-bin /your/tools/msms.x86_64Linux2.2.6.1.staticgcc \
  --surface-python /your/pymesh37/bin/python
```

Executable size: **943,144 bytes**. SHA-256:
`739b88d8195b89ee5a9c00ef7b986f6bb9d32f7d3b200f457781c98b95958088`.

Alternatively install locally under the ignored path
`third_party/surface_tools/msms-2.6.1/`. The ignored paths are conveniences,
not a permission to upload the tools; do not force-add them to Git.

## Verify preprocessing assets

* `python scripts/verify_assets.py` checks all bundled benchmark/model/result
  assets. The old checksum index still records the exact ERNIE weight for
  provenance; only this explicitly declared external row may be absent by
  default. Present-but-wrong files are rejected. No model/data row is waived.
* `python scripts/verify_assets.py --include-external` also requires both
  external assets in their documented local paths and validates their bytes.
  For alternate paths use `check_external_assets.py` above.
* Environment checks accept `--ernie-checkpoint /path` and `--msms-bin /path`.
  A feature-extraction environment without its required asset should fail;
  ordinary packed training/evaluation does not require those environments.
* Machine-readable versions, hashes, links and release exclusions are in
  `external_assets.json`.

Keep separately downloaded tools and pretrained weights local; do not force-add
them to Git or assume their terms permit redistribution. Source code, official
pretrained weights and RiboSurfBind-trained checkpoints are distinct assets.
Third-party status is documented in [REDISTRIBUTION_AUDIT.md](REDISTRIBUTION_AUDIT.md).
