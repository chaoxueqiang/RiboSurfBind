# Official ERNIE-RNA pretrained checkpoint (not bundled)

This directory intentionally contains no redistributed pretrained weight.
Obtain the author's exact checkpoint following
[EXTERNAL_ASSETS.md](../../docs/EXTERNAL_ASSETS.md), then place it locally as
`ERNIE-RNA_pretrain.pt` or pass its external path with `--checkpoint`.
The installed file is ignored by Git. Official pointer, byte count and SHA-256
are in `../../docs/external_assets.json`.

This is not a RiboSurfBind-trained checkpoint. Packed benchmark training,
evaluation and the quick example do not require it; new ERNIE feature extraction
does. RiboSurfBind-trained weights in the other checkpoint directories remain
included and unchanged.
