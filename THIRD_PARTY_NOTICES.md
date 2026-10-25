# Third-party notices and redistribution status

Reviewed 2026-10-25. **This working repository is not yet cleared for public
redistribution.** Attribution is not permission; a paper's open-access license
does not automatically license its software, datasets or external weights.
See [the audit and outstanding decisions](docs/REDISTRIBUTION_AUDIT.md).
Original RiboSurfBind code and associated documentation use the author-approved
[MIT License](LICENSE), Copyright (c) 2026 Xueqiang Chao. This does not relicense
third-party code, datasets, model weights or result assets as a whole, and does
not resolve the separate permissions listed below.

## Incorporated source and definitions

| Component | Local scope | Notice / source | Finding |
|---|---|---|---|
| ERNIE-RNA | `third_party/ERNIE-RNA-official/` | [MIT](third_party/ERNIE-RNA-official/LICENSE), Copyright (c) 2024 Weijie Yin; [source](https://github.com/Bruce-ywj/ERNIE-RNA/tree/43bc06de1088ed03ffd7de918ad4b2c2a3346a43) | Source notice retained; external checkpoint terms separately unresolved. |
| GeoBind | Surface convolution in `models/surface.py`, preparation/mesh-repair ports in `data_process/` | [MIT](licenses/GeoBind-MIT.txt), Copyright (c) 2022 zpliulab; [source](https://github.com/zpliulab/GeoBind/tree/f8e336b8239fecf655d5a306ae56f887b1cf146e) | GeoBind notice retained; underlying dMaSIF lineage is not cleared. |
| dMaSIF | Quasi-geodesic convolution / tangent-frame lineage through GeoBind | [CC BY-NC-ND 4.0](licenses/dMaSIF-CC-BY-NC-ND-4.0.txt); [source](https://github.com/FreyrS/dMaSIF/tree/0dcc26c3c218a39d5fe26beb2e788b95fb028896) | Obtain applicable permission for adapted-source distribution or document another valid grant; retaining the text is not permission. |
| FR3D Python | Donor/acceptor and backbone-acceptor definitions in `data_process/chemistry.py` | [MIT-style notice](licenses/FR3D-MIT.txt), Copyright 2026 BGSU RNA Group; [source](https://github.com/BGSU-RNA/fr3d-python/blob/ed850c00df01616e58c643b0f84bdc69662b5d55/fr3d/definitions.py) | Definition values match the inspected source; notice retained. FR3D search/classifier code is not bundled. |
| PyTorch / fairseq | ERNIE attention/encoder implementation lineage and installed dependencies | [PyTorch](licenses/PyTorch-LICENSE.txt), [fairseq MIT](licenses/fairseq-MIT.txt) | Preserve upstream notices with the ERNIE subset; installed distributions retain their own terms. |
| ATMGBs / GCNII / MVRBind | Graph/model implementation ancestry under review | [ATMGBs](https://github.com/lixiangli01/ATMGBs), [GCNII](https://github.com/chennnM/GCNII), [MVRBind](https://github.com/cschen-y/MVRBind) | Algorithm inspiration is not itself copied-source evidence. Establish lineage and permissions for any incorporated source before clearing it. |

The ERNIE utility subset was pruned to frozen feature extraction: unused CLIs
and downstream helpers were removed; encoder, checkpoint loading and feature
arithmetic were preserved. GeoBind was modified for ligand-free RNA surfaces,
RNA chemistry, process-separated repair and residue pooling. These are ports,
not unchanged upstream releases. [The source audit](docs/GEOBIND_SOURCE_AUDIT.md)
records numerical provenance. No protein-pretrained GeoBind weights are used.

## External geometry tools

* **Reduce 3.23.130521**: unchanged binary in `third_party/surface_tools/reduce-3.23/`.
  [Historical NOTICE](third_party/surface_tools/reduce-3.23/NOTICE) includes the
  original free-software notice, UCSF libpdb terms and hybrid-36 acknowledgment.
  This product includes software developed at the University of California,
  San Francisco, and in the Richardson Lab at Duke University. Do not apply the
  current upstream Apache license retroactively to the old binary.
* **MSMS 2.6.1**: not bundled in the publication snapshot. The historical local
  `third_party/surface_tools/msms-2.6.1/README`, LICENSE
  section, clause 3 restricts transfer to others. Do not upload the program or
  derivatives without a separate applicable permission. Obtain it directly from
  [Scripps](https://ccsb.scripps.edu/msms/downloads/).
  [Official installation guidance](docs/EXTERNAL_ASSETS.md) preserves version
  and hash. The original server installation is ignored and no longer tracked
  in the current tree; old experiment history still contains it and is not
  copied into publication snapshots.
* **PyMesh 0.3**: installed separately, not bundled as a wheel/environment.
  v0.3 metadata says `MPL` without a version; linked libraries have their own
  terms. We do not assert MIT or certify wheel/container redistribution. Use
  the [official installation route](docs/ENVIRONMENTS.md).

Cite Reduce (Word et al., 1999, J. Mol. Biol. 285, 1735–1747) and MSMS
(Sanner, Spehner and Olson, 1996, Biopolymers 38, 305–320) when used.

## Data and model artifacts (not relicensed as source code)

* PDB archive data: [RCSB policy](https://www.rcsb.org/pages/usage-policy), CC0 1.0.
  Retain PDB IDs and credit depositors where possible.
* HARIBOSS archive data: [official policy](https://hariboss.pasteur.cloud/about-policies-logo),
  CC0 1.0; cite Panei et al., 2022, doi:10.1093/bioinformatics/btac483. Neither
  policy automatically covers later benchmark labels, splits, language-model
  features or serialized releases from other repositories.
* RNABind: [MIT notice](licenses/RNABind-MIT.txt), Copyright (c) 2024 csbio@sjtu.
  Retain this attribution for RNABind-origin processing/protocol material.
  Its software license is not proof of rights to every item of a composite dataset.
* CoBRA: [upstream notice](licenses/CoBRA-license-notice.txt) declares GPL v3;
  [the full GPL v3 text](licenses/GPL-3.0.txt) is also retained.
  Preserve the notice with copied CoBRA-origin annotations/splits; confirm data
  scope and distribution obligations. This does **not** mean all RiboSurfBind
  code or learned weights become GPL by training on data. No CoBRA weights are bundled.
* MVRBind / GRASSP source releases: no explicit redistribution grant was
  established from inspected root listings/READMEs. Copied labels, release
  tensors and any copied implementation need applicable terms, permission or
  documented independent provenance. Public download plus citation is not a
  blanket right to relicense.
* ERNIE pretrained checkpoint: source links to Drive; the README's MIT sentence
  explicitly describes source code. Mirroring terms for
  `checkpoints/ernie_rna/ERNIE-RNA_pretrain.pt` remain unconfirmed. It is
  unchanged locally, ignored and no longer tracked. The publication snapshot
  provides the official download pointer and checksum, not the external weight.
  The old experiment LFS history is not copied into the publication snapshot.
* RiboSurfBind-trained weights, new features and results: obtain author-approved
  release terms and review any actually applicable upstream output terms.
  Do not assume software licenses automatically transfer to all outputs.

The [machine-readable inventory](docs/third_party_licenses.json) records scope
and outstanding items. `python scripts/check_redistribution.py` is an advisory
offline release gate, not legal advice or an automatic Git-push block.
Passing numerical/checksum tests is not license clearance.
