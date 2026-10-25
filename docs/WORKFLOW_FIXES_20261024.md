# Workflow corrections (2026-10-24)

## Changes

- prepare_records.py now appends both manifest tables under an exclusive lock,
  rejects duplicate IDs by default, checks old record/manifest consistency,
  and prevents overwriting a record that already has a surface cache.
- Added batch CSV/TSV input, explicit chain concatenation order, exact residue
  identity label import, and optional contact-label generation from a separate,
  explicitly selected ligand structure. Ligand coordinates are never model inputs.
- Added a separate custom training/evaluation protocol with explicit labelled,
  disjoint train/validation/test IDs; no random split or benchmark mutation.
- evaluate.py checks protocol, HARIBOSS set, and override metadata before
  loading data. An omitted HARIBOSS set is read from checkpoint config.json.
- Synchronized the 20 HARIBOSS runs for seeds 1,5,12,20,23, including original
  logs/configuration, unrounded metrics, plot CSV/JSON and exported AUC/MCC plots.
  Obsolete raw seed2/3/4 directories are moved to an external recoverable backup.
  Original result.json files retain all validation-selected variants; published
  metrics use evaluations.validation_loss.test.fixed_0_5 explicitly, not the
  generic top-level selection description in those archived runner outputs.
- HARIBOSS plots show our mean ± sample SD (ddof=1, five seeds). Literature
  baselines are unchanged; no baseline error bars or significance tests are
  invented. Exported PDF/SVG text is editable, PNG previews are 600 dpi, and
  the combined figure is a two-panel vertical layout. Final geometry checks
  cover fonts, alignment and text/path collisions. PDF/SVG are the vector
  deliverables; no TIFF raster is claimed.
- Added separate ERNIE/PyMesh environment recipes, a component/version/asset
  checker, installation instructions, and workflow regression tests.

## Protected computation

models/*.py, utils/engine.py, utils/reproducibility.py, the six published
input packages, benchmark split files, all checkpoint weights/configurations,
and the source experiment runs are unchanged. No initialization, dropout,
message passing, pooling, loss, optimizer or learning-rate arithmetic is changed.
The custom protocol is an explicitly new data-loading path, not a replacement
for paper experiments. Runtime checks do not modify the existing environments.

## Checks

- 19 tests pass: original 11 plus eight preparation/provenance/result checks.
- A real two-RNA append test retains both samples and all 53 nucleotide mappings.
- A temporary three-RNA cache fixture passes custom packaging/split loading and
  one real-model Adam update. This is interface validation, not a new experiment.
- Original and revised Test18 evaluators are compared in the same CPU execution
  context, including all 610 output probabilities, labels, IDs and offsets.
- The three existing working environments pass their version checks.
- HARIBOSS baseline values and non-HARIBOSS result rows are preserved.
- Package/weight checksums and strict checkpoint/initialization tests are rerun
  after deployment; machine-readable records accompany this note.

## Boundaries

No complete 100-epoch training suite is repeated. New-machine installation of
all three environments is not independently validated; the official legacy
PyMesh wheel and compatible Linux ABI are still prerequisites. New labels use
the explicitly selected cutoff/boundary and do not redefine benchmark labels.
New custom splits need biological independence checks by the dataset author;
record-ID checks do not prove sequence/structure independence. Long mmCIF chain
IDs need explicit normalization for the legacy surface PDB export. Exact paper
input reproduction still uses the frozen feature packages, not fresh extraction.
