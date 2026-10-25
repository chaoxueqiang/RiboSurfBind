# Local installations versus public distribution

Reduce 3.23 remains bundled with its historical notices. MSMS 2.6.1 is now an
external-download asset, not included in the tracked publication tree/snapshot.
The original server installation remains locally available and ignored by Git.

* Reduce: preserve `reduce-3.23/NOTICE` with the unchanged binary. SHA-256:
  `6d066f88bff740627d7c1d2fb0200d326978fe70a0c041a2528c8853e682b1ce`.
* MSMS: original archive README, LICENSE clause 3 restricts transfer of the
  program and derivatives. Without separate applicable permission, do not
  mirror this binary/archive. Users obtain Linux 2.6.1 directly from
  https://ccsb.scripps.edu/msms/downloads/ under Scripps terms.
  Archive SHA-256: `5f0ca50360b5938e74c538e0399d582abc4a40ef4cf410e66f31a1f91e6e3e1f`;
  static executable SHA-256:
  `739b88d8195b89ee5a9c00ef7b986f6bb9d32f7d3b200f457781c98b95958088`.

After lawful local installation, pass explicit `--reduce-bin`, `--msms-bin`
and `--surface-python` paths as described in `../../docs/PREPROCESSING.md`.
Do not replace tools and claim identical features. Frozen-feature inference
does not need either executable.

No local tool or Git history was removed. MSMS is no longer tracked in the
current tree, but older experiment history still contains it. Use the fresh
snapshot exporter described in `../../docs/EXTERNAL_ASSETS.md`; do not copy
the old `.git` into the publication repository. Other outstanding items remain
in `../../docs/REDISTRIBUTION_AUDIT.md` and were not cleared by this exclusion.
