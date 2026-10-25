# GeoBind surface environment audit

- Legacy interpreter: Python 3.7.13, isolated at `/home2/2025/25cxq/miniconda3/envs/geobind37_legacy`.
- Miniconda installer: `Miniconda3-py37_4.12.0-Linux-x86_64.sh`, SHA-256 `4dc4214839c60b2f5eb3efbdee1ef5d9b45e74f2c09fcae6c8934a13f36ffc3e`.
- PyMesh: official SourceForge `pymesh2-0.3-cp37-cp37m-linux_x86_64.whl`, SHA-256 `da1f2df6c6856f58cf537cb877aae420108d4ac34d2a78bb47769cdcf2a85980`.
- NumPy 1.21.6, SciPy 1.7.3, scikit-learn 1.0.2.
- Reduce: official version 3.23 (2013-05-21), executable SHA-256 `6d066f88bff740627d7c1d2fb0200d326978fe70a0c041a2528c8853e682b1ce`.
- MSMS: official version 2.6.1 static Linux executable, SHA-256 `739b88d8195b89ee5a9c00ef7b986f6bb9d32f7d3b200f457781c98b95958088`.

The Python 3.7 process is used only for the released PyMesh 0.3 mesh-repair
operations. NPZ files form the process boundary; model training remains in the
existing MVRBind environment. A real Train60 RNA (`1ajuA`, 30 residues) passed
the complete ligand-free Reduce -> MSMS -> PyMesh pipeline: 964 protonated
atoms, 1,382 repaired surface vertices, 2,760 faces, and 100% residue coverage.
