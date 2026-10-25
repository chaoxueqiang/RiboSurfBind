# Remote-clone validation

Validated on **2026-10-25**, using a fresh clone of
[chaoxueqiang/RiboSurfBind](https://github.com/chaoxueqiang/RiboSurfBind),
`main` commit `1d7e5dea293c2e685bb2c68331e0be796b5240a8` (the initial uploaded
snapshot, before this documentation update).

## Isolation and environment

All repository code, benchmark inputs and RiboSurfBind weights were obtained
from GitHub, including both LFS data objects. No source code, benchmark inputs,
model weights or Git object alternates were copied from the experiment
repository, and the new clone used its own LFS object store.

The checks reused the already installed GPU Python environment, a Git LFS
executable and the owner's SSH authentication. They were **not** a new OS or
from-zero environment installation. GPU checks cleared `PYTHONPATH`, disabled
user site packages and used separate KeOps/CUDA cache paths.

Runtime: Linux x86-64, Python 3.8.20, PyTorch 2.2.0+cu121, NVIDIA GeForce
RTX 4090. Installation scope is documented separately in
[ENVIRONMENTS.md](ENVIRONMENTS.md).

## Results

| Check | Result |
| --- | --- |
| Fresh remote clone and `git fsck --full` | PASS |
| Download both LFS objects and `git lfs fsck` | PASS |
| Frozen asset SHA-256 and internal links | PASS: 1,531 files and 8 links |
| `python -m unittest discover -s tests -v` | PASS: 36 tests, no failures or skips |
| Real-RNA frozen-input example on GPU | PASS: 22 nucleotides of 2JUK_A |
| Trial20 GPU one-update training smoke | PASS: seed1, batch size 16, strict determinism enabled |
| Complete Test18 evaluation on GPU | PASS: 18 RNAs; four main metrics exactly match stored values |
| Final clone working tree | Clean |

The two LFS data files were materialized as real tensors, not pointers:

- `data/processed/hariboss.pt`: 337,777,341 bytes.
- `data/processed/hariboss_tr60.pt`: 206,827,902 bytes.

The regression suite includes all six package inventories and ordered protocol
splits, strict loading of all 55 checkpoint configurations, initialization/RNG
golden hashes, fixed-threshold metrics, batching, preprocessing safeguards and
an epoch-boundary resume check with a lightweight dropout model.

The GPU example matched residue identities and predicted labels. Its maximum
absolute probability difference from the frozen reference was
`2.086162567138672e-07` (tolerance `0.0001`). The one-update smoke returned
loss `0.7598628997802734` and gradient norm `1.6506067514419556`.

Test18 seed1 evaluation, compared against `results/published_metrics.csv`:

| Metric | Observed | Difference from stored value |
| --- | --- | --- |
| AUC | 0.7569197204540824 | 0 |
| Precision | 0.6962962962962963 | 0 |
| Recall | 0.45410628019323673 | 0 |
| MCC | 0.4019037696470784 | 0 |

Confusion counts were TP=94, TN=362, FP=41 and FN=113, using
`probability > 0.5`.

## Repeating the checks

After cloning, activating the main environment and completing `git lfs pull`:

```bash
git fsck --full
git lfs fsck
python scripts/verify_assets.py
python -m unittest discover -s tests -v
python examples/2JUK_A/run_example.py --device cuda:0
python train.py --protocol test18 --seed 1 --device cuda:0 --smoke-only \
  --output results/new_runs/remote_clone_smoke
python evaluate.py --protocol test18 \
  --checkpoint checkpoints/train60/seed1/best_model.pt --device cuda:0 \
  --output results/new_runs/remote_clone_test18
```

Use new output directories. These checks did not repeat full 100-epoch,
multi-seed training, regenerate ERNIE/surface features or validate different
hardware/library combinations. Technical validation does not resolve the
outstanding third-party permissions in [REDISTRIBUTION_AUDIT.md](REDISTRIBUTION_AUDIT.md).
