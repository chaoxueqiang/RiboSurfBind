# Installation and environment checks

Use the main environment for training, evaluation and the frozen-input example.
Only feature reconstruction needs the separate ERNIE and PyMesh environments.
Do not upgrade an existing experiment environment in place: these roles use
different Python/PyTorch versions.

## Validation status (2026-10-25)

The pip/venv recipes were tested with freshly installed packages on an existing
Linux server. Python interpreters, OS/compiler, GPU driver and CUDA toolkit
were already available; this was not a new OS/container installation.

| Role | Tested stack | Runtime checks |
| --- | --- | --- |
| Main, GPU | Python 3.8.20, Torch 2.2.0+cu121, RTX 4090 | Imports, real inference/backpropagation, repeat/resume checks and Test18 comparison |
| Main, optional CPU | Python 3.8.20, Torch 2.2.0+cpu | Imports, real inference and backward/Adam step |
| ERNIE extraction | Python 3.9.23, Torch 1.10.0+cpu, fairseq 0.12.2 | Official-weight loading and real feature extraction |
| PyMesh helper | Python 3.7.13, PyMesh 0.3 | MSMS mesh repair and surface feature generation |

GPU and CPU main environments are alternatives, not two required installations.
The later [remote-clone validation](REMOTE_CLONE_VALIDATION.md) records the
36-test suite, GPU example, Trial20 batch-size-16 training smoke and complete
Test18 evaluation using code/data/weights fetched from GitHub.

Feature reconstruction is functional but is not promised to match frozen
benchmark inputs bitwise. Full 100-epoch experiments were not repeated.
Conda-prefix attempts were blocked by channel HTTP 000/timeouts; successful
pip/venv checks do not validate the Conda solver route. The PyMesh runtime
check used a hash-verified wheel; a new wheel download was not verified.

## Tested GPU installation: primary paper-training route

The paper model was trained on GPU. The CPU route below is a convenient
functional check, not a replacement for GPU validation. The GPU pip/venv
recipe uses the same Python 3.8.20 and PyTorch 2.2.0+cu121 build as the original
experiment environment, without modifying or cloning it:

```
python3.8 -m venv /absolute/path/to/envs/ribosurfbind-gpu
source /absolute/path/to/envs/ribosurfbind-gpu/bin/activate
python -m pip install pip==23.3.1 setuptools==68.2.2 wheel==0.41.3
python -m pip install 'https://download.pytorch.org/whl/cu121/torch-2.2.0%2Bcu121-cp38-cp38-linux_x86_64.whl#sha256=a27daf0405b924359795c39a5a73ff31151880d63f65e3f0051c22f9bb3b231e' nvidia-nvjitlink-cu12==12.9.86
python -m pip install --no-build-isolation -r environments/requirements-training.txt
python -m pip check
python scripts/check_environment.py --component training --strict-versions
python -c "import torch; assert torch.version.cuda == '12.1'; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python examples/2JUK_A/run_example.py --device cuda:0
python train.py --protocol test18 --seed 1 --device cuda:0 --smoke-only \
  --output results/new_runs/install_gpu_smoke
```

The smoke command uses the unchanged Trial20 configuration, including batch
size 16 and strict determinism. It performs one update, not a full training
run; choose an unused output directory. The nvJitLink pin matches the original
environment. A compatible NVIDIA driver and CUDA toolkit/NVRTC for KeOps must
already be installed. The Torch build targets CUDA 12.1; the validation host's
separate pre-existing system toolkit is CUDA 12.2. Do not confuse the Torch
runtime version with the driver-supported CUDA version shown by `nvidia-smi`.

## Tested CPU installation route (optional)

Prerequisites: Linux x86-64, C/C++ compiler, Python 3.8/3.9/3.7 interpreters with
`venv` support, and a writable KeOps cache. Replace `/absolute/path/to/envs`
with a directory outside the repository. Activate each environment: KeOps 2.1
invokes `python3` from `PATH` during compilation; an absolute interpreter path
alone is insufficient if a different environment is active.

### Training, evaluation and main preprocessing

```
python3.8 -m venv /absolute/path/to/envs/ribosurfbind
source /absolute/path/to/envs/ribosurfbind/bin/activate
python -m pip install pip==23.3.1 setuptools==68.2.2 wheel==0.41.3
python -m pip install --no-deps 'https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp38-cp38-linux_x86_64.whl'
python -m pip install --no-build-isolation -r environments/requirements-training.txt
python -m pip check
python scripts/check_environment.py --component training --strict-versions
python -m unittest discover -s tests -v
python examples/2JUK_A/run_example.py
```

Preinstalling the CPU wheel satisfies `torch==2.2.0` without fetching a CUDA
stack. Official guidance: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).
First inference compiles KeOps kernels. Full CPU training is slow; use the paper
GPU stack for practical training. This main environment also runs
`prepare_records.py`, `prepare_surface.py`, `pack_dataset.py` and `validate_cache.py`.

### Frozen ERNIE extraction

```
python3.9 -m venv /absolute/path/to/envs/ribosurfbind-ernie
source /absolute/path/to/envs/ribosurfbind-ernie/bin/activate
python -m pip install pip==23.3.1 setuptools==68.2.2 wheel==0.41.3 Cython==0.29.36
python -m pip install --no-deps 'https://download.pytorch.org/whl/cpu/torch-1.10.0%2Bcpu-cp39-cp39-linux_x86_64.whl'
python -m pip install --no-deps 'https://download.pytorch.org/whl/cpu/torchaudio-0.10.0%2Bcpu-cp39-cp39-linux_x86_64.whl'
python -m pip install numpy==1.26.4 scipy==1.13.1 scikit-learn==1.6.1
MAX_JOBS=2 python -m pip install --no-build-isolation -r environments/requirements-ernie.txt
python -m pip check
python scripts/check_environment.py --component ernie --strict-versions
```

Actual fresh installation exposed and resolved two problems:

- The bundled ERNIE loader imports `sklearn`; it must be explicitly installed.
- The PyPI fairseq 0.12.2 source archive lacked
  `fairseq/clib/libbase/balanced_assignment.cpp` and failed compilation. Use
  the complete [official v0.12.2 source](https://github.com/facebookresearch/fairseq/tree/v0.12.2),
  not a different fairseq version. Install Torch/NumPy/Cython/build tools first
  and disable build isolation. The tested source archive SHA-256 is
  `b5bb076f8b0d5b06a897b11627da278816752b2202650c34becd8f4240f86961`.

Keep pip below 24.1 because newer installers reject legacy OmegaConf metadata.
The optional fairseq tensorboardX message does not prevent frozen extraction.
The vendored source and official pretrained weight must be present for feature
extraction. The weight is **not bundled or mirrored** in the publication copy;
obtain it directly from the authors using [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md),
verify its SHA-256, and install it at `checkpoints/ernie_rna/ERNIE-RNA_pretrain.pt`
or pass a separate local `--checkpoint` path to extraction. The environment
checker accepts `--ernie-checkpoint /path` as well. Packed training/evaluation
and the frozen-input quick example do not need this pretrained weight.

### PyMesh helper

Obtain the exact wheel from the [official PyMesh v0.3 release](https://github.com/PyMesh/PyMesh/releases/tag/v0.3):

```
curl -L --fail --retry 3 \
  -o pymesh2-0.3-cp37-cp37m-linux_x86_64.whl \
  https://github.com/PyMesh/PyMesh/releases/download/v0.3/pymesh2-0.3-cp37-cp37m-linux_x86_64.whl
sha256sum pymesh2-0.3-cp37-cp37m-linux_x86_64.whl
python3.7 -m venv /absolute/path/to/envs/ribosurfbind-surface
source /absolute/path/to/envs/ribosurfbind-surface/bin/activate
python -m pip install pip==23.1.2
python -m pip install -r environments/requirements-surface.txt
python -m pip install --no-deps ./pymesh2-0.3-cp37-cp37m-linux_x86_64.whl
python -m pip check
python scripts/check_environment.py --component surface --strict-versions \
  --pymesh-wheel ./pymesh2-0.3-cp37-cp37m-linux_x86_64.whl
```

Expected SHA-256:
`da1f2df6c6856f58cf537cb877aae420108d4ac34d2a78bb47769cdcf2a85980`.
This validation installed a previously downloaded, hash-verified **wheel** into
the empty environment; a new GitHub wheel download timed out. Fresh package
installation/runtime passed, but a new end-to-end wheel download was not
verified. The wheel is not bundled. Do not install it in Python 3.8 or substitute
a similarly named PyPI package.

## Conda alternatives

The original GPU training manifest remains unchanged:

```
conda env create -f environment.yml
conda activate ribosurfbind
python scripts/check_environment.py --component training --strict-versions
```

For ERNIE, create the base environment, then install the complete fairseq source
separately so its build sees the installed dependencies:

```
conda env create -f environments/ernie-rna.yml
conda activate ribosurfbind-ernie
MAX_JOBS=2 python -m pip install --no-build-isolation -r environments/requirements-ernie.txt
python scripts/check_environment.py --component ernie --strict-versions
```

`conda env create -f environments/surface.yml` is an alternative to the PyMesh
venv step; install the same wheel afterward. Fresh Conda resolution, compiler/OS
ABI and GPU support remain unverified because channel access failed.

## Reduce and MSMS

Reduce 3.23 remains supplied with its historical notices. MSMS 2.6.1 is
**not bundled** in the publication copy: obtain it directly under Scripps terms
using [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md), verify the exact version/hash
and supply `--msms-bin /path/to/msms` to surface preparation. The environment
checker also accepts `--msms-bin /path/to/msms`. The original server checkout
retains its ignored local installation for reproducibility.
The unchanged Reduce 3.23 binary has its historical declarations in
`third_party/surface_tools/reduce-3.23/NOTICE`; its old publisher endpoint is
unreliable. Recovery provenance is in [GEOBIND_SOURCE_AUDIT.md](GEOBIND_SOURCE_AUDIT.md).
See [REDISTRIBUTION_AUDIT.md](REDISTRIBUTION_AUDIT.md) for restrictions and the
existing-history caveat. Frozen-feature inference needs neither executable.

```
sha256sum third_party/surface_tools/reduce-3.23/reduce
sha256sum third_party/surface_tools/msms-2.6.1/msms.x86_64Linux2.2.6.1.staticgcc
```

Expected hashes, respectively:

```
6d066f88bff740627d7c1d2fb0200d326978fe70a0c041a2528c8853e682b1ce
739b88d8195b89ee5a9c00ef7b986f6bb9d32f7d3b200f457781c98b95958088
```

Run surface preparation in the activated **main training** environment, passing
the separate helper interpreter:

```
python data_process/prepare_surface.py \
  --manifest data/raw/custom/surface_manifest --output data/raw/custom/surface \
  --reduce-bin third_party/surface_tools/reduce-3.23/reduce \
  --msms-bin third_party/surface_tools/msms-2.6.1/msms.x86_64Linux2.2.6.1.staticgcc \
  --surface-python /absolute/path/to/envs/ribosurfbind-surface/bin/python
```

## Reproduction boundary

Exact paper-input reproduction uses supplied, checksummed feature packages.
Fresh extraction is functional but is not promised to match historical caches
bitwise; do not overwrite benchmark inputs. These same-machine package/runtime
checks do not establish numerical identity across GPUs, drivers, operating systems or builds,
nor rerun the paper experiments. The model, compatibility layers, RNG
construction order, training settings and determinism controls are unchanged.
