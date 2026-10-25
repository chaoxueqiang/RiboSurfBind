"""Deterministic settings used for the published Trial20 experiments."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import numpy as np
import torch

def seed_all(seed, strict=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(strict)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(bool(strict), warn_only=False)
    if os.environ.get('MVRBIND_CANONICAL_REDUCTIONS', '0') != '0':
        raise RuntimeError('Published runs use automatic KeOps reductions; unset MVRBIND_CANONICAL_REDUCTIONS')

def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                loader=generator.get_state())

def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('GPU resume state requires a compatible GPU environment')
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])
