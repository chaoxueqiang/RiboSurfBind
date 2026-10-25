#!/usr/bin/env python3
"""Check one preprocessing/training environment without downloading anything."""
import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component', required=True, choices=['training', 'ernie', 'surface'])
    p.add_argument('--output', type=Path)
    p.add_argument('--pymesh-wheel', type=Path, help='Optional official cp37 wheel to verify before installation')
    p.add_argument('--strict-versions', action='store_true')
    p.add_argument('--ernie-checkpoint', type=Path, default=ROOT/'checkpoints/ernie_rna/ERNIE-RNA_pretrain.pt')
    p.add_argument('--msms-bin', type=Path, default=ROOT/'third_party/surface_tools/msms-2.6.1/msms.x86_64Linux2.2.6.1.staticgcc')
    args = p.parse_args()
    modules = {'training': ['torch', 'numpy', 'scipy', 'sklearn', 'Bio', 'torch_geometric', 'pykeops'],
               'ernie': ['torch', 'numpy', 'fairseq', 'omegaconf', 'hydra'],
               'surface': ['numpy', 'scipy', 'sklearn', 'pymesh']}[args.component]
    expected = {'training': {'torch':'2.2.0', 'numpy':'1.24.4', 'scipy':'1.10.1',
                            'sklearn':'1.3.2', 'Bio':'1.83', 'torch_geometric':'2.6.1', 'pykeops':'2.1'},
                'ernie': {'torch':'1.10.0', 'numpy':'1.26.4', 'fairseq':'0.12.2', 'omegaconf':'2.0.6', 'hydra':'1.0.7'},
                'surface': {'numpy':'1.21.6', 'scipy':'1.7.3', 'sklearn':'1.0.2', 'pymesh':'0.3'}}[args.component]
    result = dict(component=args.component, python=sys.version.split()[0], modules={}, errors=[], warnings=[])
    wanted_python = {'training':(3,8), 'ernie':(3,9), 'surface':(3,7)}[args.component]
    if sys.version_info[:2] != wanted_python:
        result['errors'].append('Expected Python %d.%d for this component' % wanted_python)
    for name in modules:
        try:
            module = importlib.import_module(name)
            version = str(getattr(module, '__version__', 'unknown'))
            result['modules'][name] = version
            if version.split('+')[0] != expected[name]:
                result['warnings'].append('%s: expected %s, found %s' % (name, expected[name], version))
        except Exception as error:
            result['errors'].append(name + ': ' + str(error))
    if args.component == 'training' and not shutil.which('g++'):
        result['errors'].append('KeOps needs a working g++ compiler')
    if args.component == 'ernie':
        repo = ROOT/'third_party/ERNIE-RNA-official'
        sys.path.insert(0, str(repo))
        try:
            importlib.import_module('src.utils')
        except Exception as error:
            result['errors'].append('Bundled ERNIE loader: ' + str(error))
        expected_ernie = json.loads((ROOT/'docs/external_assets.json').read_text())['assets']['ernie']['sha256']
        if not args.ernie_checkpoint.is_file() or file_sha256(args.ernie_checkpoint) != expected_ernie:
            result['errors'].append('Missing or mismatched ERNIE checkpoint; obtain officially, see docs/EXTERNAL_ASSETS.md or pass --ernie-checkpoint')
    if args.component == 'surface':
        checksums = {'third_party/surface_tools/reduce-3.23/reduce': '6d066f88bff740627d7c1d2fb0200d326978fe70a0c041a2528c8853e682b1ce',
                     str(args.msms_bin.resolve()): '739b88d8195b89ee5a9c00ef7b986f6bb9d32f7d3b200f457781c98b95958088'}
        if args.pymesh_wheel:
            checksums[str(args.pymesh_wheel.resolve())] = 'da1f2df6c6856f58cf537cb877aae420108d4ac34d2a78bb47769cdcf2a85980'
        for name, wanted in checksums.items():
            path = ROOT/name
            if not path.is_file() or file_sha256(path) != wanted:
                result['errors'].append('Missing or mismatched external asset: ' + name + '; see docs/EXTERNAL_ASSETS.md')
    result['status'] = 'FAIL' if result['errors'] or (args.strict_versions and result['warnings']) else 'PASS'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
