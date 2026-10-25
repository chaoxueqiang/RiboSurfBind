#!/usr/bin/env python3
"""Verify user-obtained preprocessing assets; never download or accept terms."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_asset(path, specification):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Missing external asset: ' + str(path))
    if path.stat().st_size != specification['bytes']:
        raise ValueError('External asset size mismatch: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != specification['sha256']:
        raise ValueError('External asset SHA-256 mismatch: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', choices=['ernie', 'msms'], required=True)
    parser.add_argument('--path', type=Path, help='Override the documented local installation path')
    args = parser.parse_args()
    assets = json.loads((ROOT / 'docs/external_assets.json').read_text())['assets']
    specification = assets[args.asset]
    path = args.path if args.path else ROOT / specification['path']
    try:
        verify_asset(path, specification)
    except (OSError, ValueError) as error:
        print('FAIL:', error)
        print('Obtain from:', specification['official_url'])
        print('Instructions: docs/EXTERNAL_ASSETS.md')
        return 1
    print('PASS:', args.asset, specification['version'], specification['sha256'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
