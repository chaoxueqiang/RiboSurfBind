#!/usr/bin/env python3
"""Verify the benchmark/weight assets after downloading or copying."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.io import sha256_file

def verify_aliases(root):
    """Require portable, internal links to the documented frozen assets."""
    root = root.resolve()
    rows = json.loads((root/'docs/asset_aliases.json').read_text())
    for row in rows:
        path, target = root/row['path'], root/row['target']
        path.parent.resolve().relative_to(root)
        target.resolve().relative_to(root)
        if (not path.is_symlink() or os.path.isabs(os.readlink(path))
                or path.resolve() != target.resolve() or target.is_symlink()
                or not target.is_file() or sha256_file(target) != row['sha256']):
            raise ValueError('Invalid shared asset: '+row['path'])
    return len(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,help='Check only this manifest; default checks inputs/weights and result assets')
    parser.add_argument('--include-external', action='store_true',
                        help='Also require the separately obtained MSMS and ERNIE preprocessing assets')
    args=parser.parse_args()
    manifests=[args.manifest] if args.manifest else [ROOT/'docs/release_asset_checksums.csv']
    if not args.manifest and (ROOT/'docs/result_asset_checksums.csv').is_file():
        manifests.append(ROOT/'docs/result_asset_checksums.csv')
    rows=[]
    for manifest in manifests:
        with manifest.open() as handle:
            rows.extend(csv.DictReader(handle))
    external = json.loads((ROOT/'docs/external_assets.json').read_text())['assets']
    external_by_path = {item['path']:item for item in external.values()}
    optional = external_by_path if not args.manifest and not args.include_external else {}
    if args.include_external:
        seen = {row['path'] for row in rows}
        rows.extend(dict(path=item['path'], bytes=item['bytes'], sha256=item['sha256'])
                    for item in external.values() if item['path'] not in seen)
    checked, skipped = 0, []
    for row in rows:
        path=ROOT/row['path']
        if not path.exists() and row['path'] in optional:
            skipped.append(row['path'])
            continue
        if not path.is_file() or path.stat().st_size!=int(row['bytes']) or sha256_file(path)!=row['sha256']:
            raise SystemExit('Asset mismatch: '+row['path'])
        checked += 1
    aliases = verify_aliases(ROOT)
    print('PASS',checked,'asset files;',aliases,'internal shared-asset links')
    if skipped:
        print('Not bundled (optional preprocessing only):', ', '.join(skipped))
    if not args.include_external:
        print('MSMS / ERNIE preprocessing installation: see docs/EXTERNAL_ASSETS.md; use --include-external to require both.')

if __name__=='__main__':
    main()
