#!/usr/bin/env python3
"""Offline manual-license-inventory gate; not a legal certification or push hook."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def contained_path(root, relative):
    """Reject absolute paths, traversal, and links escaping the reviewed tree."""
    path = Path(relative)
    if not relative or path.is_absolute() or '..' in path.parts:
        raise ValueError('Invalid evidence path: ' + str(relative))
    resolved = (root / path).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def audit(root, check_history=False):
    root = Path(root).resolve()
    inventory = json.loads((root / 'docs/third_party_licenses.json').read_text(encoding='utf-8'))
    if inventory.get('schema') != 'ribosurfbind_third_party_review_v1':
        raise ValueError('Unsupported license inventory schema')
    components = inventory['components']
    documents = inventory['license_documents']
    if not components or not documents:
        raise ValueError('An empty inventory cannot establish clearance')
    ids = [item['id'] for item in components]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate component IDs')
    errors, blockers = [], []
    for document in documents:
        expected = document['retained_sha256']
        if not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid evidence digest: ' + document['path'])
        path = contained_path(root, document['path'])
        if not path.is_file():
            if document.get('local_only') is True:
                continue
            errors.append('Missing notice: ' + document['path'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append('Changed notice: ' + document['path'])
    for item in components:
        if type(item['blocking']) is not bool:
            raise ValueError('blocking must be a boolean: ' + item['id'])
        if not all(item.get(key) for key in ('status', 'scope', 'reason', 'action')):
            raise ValueError('Incomplete component review: ' + item['id'])
        if item['blocking']:
            blockers.append(item['id'] + ': ' + item['reason'] + ' Action: ' + item['action'])
    if type(inventory.get('release_ready')) is not bool:
        raise ValueError('release_ready must be a boolean')
    if inventory['release_ready'] != (not blockers):
        raise ValueError('release_ready contradicts component review; review evidence explicitly')
    policy_path = root / 'docs/external_assets.json'
    if policy_path.is_file():
        rules = json.loads(policy_path.read_text())['release_exclusions']
        for relative in rules:
            contained_path(root, relative)
        if (root / '.git').exists():
            tracked = subprocess.run(['git', '-C', str(root), 'ls-files', '-z', '--', *rules],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if tracked.returncode:
                errors.append('Cannot check external-asset tracking policy')
            elif tracked.stdout:
                errors.append('External-download assets are still tracked in the current Git index')
        elif (root / 'RELEASE_SNAPSHOT.json').exists():
            for relative in rules:
                if (root / relative.rstrip('/')).exists():
                    errors.append('Publication snapshot contains external-download asset: ' + relative)
    if check_history:
        for relative in inventory.get('restricted_history_paths', []):
            contained_path(root, relative)
            result = subprocess.run(
                ['git', '-C', str(root), 'rev-list', '-1', '--all', '--', relative],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
            )
            if result.returncode:
                errors.append('Git history could not be checked: ' + relative)
            elif result.stdout.strip():
                blockers.append('Restricted path occurs in Git history: ' + relative
                                + ' (adding an ignore rule or deleting the current file does not remove history)')
    return errors, blockers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-history', action='store_true',
                        help='Also detect recorded restricted paths in local Git history')
    args = parser.parse_args()
    try:
        errors, blockers = audit(ROOT, args.check_history)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('ERROR:', exc)
        return 1
    for message in errors:
        print('ERROR:', message)
    for message in blockers:
        print('BLOCKED:', message)
    if errors:
        return 1
    if blockers:
        print('NOT CLEARED: manual permission/author decisions remain; no files were modified.')
        return 2
    print('Inventory checks passed. This is not legal certification; re-review new assets before release.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
