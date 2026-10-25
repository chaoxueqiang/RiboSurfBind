#!/usr/bin/env python3
"""Create a fresh tracked-file snapshot without local external assets or Git history.

This does not certify remaining code/data permissions and never deletes files.
Run from a clean, committed source checkout. LFS assets are copied as real
working-tree bytes, not archive pointers. An existing output is never replaced.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def excluded(name, rules):
    return any(name == rule.rstrip('/') or name.startswith(rule.rstrip('/') + '/')
               for rule in rules)


def export_snapshot(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new path: ' + str(output))
    if output == root or root in output.parents:
        raise ValueError('Output must be outside the source checkout')
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(root), *arguments])
    if Path(git('rev-parse', '--show-toplevel').decode().strip()).resolve() != root:
        raise ValueError('Source must be the repository root')
    if git('status', '--porcelain', '--untracked-files=normal').strip():
        raise ValueError('Commit source changes before making a publication snapshot')
    commit = git('rev-parse', 'HEAD').decode().strip()
    settings = json.loads((root/'docs/external_assets.json').read_text())
    rules = settings['release_exclusions']
    names = [os.fsdecode(value) for value in git('ls-files', '-z').split(b'\0') if value]
    copied = [name for name in names if not excluded(name, rules)]
    # Validate every target before creating a partial output directory.
    for name in copied:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
            raise ValueError('Unsafe tracked path: ' + name)
        source = root/relative
        source.resolve().relative_to(root)
        if source.is_symlink() and os.path.isabs(os.readlink(source)):
            raise ValueError('Non-portable absolute link: ' + name)
        if not source.is_file():
            raise ValueError('Missing tracked file: ' + name)
    output.mkdir(parents=True)
    for name in copied:
        destination = output/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, destination, follow_symlinks=False)
    subprocess.run([sys.executable, '-B', str(output/'scripts/verify_assets.py')],
                   cwd=str(output), check=True)
    if git('rev-parse', 'HEAD').decode().strip() != commit:
        raise ValueError('Source commit changed during export; do not use this partial snapshot')
    for name in rules:
        if (output/name.rstrip('/')).exists():
            raise ValueError('Excluded asset appeared in snapshot: ' + name)
    if (output/'.git').exists():
        raise ValueError('Snapshot must not contain Git history')
    review = json.loads((root/'docs/third_party_licenses.json').read_text())
    report = dict(schema='ribosurfbind_public_snapshot_v1', source_commit=commit,
                  copied_files=len(copied), excluded_paths=rules, git_history_copied=False,
                  release_ready=review['release_ready'],
                  remaining_review_items=[item['id'] for item in review['components'] if item['blocking']])
    (output/'RELEASE_SNAPSHOT.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = export_snapshot(ROOT, args.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print('Export failed:', error, file=sys.stderr)
        print('No existing output was overwritten; inspect any new partial output.', file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
