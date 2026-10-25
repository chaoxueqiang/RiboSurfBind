#!/usr/bin/env python3
"""Predict one real RNA and check identities, labels and reference probabilities."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

EXAMPLE = Path(__file__).resolve().parent
ROOT = EXAMPLE.parents[1]


def compare_predictions(actual, expected, tolerance):
    with Path(actual).open(newline='') as handle:
        observed = list(csv.DictReader(handle))
    with Path(expected).open(newline='') as handle:
        reference = list(csv.DictReader(handle))
    if not reference or len(observed) != len(reference):
        raise ValueError('Prediction row count differs from the reference')
    errors = []
    for left, right in zip(observed, reference):
        for field in ['record_id', 'sequence_position', 'residue_id', 'nucleotide', 'predicted_label']:
            if left[field] != right[field]:
                raise ValueError('Prediction identity/label mismatch: '+field)
        p = float(left['probability'])
        q = float(right['probability'])
        if not math.isfinite(p) or not 0 <= p <= 1 or int(left['predicted_label']) != int(p > .5):
            raise ValueError('Invalid probability or decision threshold')
        errors.append(abs(p-q))
    maximum = max(errors)
    if maximum > tolerance:
        raise ValueError('Reference probability mismatch: max absolute error %.8g > %.8g' % (maximum, tolerance))
    return dict(status='PASS', nucleotides=len(observed), max_absolute_probability_error=maximum,
                absolute_tolerance=tolerance, residue_identities_equal=True, predicted_labels_equal=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=ROOT/'results/new_runs/example_2JUK_A')
    args = parser.parse_args()
    manifest = json.loads((EXAMPLE/'manifest.json').read_text())
    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1048576), b''):
                h.update(block)
        return h.hexdigest()
    for name, digest in manifest['files'].items():
        if sha(EXAMPLE/name) != digest:
            raise ValueError('Example asset changed: '+name)
    checkpoint = ROOT/manifest['checkpoint']
    if sha(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Example requires the published Train60 seed1 checkpoint')
    subprocess.run([sys.executable, str(ROOT/'predict.py'), '--data', str(EXAMPLE/'2JUK_A.pt'),
                    '--checkpoint', str(checkpoint), '--device', args.device,
                    '--output', str(args.output)], cwd=ROOT, check=True)
    report = compare_predictions(args.output/'2JUK_A.csv', EXAMPLE/'expected_predictions.csv',
                                 manifest['probability_absolute_tolerance'])
    report['device'] = args.device
    (args.output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
