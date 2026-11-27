"""Freeze test inputs once. Training never reads this bank."""
from collections import Counter
import numpy as np
from cardiac.data.sampling import clip_indices, phase_indices
from cardiac.data.missing import make_mask, RATIOS
from cardiac.utils.io import stable_seed, digest, read_json, write_json


def build_bank(manifest_path, output, length=32, stride=2, seed=2026, split='TEST'):
    manifest = read_json(manifest_path)
    if manifest['errors']:
        raise ValueError('Cannot use a manifest with data errors.')
    entries, excluded = [], []
    for row in manifest['records']:
        if row['split'] != split:
            continue
        main = clip_indices(row['frames'], length, stride)
        cohorts = [('main', main, None, ('random','block'))]
        try:
            if row['trace'] is None:
                raise ValueError('missing_or_invalid_traces')
            idx, phase = phase_indices(row['frames'], **row['trace'], length=length, stride=stride)
            cohorts.append(('phase', idx, phase, ('ed','es','systole','diastole')))
        except ValueError as error:
            excluded.append({'id': row['id'], 'reason': str(error)})
        for cohort, indices, phase, patterns in cohorts:
            masks = {}
            for pattern in patterns:
                masks[pattern] = {}
                for ratio in RATIOS:
                    mask = make_mask(length, ratio, pattern, stable_seed(seed, row['id'], cohort, pattern, ratio), phase)
                    masks[pattern][str(ratio)] = mask.astype(int).tolist()
            entries.append({'id': row['id'], 'cohort': cohort, 'indices': indices.tolist(),
                            'phase': phase, 'repeated_source_frames': length - len(np.unique(indices)), 'masks': masks})
    if not entries:
        raise ValueError(f'No {split} records.')
    result = {'schema': 1, 'manifest_sha256': digest(manifest_path), 'seed': seed, 'split': split,
              'length': length, 'stride': stride, 'ratios': list(RATIOS), 'entries': entries,
              'phase_exclusions': excluded, 'phase_exclusion_counts': dict(Counter(x['reason'] for x in excluded))}
    write_json(output, result)
    return result
