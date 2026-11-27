"""Validate the supplied release. Never download or manufacture research labels."""
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from cardiac.utils.io import digest, write_json


def filename(value):
    value = str(value)
    if '/' in value or '\\' in value or value in ('.', '..'):
        raise ValueError('FileName must be a basename.')
    return value if value.lower().endswith('.avi') else value + '.avi'


def trace_positions(path):
    if not path.exists():
        raise FileNotFoundError(f'[ERROR] Missing tracing metadata: {path}')
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to read VolumeTracings.csv: {e}") from e

    required = {'FileName', 'Frame', 'X1', 'Y1', 'X2', 'Y2'}
    if not required.issubset(df.columns):
        raise ValueError(f'Tracing columns required: {sorted(required)}')
    result = {}
    for name, video in df.groupby('FileName', sort=True):
        areas = []
        for frame, rows in video.groupby('Frame', sort=True):
            if not float(frame).is_integer() or not np.isfinite(rows[['X1','Y1','X2','Y2']].values).all():
                continue
            # Official tracing polygon convention omits the first long-axis line.
            p = rows[['X1','Y1','X2','Y2']].to_numpy(float)[1:]
            if len(p) < 2:
                continue
            x = np.r_[p[:, 0], p[::-1, 2]]
            y = np.r_[p[:, 1], p[::-1, 3]]
            area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
            areas.append((float(area), int(frame)))
        areas.sort()
        if len(areas) == 2 and 0 < areas[0][0] < areas[1][0]:
            result[filename(name)] = {'ed': areas[-1][1], 'es': areas[0][1]}
    return result


def build_manifest(root, output, allow_subset=False, synthetic=False):
    """Build and validate a manifest of all dataset videos, labels, and traces."""
    root = Path(root)
    print(f"[CHECKPOINT] Auditing dataset at {root} (Synthetic: {synthetic}, Allow Subset: {allow_subset})...")

    if synthetic != (root / 'SYNTHETIC_FIXTURE.json').exists():
        raise ValueError('Synthetic provenance mismatch. Use --synthetic only for generated fixtures.')

    label_path = root / 'FileList.csv'
    if not label_path.exists():
        raise FileNotFoundError(f'[ERROR] Place official FileList.csv and Videos/ under {root}.')

    try:
        labels = pd.read_csv(label_path)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed reading FileList.csv at {label_path}: {e}") from e

    if not {'FileName','EF','Split'}.issubset(labels.columns):
        raise ValueError('FileList.csv requires FileName, EF, and Split columns.')

    labels['FileName'] = labels.FileName.map(filename)
    labels['Split'] = labels.Split.astype(str).str.upper()

    if labels.FileName.duplicated().any():
        raise ValueError('Duplicate video IDs found in FileList.csv. Possible split leakage.')
    if not set(labels.Split).issubset({'TRAIN','VAL','TEST'}):
        raise ValueError('Unknown split values. Expected TRAIN, VAL, and TEST.')
    if 'PatientID' in labels and labels.groupby('PatientID').Split.nunique().max() > 1:
        raise ValueError('Patient ID crosses splits.')

    labels['EF'] = pd.to_numeric(labels.EF, errors='coerce')
    if not np.isfinite(labels.EF).all() or not labels.EF.between(0, 100).all():
        raise ValueError('EF must contain finite percentage values in [0,100].')

    counts = Counter(labels.Split)
    if not all(counts[s] for s in ('TRAIN','VAL','TEST')):
        raise ValueError(f'At least one row in each official split is required. Current counts: {dict(counts)}')

    if not allow_subset and (len(labels) != 10030 or counts['TRAIN'] != 7465):
        raise ValueError(
            f'Release count differs from official 10,030 total / 7,465 TRAIN (found {len(labels)} total, {counts["TRAIN"]} train). '
            'Use --allow-subset if you are using a preliminary subset of the dataset.'
        )

    tracing_path = root / 'VolumeTracings.csv'
    traces = trace_positions(tracing_path) if tracing_path.exists() else {}

    records, errors = [], []
    total = len(labels)
    print(f"[CHECKPOINT] Auditing and decoding {total} video files from {root / 'Videos'}...")

    for idx, row in enumerate(labels.itertuples(index=False)):
        if (idx + 1) % 500 == 0 or idx == total - 1:
            print(f"[DEBUG] Decoded {idx + 1}/{total} videos ({len(records)} valid, {len(errors)} errors)...", flush=True)

        path = root / 'Videos' / row.FileName
        if not path.exists():
            errors.append({'id': row.FileName, 'error': f'file_not_found: {path}'})
            continue

        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                errors.append({'id': row.FileName, 'error': 'cannot_open_videocapture'})
                continue

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            n, shape = 0, None

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if shape is not None and shape != frame.shape[:2]:
                    errors.append({'id': row.FileName, 'error': 'variable_resolution'})
                shape = frame.shape[:2]
                n += 1
            cap.release()

            if n == 0 or not np.isfinite(fps) or fps <= 0:
                errors.append({'id': row.FileName, 'error': f'corrupt_video decoded={n} declared={declared} fps={fps}'})
                continue

            # Check if discrepancy is larger than 2 frames
            if declared > 0 and abs(n - declared) > 2:
                errors.append({'id': row.FileName, 'error': f'frame_count_mismatch decoded={n} declared={declared}'})
                continue

            records.append({
                'id': row.FileName,
                'path': 'Videos/' + row.FileName,
                'split': row.Split,
                'ef': float(row.EF),
                'frames': n,
                'fps': fps,
                'height': shape[0],
                'width': shape[1],
                'sha256': digest(path),
                'trace': traces.get(row.FileName)
            })
        except Exception as exc:
            errors.append({'id': row.FileName, 'error': f'exception: {str(exc)}'})

    source_hashes = {'FileList.csv': digest(label_path)}
    if tracing_path.exists():
        source_hashes['VolumeTracings.csv'] = digest(tracing_path)

    manifest = {
        'schema': 1,
        'provenance': 'synthetic_software_test' if synthetic else 'user_supplied_echonet',
        'subset': allow_subset,
        'split_counts': dict(counts),
        'records': records,
        'source_hashes': source_hashes,
        'errors': errors,
        'patient_separation': 'PatientID audited if supplied; otherwise official one-video-per-patient release assumption'
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, manifest)

    if errors:
        raise ValueError(f'{len(errors)} data errors encountered during audit. Inspect {output_path}.')

    print(f"[CHECKPOINT] Manifest successfully written to {output_path} ({len(records)} valid videos).")
    return manifest
