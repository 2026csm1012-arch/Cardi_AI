"""End-to-end CPU integration with clearly marked synthetic AVI fixtures."""
import argparse
import os
from pathlib import Path
import platform
import sys
import time

# Add scripts and src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / 'src'))

from make_fixture import create
from cardiac.data.manifest import build_manifest
from cardiac.experiments.bank import build_bank
from cardiac.training.engine import train
from cardiac.evaluation.runner import evaluate
from cardiac.evaluation.reporting import report
from cardiac.utils.io import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/smoke_test')
    p.add_argument('--full-handling', action='store_true')
    a = p.parse_args()

    out = Path(a.out)
    started = time.perf_counter()
    print(f"\n[CHECKPOINT] Starting Smoke Test in {out}...")

    data = create(out / 'fixture')
    manifest = out / 'manifest.json'
    build_manifest(data, manifest, allow_subset=True, synthetic=True)

    bank = out / 'bank.json'
    build_bank(manifest, bank, length=8, stride=2)

    predictions = []
    runs = []

    for name in ['baseline', 'mamba']:
        config = {
            'model': 'baseline' if name == 'baseline' else 'mamba',
            'corruption': (name == 'mamba'),
            'length': 8,
            'stride': 2,
            'size': 32,
            'epochs': 1,
            'width': 16,
            'state_size': 4,
            'layers': 1,
            'encoder': 'tiny',
            'batch_size': 1,
            'workers': 0,
            'accumulation': 2,
            'patience': 1,
            'save_epochs': True
        }
        path = out / f'{name}_config.json'
        write_json(path, config)

        run_dir = out / name
        runs.append(train(
            root=data,
            manifest_path=manifest,
            config_path=path,
            output=run_dir,
            device='cpu',
            synthetic=True,
            max_steps=2,
            save_epochs=True
        ))

        evaluate(
            root=data,
            manifest_path=manifest,
            bank_path=bank,
            checkpoint_path=run_dir / 'best.pt',
            output=run_dir / 'evaluation',
            device='cpu',
            synthetic=True,
            handlings=['zero', 'removal', 'repeat', 'interpolate'] if a.full_handling else ['zero'],
            mc_passes=3,
            model_name=name
        )
        predictions.append(run_dir / 'evaluation/predictions.csv')

    summary = report(predictions, out / 'comparison')
    write_json(out / 'SMOKE_RESULT.json', {
        'status': 'passed',
        'provenance': 'synthetic_software_test',
        'research_results': False,
        'seconds': time.perf_counter() - started,
        'python': platform.python_version(),
        'training_runs': runs,
        'report': summary
    })
    print('\n[CHECKPOINT] ============================================')
    print('SMOKE TEST PASSED! Synthetic pipeline verified successfully.')
    print('Epoch checkpoints, Baseline & Mamba models, evaluation, and reporting all working.')
    print('============================================\n')


if __name__ == '__main__':
    main()
