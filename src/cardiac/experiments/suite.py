"""Run selected configurations sequentially; never launch redundant GPU jobs."""
import argparse
from pathlib import Path
from cardiac.training.engine import train
from cardiac.evaluation.runner import evaluate
from cardiac.evaluation.reporting import report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--bank', required=True); p.add_argument('--out', default='runs/study')
    p.add_argument('--configs', nargs='+', default=['configs/baseline.json','configs/mamba.json'])
    p.add_argument('--device', default='auto'); p.add_argument('--mc-passes', type=int, default=20)
    p.add_argument('--limit', type=int); p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--handlings', nargs='+', default=['zero','removal','repeat','interpolate'])
    a = p.parse_args(); predictions = []
    for config in a.configs:
        name = Path(config).stem
        run = Path(a.out) / name
        if not a.evaluate_only:
            train(a.data, a.manifest, config, run, device=a.device, limit=a.limit)
        evaluate(a.data, a.manifest, a.bank, run / 'best.pt', run / 'evaluation', device=a.device,
                 handlings=a.handlings, mc_passes=a.mc_passes, model_name=name, limit=a.limit)
        predictions.append(str(run / 'evaluation/predictions.csv'))
    report(predictions, Path(a.out) / 'comparison')


if __name__ == '__main__': main()
