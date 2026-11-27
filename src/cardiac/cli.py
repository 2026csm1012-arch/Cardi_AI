import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description='Cardiac EF Missing-Frame Prediction Pipeline (Baseline & Mamba)')
    sub = parser.add_subparsers(dest='command', required=True)

    # Manifest subcommand
    p = sub.add_parser('manifest', help='Decode and audit dataset videos')
    p.add_argument('--data', required=True, help='Path to dataset directory (containing FileList.csv and Videos/)')
    p.add_argument('--out', required=True, help='Path to write manifest.json')
    p.add_argument('--allow-subset', action='store_true', help='Allow partial or preliminary dataset subsets')
    p.add_argument('--synthetic', action='store_true', help='Flag indicating synthetic test fixture')

    # Mask bank subcommand
    p = sub.add_parser('bank', help='Freeze clips and missing-frame masks for reproducibility')
    p.add_argument('--manifest', required=True, help='Path to manifest.json')
    p.add_argument('--out', required=True, help='Path to write bank.json')
    p.add_argument('--length', type=int, default=32, help='Clip frame count (default: 32)')
    p.add_argument('--stride', type=int, default=2, help='Sampling stride (default: 2)')
    p.add_argument('--seed', type=int, default=2026, help='Deterministic random seed')
    p.add_argument('--split', choices=['VAL', 'TEST'], default='TEST', help='Dataset split to bank')

    # Train subcommand
    p = sub.add_parser('train', help='Train Baseline or Mamba model with epoch checkpointing')
    p.add_argument('--data', required=True, help='Path to dataset directory')
    p.add_argument('--manifest', required=True, help='Path to manifest.json')
    p.add_argument('--config', required=True, help='Path to configuration JSON (e.g., configs/baseline.json or configs/mamba.json)')
    p.add_argument('--out', required=True, help='Output directory to save checkpoints and training metrics')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Compute device')
    p.add_argument('--limit', type=int, help='Limit number of samples for quick tests')
    p.add_argument('--max-steps', type=int, help='Limit number of steps per epoch')
    p.add_argument('--synthetic', action='store_true', help='Flag if training on synthetic fixture')
    p.add_argument('--no-save-epochs', action='store_true', help='Disable storing per-epoch checkpoint files')
    p.add_argument('--resume', help='Checkpoint file to resume training from (or "last")')
    p.add_argument('--strict-verify', action='store_true', help='Perform full SHA256 hashing check on all video files')

    # Evaluate subcommand
    p = sub.add_parser('evaluate', help='Evaluate model under missing-frame conditions')
    p.add_argument('--data', required=True, help='Path to dataset directory')
    p.add_argument('--manifest', required=True, help='Path to manifest.json')
    p.add_argument('--bank', required=True, help='Path to bank.json')
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pt)')
    p.add_argument('--out', required=True, help='Directory to save predictions.csv and evaluation.json')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Compute device')
    p.add_argument('--handlings', nargs='+', default=['zero'], choices=['zero', 'removal', 'repeat', 'interpolate'],
                   help='Missing frame imputation / handling strategies')
    p.add_argument('--mc-passes', type=int, default=20, help='Number of Monte Carlo dropout passes for uncertainty')
    p.add_argument('--model-name', help='Custom model name for reporting')
    p.add_argument('--limit', type=int, help='Limit evaluation samples')
    p.add_argument('--synthetic', action='store_true', help='Flag if evaluating on synthetic fixture')
    p.add_argument('--allow-hash-mismatch', action='store_true', help='Allow evaluating checkpoint even if manifest hash differed')
    p.add_argument('--strict-verify', action='store_true', help='Perform full SHA256 hashing check on all video files')

    # Report subcommand
    p = sub.add_parser('report', help='Generate comparative performance tables and figures')
    p.add_argument('--predictions', nargs='+', required=True, help='One or more predictions.csv files')
    p.add_argument('--out', required=True, help='Output directory for tables and plots')

    args = parser.parse_args()

    if args.command == 'manifest':
        from cardiac.data.manifest import build_manifest
        result = build_manifest(args.data, args.out, args.allow_subset, args.synthetic)
        result = {'videos': len(result['records']), 'splits': result['split_counts'], 'errors': len(result['errors'])}
    elif args.command == 'bank':
        from cardiac.experiments.bank import build_bank
        result = build_bank(args.manifest, args.out, args.length, args.stride, args.seed, args.split)
        result = {'entries': len(result['entries']), 'phase_exclusions': result['phase_exclusion_counts']}
    elif args.command == 'train':
        from cardiac.training.engine import train
        result = train(
            root=args.data,
            manifest_path=args.manifest,
            config_path=args.config,
            output=args.out,
            device=args.device,
            limit=args.limit,
            synthetic=args.synthetic,
            max_steps=args.max_steps,
            save_epochs=not args.no_save_epochs,
            resume=args.resume,
            quick_verify=not args.strict_verify
        )
    elif args.command == 'evaluate':
        from cardiac.evaluation.runner import evaluate
        result = evaluate(
            root=args.data,
            manifest_path=args.manifest,
            bank_path=args.bank,
            checkpoint_path=args.checkpoint,
            output=args.out,
            device=args.device,
            handlings=args.handlings,
            mc_passes=args.mc_passes,
            synthetic=args.synthetic,
            model_name=args.model_name,
            limit=args.limit,
            quick_verify=not args.strict_verify,
            allow_hash_mismatch=args.allow_hash_mismatch
        )
    else:
        from cardiac.evaluation.reporting import report
        result = report(args.predictions, args.out)

    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
