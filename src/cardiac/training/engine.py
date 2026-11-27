import csv
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from cardiac.data.dataset import EchoDataset, verify_files
from cardiac.models.factory import build_model
from cardiac.training.config import load_config
from cardiac.utils.runtime import setup, sync
from cardiac.utils.io import digest, read_json, write_json


def train(root, manifest_path, config_path, output, device='auto', limit=None, synthetic=False,
          max_steps=None, save_epochs=True, resume=None, quick_verify=True):
    """
    Train either Baseline (R3D-18 / R(2+1)D) or Mamba (GapSSM) model.
    Saves best.pt, last.pt, and per-epoch checkpoints in output/epochs/.
    """
    c = load_config(config_path)
    manifest = read_json(manifest_path)
    is_synthetic = manifest['provenance'] == 'synthetic_software_test'
    if synthetic != is_synthetic:
        raise ValueError('Use --synthetic only with a synthetic manifest.')
    if max_steps and not synthetic and not limit:
        raise ValueError('--max-steps requires --limit for a real-data preliminary run.')

    print(f"\n[CHECKPOINT] Initializing training pipeline...")
    print(f"[DEBUG] Dataset: {root} | Config: {config_path} | Output: {output}")

    verify_files(root, manifest, check_all_hashes=not quick_verify)
    dev, runtime = setup(c['seed'], device)

    out = Path(output)
    epochs_dir = out / 'epochs'
    epochs_dir.mkdir(parents=True, exist_ok=True)

    c['batch_size'] = c['batch_size'] or runtime['batch_size']
    c['workers'] = runtime['workers'] if c['workers'] is None else c['workers']

    train_ds = EchoDataset(root, manifest, 'TRAIN', c, train=True, limit=limit)
    val_ds = EchoDataset(root, manifest, 'VAL', c, train=False, limit=limit)
    generator = torch.Generator().manual_seed(c['seed'])

    loader = DataLoader(
        train_ds,
        batch_size=c['batch_size'],
        shuffle=True,
        num_workers=c['workers'],
        generator=generator,
        pin_memory=(dev.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=c['batch_size'],
        shuffle=False,
        num_workers=c['workers']
    )

    model = build_model(c).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=runtime['amp'])

    start_epoch = 0
    best = float('inf')
    stale = 0
    rows = []

    # Handle resumption if requested
    if resume:
        resume_file = out / 'last.pt' if resume == 'last' else Path(resume)
        if resume_file.exists():
            print(f"[CHECKPOINT] Resuming training from checkpoint: {resume_file}")
            ckpt = torch.load(resume_file, map_location=dev)
            model.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0)
            best = ckpt.get('best_val_mae', ckpt.get('val_mae', float('inf')))
            csv_file = out / 'training.csv'
            if csv_file.exists():
                try:
                    with open(csv_file, 'r', newline='') as stream:
                        rows = list(csv.DictReader(stream))
                except Exception:
                    pass
            print(f"[DEBUG] Resumed at epoch {start_epoch + 1} with previous best Val MAE: {best:.4f}")
        else:
            print(f"[CHECKPOINT] No checkpoint found at {resume_file}. Starting fresh.")

    param_count = sum(p.numel() for p in model.parameters())
    metadata = {
        'config': c,
        'runtime': runtime,
        'manifest_sha256': digest(manifest_path),
        'provenance': manifest['provenance'],
        'preliminary': bool(limit or manifest['subset'] or synthetic),
        'train_n': len(train_ds),
        'val_n': len(val_ds),
        'parameter_count': param_count,
        'selection_metric': 'complete_validation_mae',
        'target_scale': 'EF percentage points / 100 during loss'
    }
    write_json(out / 'run.json', metadata)

    print(f"[CHECKPOINT] Training Environment Ready:")
    print(f"  - Device: {dev.type.upper()} (AMP: {runtime['amp']})")
    print(f"  - Model: {c['model']} ({param_count:,} parameters)")
    print(f"  - Dataset: {len(train_ds)} train videos, {len(val_ds)} val videos")
    print(f"  - Batch Size: {c['batch_size']} (Accumulation: {c['accumulation']}, Effective: {c['batch_size'] * c['accumulation']})")
    print(f"  - Learning Rate: {c['learning_rate']} | Max Epochs: {c['epochs']} | Patience: {c['patience']}")
    print(f"  - Storing Epoch Checkpoints: {save_epochs} -> {epochs_dir}\n")

    started = time.perf_counter()
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)

    try:
        for epoch in range(start_epoch, c['epochs']):
            epoch_start = time.perf_counter()
            train_ds.epoch = epoch
            model.train()

            # BatchNorm handling for tiny synthetic test inputs
            if synthetic and c['size'] < 64:
                for module in model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()

            optimizer.zero_grad(set_to_none=True)
            train_sum, count = 0.0, 0
            steps = min(len(loader), max_steps) if max_steps else len(loader)

            print(f"[CHECKPOINT] Starting Epoch {epoch + 1}/{c['epochs']} ({steps} steps)...", flush=True)

            for step, batch in enumerate(loader):
                if step >= steps:
                    break

                video, mask, times, y = [batch[k].to(dev) for k in ('video', 'mask', 'times', 'target')]
                group_start = (step // c['accumulation']) * c['accumulation']
                group_end = min(group_start + c['accumulation'], steps)
                group_n = min(group_end * c['batch_size'], len(train_ds)) - group_start * c['batch_size']

                try:
                    with torch.autocast(device_type=dev.type, enabled=runtime['amp']):
                        p = model(video, mask, times)
                        loss = torch.mean((p - y / 100.0) ** 2)

                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Non-finite training loss at epoch {epoch + 1}, step {step}.')

                    scaler.scale(loss * len(y) / group_n).backward()

                    if step + 1 == group_end or step + 1 == steps:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'])
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                except torch.cuda.OutOfMemoryError as oom:
                    if dev.type == 'cuda':
                        torch.cuda.empty_cache()
                    print(f"\n[ERROR] CUDA Out of Memory at Epoch {epoch + 1}, Step {step + 1}!")
                    print(f"[TIP] Reduce 'batch_size' in your configuration (e.g. 1) and increase 'accumulation'.")
                    raise oom

                train_sum += float(loss.detach()) * len(y)
                count += len(y)

                # Periodic debug log
                log_interval = max(1, steps // 4)
                if (step + 1) % log_interval == 0 or step + 1 == steps:
                    curr_lr = optimizer.param_groups[0]['lr']
                    print(f"  [DEBUG] Epoch {epoch + 1} [{step + 1}/{steps}] | Batch Loss: {loss.item():.4f} | LR: {curr_lr:.2e}", flush=True)

            # Validation phase
            model.eval()
            total_abs, val_mse, val_n = 0.0, 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    x, mask, times, y = [batch[k].to(dev) for k in ('video', 'mask', 'times', 'target')]
                    with torch.autocast(device_type=dev.type, enabled=runtime['amp']):
                        p = 100.0 * model(x, mask, times)
                    total_abs += float((p - y).abs().sum())
                    val_mse += float(((p - y) ** 2).sum())
                    val_n += len(y)

            mae = total_abs / max(val_n, 1)
            sync(dev)
            epoch_duration = time.perf_counter() - epoch_start
            total_elapsed = time.perf_counter() - started

            row = {
                'epoch': epoch + 1,
                'train_mse_fraction': train_sum / max(count, 1),
                'val_mae_pp': mae,
                'val_mse_pp2': val_mse / max(val_n, 1),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'elapsed_seconds': total_elapsed
            }
            rows.append(row)

            # Save training metrics to CSV
            with open(out / 'training.csv', 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)

            # Base checkpoint data
            checkpoint_data = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': c,
                'metadata': metadata,
                'val_mae': mae,
                'train_mse': row['train_mse_fraction'],
                'best_val_mae': min(best, mae)
            }

            # 1. Save per-epoch checkpoint file
            if save_epochs:
                epoch_path = epochs_dir / f'checkpoint_epoch_{epoch + 1:03d}.pt'
                torch.save(checkpoint_data, epoch_path)

            # 2. Save last.pt (latest checkpoint for easy resumption)
            torch.save(checkpoint_data, out / 'last.tmp.pt')
            (out / 'last.tmp.pt').replace(out / 'last.pt')

            # 3. Check for improvement and save best.pt
            is_best = mae < best
            if is_best:
                best = mae
                stale = 0
                best_ckpt = {
                    'model': model.state_dict(),
                    'config': c,
                    'metadata': metadata,
                    'epoch': epoch + 1,
                    'best_val_mae': best
                }
                torch.save(best_ckpt, out / 'best.tmp.pt')
                (out / 'best.tmp.pt').replace(out / 'best.pt')
                print(f"[CHECKPOINT] *** NEW BEST VAL MAE: {best:.4f} pp! Saved to {out}/best.pt ***", flush=True)
            else:
                stale += 1

            scheduler.step()

            print(
                f"[CHECKPOINT] Epoch {epoch + 1}/{c['epochs']} Completed in {epoch_duration:.1f}s | "
                f"Train MSE: {row['train_mse_fraction']:.6f} | Val MAE: {mae:.4f} pp (Best: {best:.4f}) | Stale: {stale}/{c['patience']}",
                flush=True
            )

            # Early stopping check
            if stale >= c['patience']:
                print(f"[CHECKPOINT] Early stopping triggered after {c['patience']} epochs without improvement.", flush=True)
                break

    except KeyboardInterrupt:
        print(f"\n[WARNING] Training interrupted by user! Saving emergency checkpoint to {out}/interrupted.pt...")
        emergency_ckpt = {
            'epoch': epoch + 1 if 'epoch' in locals() else 0,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': c,
            'metadata': metadata
        }
        torch.save(emergency_ckpt, out / 'interrupted.pt')
        print(f"[CHECKPOINT] Emergency checkpoint saved. You can resume with --resume {out}/interrupted.pt")
        raise

    metadata.update(
        training_seconds=time.perf_counter() - started,
        epochs_completed=len(rows),
        best_val_mae=best,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(dev) if dev.type == 'cuda' else None
    )
    write_json(out / 'run.json', metadata)

    print(f"\n[CHECKPOINT] ========================================================")
    print(f"[CHECKPOINT] Training Complete for '{c['model']}'!")
    print(f"[CHECKPOINT] Total Time: {time.perf_counter() - started:.1f}s | Epochs: {len(rows)} | Best Val MAE: {best:.4f}")
    print(f"[CHECKPOINT] Checkpoints stored in {out}:")
    print(f"  - Best weights: {out}/best.pt")
    print(f"  - Latest weights: {out}/last.pt")
    if save_epochs:
        print(f"  - Epoch checkpoints: {epochs_dir}/")
    print(f"[CHECKPOINT] ========================================================\n")

    return metadata
