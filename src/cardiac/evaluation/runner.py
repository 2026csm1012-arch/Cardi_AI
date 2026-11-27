import csv
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from cardiac.data.dataset import load_video, normalize, verify_files
from cardiac.data.missing import handle_missing, HANDLINGS
from cardiac.models.factory import build_model
from cardiac.utils.runtime import setup, sync
from cardiac.utils.io import digest, read_json, write_json, stable_seed


def mc_predict(model, x, mask, times, passes, use_amp=False):
    """Only explicit Dropout modules enter training mode. Restore all flags."""
    if passes < 2:
        raise ValueError('At least two MC dropout passes are required.')
    states = {m: m.training for m in model.modules()}
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        with torch.no_grad():
            with torch.autocast(device_type=x.device.type, enabled=use_amp):
                values = torch.stack([100.0 * model(x, mask, times) for _ in range(passes)])
        return values.mean(0), values.std(0, unbiased=True)
    finally:
        for module, state in states.items():
            module.training = state


def evaluate(root, manifest_path, bank_path, checkpoint_path, output, device='auto',
             handlings=('zero',), mc_passes=20, synthetic=False, model_name=None, limit=None,
             quick_verify=True, allow_hash_mismatch=False):
    """Evaluate model checkpoint under varying missing-frame conditions and input handlings."""
    if not set(handlings).issubset(HANDLINGS) or not handlings:
        raise ValueError(f'Unsupported input handling method. Choose from: {HANDLINGS}')

    print(f"\n[CHECKPOINT] Initializing Evaluation...")
    print(f"[DEBUG] Checkpoint: {checkpoint_path} | Bank: {bank_path} | Output: {output}")

    manifest, bank = read_json(manifest_path), read_json(bank_path)
    if synthetic != (manifest['provenance'] == 'synthetic_software_test'):
        raise ValueError('Evaluation provenance mismatch.')

    manifest_hash = digest(manifest_path)
    if bank['manifest_sha256'] != manifest_hash:
        msg = f"Bank manifest hash ({bank['manifest_sha256'][:8]}) does not match current manifest ({manifest_hash[:8]})."
        if allow_hash_mismatch:
            print(f"[WARNING] {msg} Proceeding because allow_hash_mismatch=True.")
        else:
            raise ValueError(f"{msg} Pass --allow-hash-mismatch to bypass.")

    verify_files(root, manifest, check_all_hashes=not quick_verify)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    c = ckpt['config']

    ckpt_manifest_hash = ckpt.get('metadata', {}).get('manifest_sha256')
    if ckpt_manifest_hash and ckpt_manifest_hash != manifest_hash:
        msg = f"Checkpoint manifest hash ({ckpt_manifest_hash[:8]}) differs from current manifest ({manifest_hash[:8]})."
        if allow_hash_mismatch:
            print(f"[WARNING] {msg} Proceeding because allow_hash_mismatch=True.")
        else:
            raise ValueError(f"{msg} Pass --allow-hash-mismatch to bypass.")

    if (c['length'], c['stride']) != (bank['length'], bank['stride']):
        raise ValueError('Checkpoint and bank preprocessing differ.')

    dev, runtime = setup(c['seed'], device)
    model = build_model({**c, 'pretrained': False}).to(dev)
    model.load_state_dict(ckpt['model'])
    model.eval()

    records = {r['id']: r for r in manifest['records'] if r['split'] == bank['split']}
    entries = bank['entries']
    if limit:
        selected = sorted(records)[:limit]
        entries = [e for e in entries if e['id'] in selected]

    out = Path(output)
    if (out / 'predictions.csv').exists():
        print(f"[WARNING] Overwriting existing predictions in {out}")
    out.mkdir(parents=True, exist_ok=True)

    active_model_name = model_name or c.get('model', 'model')
    param_count = sum(p.numel() for p in model.parameters())

    meta = {
        'model': active_model_name,
        'checkpoint_sha256': digest(checkpoint_path),
        'bank_sha256': digest(bank_path),
        'manifest_sha256': manifest_hash,
        'split': bank['split'],
        'runtime': runtime,
        'provenance': manifest['provenance'],
        'preliminary': bool(limit or ckpt.get('metadata', {}).get('preliminary')),
        'mc_passes': mc_passes,
        'parameter_count': param_count,
        'handlings': list(handlings),
        'phase_exclusions': bank['phase_exclusion_counts'],
        'timing': 'single forward; synchronized; excludes decode, transfer and MC passes'
    }

    run_path = Path(checkpoint_path).parent / 'run.json'
    if run_path.exists():
        try:
            train_meta = read_json(run_path)
            meta['training_seconds'] = train_meta.get('training_seconds')
            meta['training_peak_gpu_bytes'] = train_meta.get('peak_gpu_bytes')
        except Exception:
            pass

    print(f"[CHECKPOINT] Evaluation Configuration:")
    print(f"  - Model: {active_model_name} ({param_count:,} parameters)")
    print(f"  - Split: {bank['split']} ({len(entries)} entries)")
    print(f"  - Handlings: {list(handlings)} | MC Passes: {mc_passes}")
    print(f"  - Device: {dev.type.upper()}\n")

    rows = []
    warmed = set()
    total_entries = len(entries)
    start_eval_time = time.perf_counter()

    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.no_grad():
        for idx, entry in enumerate(entries):
            row = records.get(entry['id'])
            if row is None:
                continue

            try:
                video = load_video(Path(root) / row['path'], row['frames'], c['size'])
            except Exception as e:
                print(f"[WARNING] Skipping video {entry['id']} due to load error: {e}")
                continue

            indices = np.array(entry['indices'])
            frames = video[indices]
            times = (indices - indices[0]).astype(np.float32) / c['stride']

            # Group conditions by sequence length T so each group runs in a single GPU batch
            grouped = {}
            for handling in handlings:
                for pattern, conditions in entry['masks'].items():
                    for ratio, mask_list in conditions.items():
                        original_mask = np.array(mask_list, bool)
                        x, mask, timestamps = handle_missing(frames, original_mask, times, handling)
                        t_len = x.shape[0]
                        grouped.setdefault(t_len, []).append({
                            'x': normalize(x),
                            'mask': mask,
                            'timestamps': timestamps,
                            'handling': handling,
                            'pattern': pattern,
                            'ratio': float(ratio),
                            'actual_ratio': float(1 - original_mask.mean()),
                            'missing_count': int((~original_mask).sum()),
                        })

            for t_len, items in grouped.items():
                bx = torch.from_numpy(np.stack([it['x'] for it in items], axis=0)).to(dev)
                bmask = torch.from_numpy(np.stack([it['mask'] for it in items], axis=0)).to(dev)
                btimes = torch.from_numpy(np.stack([it['timestamps'] for it in items], axis=0)).to(dev)

                shape = tuple(bx.shape)
                if shape not in warmed:
                    with torch.autocast(device_type=dev.type, enabled=runtime['amp']):
                        model(bx[:1], bmask[:1], btimes[:1])
                    sync(dev)
                    warmed.add(shape)

                sync(dev)
                start = time.perf_counter()
                with torch.autocast(device_type=dev.type, enabled=runtime['amp']):
                    preds = (100.0 * model(bx, bmask, btimes)).float().cpu().numpy()
                sync(dev)
                per_item_elapsed = (time.perf_counter() - start) / len(items)

                torch.manual_seed(
                    stable_seed(c['seed'], entry['id'], entry['cohort'], t_len, 'mc')
                )
                bmean, bstd = mc_predict(model, bx, bmask, btimes, mc_passes, use_amp=runtime['amp'])
                means = bmean.float().cpu().numpy()
                stds = bstd.float().cpu().numpy()

                for i, it in enumerate(items):
                    rows.append({
                        'model': meta['model'],
                        'split': bank['split'],
                        'cohort': entry['cohort'],
                        'id': row['id'],
                        'pattern': it['pattern'],
                        'handling': it['handling'],
                        'ratio': it['ratio'],
                        'actual_ratio': it['actual_ratio'],
                        'missing_count': it['missing_count'],
                        'target': row['ef'],
                        'prediction': float(preds[i]),
                        'mc_mean': float(means[i]),
                        'mc_std': float(stds[i]),
                        'inference_seconds': per_item_elapsed,
                        'bank_sha256': meta['bank_sha256'],
                        'provenance': meta['provenance'],
                        'preliminary': meta['preliminary']
                    })

            # Periodic progress log (first video, every 100 videos, or at the end)
            if (idx + 1) == 1 or (idx + 1) % 100 == 0 or idx == total_entries - 1:
                elapsed_so_far = time.perf_counter() - start_eval_time
                speed = (idx + 1) / max(elapsed_so_far, 0.001)
                eta_sec = (total_entries - (idx + 1)) / max(speed, 0.001)
                print(f"  [DEBUG] Evaluated {idx + 1}/{total_entries} videos ({(idx + 1)/total_entries*100:.1f}%) | {len(rows)} predictions | Speed: {speed:.1f} vid/s | ETA: {int(eta_sec//60)}m {int(eta_sec%60):02d}s", flush=True)

    if not rows:
        raise ValueError('[ERROR] No eligible evaluation rows were generated.')

    pred_file = out / 'predictions.csv'
    with open(pred_file, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_eval_seconds = time.perf_counter() - start_eval_time
    meta.update(
        inference_seconds_per_video=float(np.mean([r['inference_seconds'] for r in rows])),
        evaluation_peak_gpu_bytes=torch.cuda.max_memory_allocated(dev) if dev.type == 'cuda' else None,
        predictions=len(rows),
        total_eval_seconds=total_eval_seconds
    )
    write_json(out / 'evaluation.json', meta)

    print(f"\n[CHECKPOINT] Evaluation Complete in {total_eval_seconds:.1f}s!")
    print(f"[CHECKPOINT] Saved {len(rows)} predictions to {pred_file}")
    return meta
