"""Tables and figures from recorded predictions only. No example research scores."""
from pathlib import Path
import hashlib
import io
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cardiac.evaluation.metrics import metrics, robustness
from cardiac.utils.io import read_json, write_json

def save_plot(fig, path):
    """Finish encoding before atomic replacement and verify the final bytes."""
    path = Path(path)
    buffer = io.BytesIO()
    fig.savefig(buffer, format=path.suffix.lstrip('.'))
    payload = buffer.getvalue()
    temp = path.with_suffix(path.suffix + '.tmp')
    with open(temp, 'wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    if path.read_bytes() != payload:
        raise IOError(f'Figure write did not preserve bytes: {path}')
    if path.suffix == '.png':
        with Image.open(path) as image:
            image.verify()


KEYS = ['model','split','cohort','pattern','handling']
MEASURES = ['mae','mse','rmse','r2','mape','pearson','bias','delta_mae','relative_delta_mae','prediction_shift','mean_mc_std']


def clean_records(df):
    return df.astype(object).where(pd.notna(df), None).to_dict('records')


def summarize(df):
    if df.empty or df.bank_sha256.nunique() != 1 or df.provenance.nunique() != 1:
        raise ValueError('Use nonempty predictions from one bank and one provenance.')
    signatures = [set(map(tuple, g[['split','cohort','pattern','handling','ratio','id']].to_numpy()))
                  for _, g in df.groupby('model')]
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError('Models do not cover identical experiment conditions and videos.')
    identities = KEYS + ['ratio','id']
    if df.duplicated(identities).any():
        raise ValueError('Duplicate prediction keys; do not merge repeated runs under one model name.')
    for _, group in df.groupby(['split','cohort','pattern','handling','ratio']):
        sets = [tuple(zip(g.sort_values('id').id, g.sort_values('id').target)) for _, g in group.groupby('model')]
        if any(s != sets[0] for s in sets):
            raise ValueError('Models were evaluated on different videos or targets.')
    rows = []
    for key, group in df.groupby(KEYS, sort=True):
        complete = group[group.ratio == 0].set_index('id').sort_index()
        if complete.empty:
            raise ValueError('A paired complete condition is required.')
        for ratio, subset in group.groupby('ratio', sort=True):
            subset = subset.set_index('id').sort_index()
            if not subset.index.equals(complete.index) or not np.array_equal(subset.target, complete.target):
                raise ValueError('Each missing condition must use the exact complete cohort and targets.')
            if subset.actual_ratio.nunique() != 1:
                raise ValueError('Mixed actual missing ratios in one condition.')
            values = metrics(subset.target, subset.prediction, complete.prediction)
            values.update(dict(zip(KEYS, key)), ratio=float(ratio), actual_ratio=float(subset.actual_ratio.iloc[0]),
                          mean_mc_std=float(subset.mc_std.mean()),
                          inference_seconds=float(subset.inference_seconds.mean()),
                          cohort_sha256=hashlib.sha256('\n'.join(subset.index).encode()).hexdigest(),
                          provenance=str(subset.provenance.iloc[0]), preliminary=bool(subset.preliminary.any()))
            rows.append(values)
    return pd.DataFrame(rows)


def report(prediction_paths, output):
    frames, efficiency = [], []
    for path in prediction_paths:
        frames.append(pd.read_csv(path))
        meta = Path(path).parent / 'evaluation.json'
        if meta.exists():
            efficiency.append(read_json(meta))
    predictions = pd.concat(frames, ignore_index=True)
    summary = summarize(predictions)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    figs = out / 'figures'; figs.mkdir(exist_ok=True)
    tables = out / 'tables'; tables.mkdir(exist_ok=True)
    summary.to_csv(tables / 'metrics.csv', index=False)
    write_json(tables / 'metrics.json', clean_records(summary))
    predictions.to_csv(tables / 'predictions.csv', index=False)
    complete = summary[summary.ratio == 0].drop_duplicates(['model','split','cohort','handling'])
    complete.to_csv(tables / 'complete_comparison.csv', index=False)
    summary[summary.ratio == .25].to_csv(tables / 'input_handling_at_25pct.csv', index=False)
    summary[(summary.cohort == 'phase') & (summary.ratio == .25)].to_csv(tables / 'phase_position_at_25pct.csv', index=False)
    summary[summary.model.str.startswith(('ablation_', 'control_'))].to_csv(tables / 'ablations_and_controls.csv', index=False)
    uncertainty_rows = []
    auc_rows = []
    for key, group in summary.groupby(KEYS):
        group = group.sort_values('actual_ratio')
        auc_rows.append({**dict(zip(KEYS, key)), **robustness(group.actual_ratio, group.mae),
                         'provenance': group.provenance.iloc[0], 'split_n': int(group.n.iloc[0])})
        corr = np.corrcoef(group.actual_ratio, group.mean_mc_std)[0,1] if group.mean_mc_std.std() > 0 else None
        uncertainty_rows.append({**dict(zip(KEYS, key)), 'ratio_std_correlation': corr,
                                 'std_increase_at_max_ratio': float(group.mean_mc_std.iloc[-1] - group.mean_mc_std.iloc[0]),
                                 'interpretation': 'MC spread, not a calibrated clinical interval'})
    pd.DataFrame(auc_rows).to_csv(tables / 'robustness.csv', index=False)
    write_json(tables / 'robustness.json', auc_rows)
    pd.DataFrame(uncertainty_rows).to_csv(tables / 'uncertainty_trend.csv', index=False)
    if efficiency:
        pd.DataFrame([{k: e.get(k) for k in ['model','split','provenance','parameter_count','training_seconds',
                                            'training_peak_gpu_bytes','inference_seconds_per_video','evaluation_peak_gpu_bytes']}
                      for e in efficiency]).to_csv(tables / 'efficiency.csv', index=False)
    # All metrics get one table per pattern/cohort/handling, with requested ratios as columns.
    long = summary.melt(id_vars=KEYS + ['ratio','actual_ratio','n','provenance'], value_vars=MEASURES,
                        var_name='metric', value_name='value')
    for key, group in long.groupby(['split','cohort','pattern','handling']):
        name = '_'.join(key)
        group.pivot(index=['model','metric','n','provenance'], columns='ratio', values='value').to_csv(tables / f'{name}.csv')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120, 'savefig.dpi': 220})
    captions = []
    suffix = 'SYNTHETIC SOFTWARE TEST - NOT RESEARCH RESULTS' if (predictions.provenance == 'synthetic_software_test').all() else ('PRELIMINARY' if predictions.preliminary.any() else 'Official test split')
    panels = [('mae','MAE (EF pp)'), ('rmse','RMSE (EF pp)'), ('delta_mae','MAE drop (EF pp)'),
              ('relative_delta_mae','Relative MAE drop (fraction)'), ('prediction_shift','Prediction shift (EF pp)'),
              ('bias','Bias: prediction - target (EF pp)'), ('mean_mc_std','MC dropout spread (EF pp)'), ('pearson','Pearson r')]
    for key, group in summary.groupby(['split','cohort','pattern','handling']):
        name = '_'.join(key)
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout='constrained')
        for ax, (measure, label) in zip(axes.flat, panels):
            for model, values in group.groupby('model'):
                values = values.sort_values('actual_ratio')
                ax.plot(values.actual_ratio * 100, values[measure], marker='o', label=model)
            ax.set(xlabel='Missing frames (%)', ylabel=label)
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'{" / ".join(key)} | {suffix}', fontsize=11)
        save_plot(fig, figs / f'{name}.png'); save_plot(fig, figs / f'{name}.pdf'); plt.close(fig)
        captions.append(f'- `{name}.png`: {key[0]} split, {key[1]} cohort, {key[2]} pattern, {key[3]} handling. X uses actual mask fractions. Bias is predicted minus true EF. MC spread is the sample standard deviation across stochastic passes. {suffix}.')
    for key, group in predictions[predictions.ratio == 0].groupby(['model','split','cohort']):
        group = group.drop_duplicates('id')
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
        axes[0].scatter(group.target, group.prediction, alpha=.5, s=14)
        lo = min(group.target.min(), group.prediction.min()); hi = max(group.target.max(), group.prediction.max())
        axes[0].plot([lo, hi], [lo, hi], '--', color='gray')
        axes[0].set(xlabel='Actual EF (%)', ylabel='Predicted EF (%)')
        axes[1].hist(group.prediction - group.target, bins=min(25, max(3,len(group)//10)))
        axes[1].set(xlabel='Prediction - target (EF pp)', ylabel='Video count')
        fig.suptitle(f'{" / ".join(key)} complete | {suffix}', fontsize=10)
        name = '_'.join(key) + '_complete_errors'
        save_plot(fig, figs / f'{name}.png'); save_plot(fig, figs / f'{name}.pdf'); plt.close(fig)
        captions.append(f'- `{name}.png`: Complete clips only. The diagonal marks exact agreement; the histogram shows signed EF error. {suffix}.')
    (out / 'CAPTIONS.md').write_text('# Figure captions\n\n' + '\n\n'.join(captions) + '\n')
    (out / 'STATUS.md').write_text(f'# Results status\n\n{suffix}.\n\nValues come from the listed prediction files. No published reference scores are inserted.\n')
    write_json(out / 'report_sources.json', {'prediction_files': [str(p) for p in prediction_paths],
                                            'bank_sha256': predictions.bank_sha256.iloc[0], 'status': suffix})
    return {'conditions': len(summary), 'figures': len(captions)}
