from cardiac.utils.io import read_json

DEFAULTS = dict(
    model='gap_ssm',
    length=32,
    stride=2,
    size=112,
    seed=2026,
    epochs=45,
    learning_rate=1e-4,
    weight_decay=1e-4,
    dropout=0.2,
    width=128,
    state_size=16,
    layers=2,
    encoder='resnet18',
    use_mask=True,
    use_gap=True,
    hold_missing=True,
    corruption=True,
    pretrained=False,
    batch_size=None,
    workers=None,
    accumulation=4,
    grad_clip=5.0,
    patience=10,
    save_epochs=True,
)

VALID_MODELS = {'baseline', 'r3d_18', 'r2plus1d_18', 'mamba', 'gap_ssm'}


def load_config(path):
    supplied = read_json(path)
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration fields: {unknown}')
    c = {**DEFAULTS, **supplied}

    model_key = str(c['model']).lower()
    if model_key not in VALID_MODELS:
        raise ValueError(f"Invalid model '{c['model']}'. Supported models: {sorted(VALID_MODELS)}")

    for key in ('length', 'stride', 'size', 'epochs', 'width', 'state_size', 'layers', 'accumulation', 'patience'):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if not (0 < c['dropout'] < 1):
        raise ValueError('Dropout must be in (0,1) to support MC dropout.')
    if c['learning_rate'] <= 0 or c['weight_decay'] < 0 or c['grad_clip'] <= 0:
        raise ValueError('Invalid optimizer settings.')
    for key in ('batch_size', 'workers'):
        if c[key] is not None and (not isinstance(c[key], int) or c[key] < (0 if key == 'workers' else 1)):
            raise ValueError(f'Invalid {key}.')
    return c
