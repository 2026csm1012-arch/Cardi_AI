from cardiac.models.baseline import VideoBaseline
from cardiac.models.gap_ssm import GapSSM

SUPPORTED_MODELS = {'baseline', 'r3d_18', 'r2plus1d_18', 'mamba', 'gap_ssm'}


def build_model(config):
    """Factory to instantiate models: Baseline (R3D-18 / R(2+1)D-18) or Mamba (GapSSM)."""
    raw_model = config.get('model', 'gap_ssm')
    model_name = str(raw_model).lower()

    if model_name in ('baseline', 'r3d_18', 'r2plus1d_18'):
        arch = 'r3d_18' if model_name == 'baseline' else model_name
        return VideoBaseline(
            architecture=arch,
            dropout=config.get('dropout', 0.2),
            pretrained=config.get('pretrained', False)
        )
    elif model_name in ('mamba', 'gap_ssm'):
        ssm_keys = ('width', 'state_size', 'layers', 'dropout', 'encoder',
                    'use_mask', 'use_gap', 'hold_missing')
        defaults = {
            'width': 128, 'state_size': 16, 'layers': 2, 'dropout': 0.2,
            'encoder': 'resnet18', 'use_mask': True, 'use_gap': True, 'hold_missing': True
        }
        kwargs = {k: config.get(k, defaults[k]) for k in ssm_keys}
        return GapSSM(**kwargs)
    else:
        raise ValueError(
            f"Unsupported model name '{raw_model}'. Choose from: {sorted(SUPPORTED_MODELS)}"
        )
