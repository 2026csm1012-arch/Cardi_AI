import numpy as np
import pytest
import torch
from torch import nn
from cardiac.models.gap_ssm import GapSSM, SelectiveBlock
from cardiac.models.baseline import VideoBaseline
from cardiac.evaluation.runner import mc_predict

torch.set_num_threads(2)


def small_model(**kwargs):
    return GapSSM(width=16,state_size=4,layers=1,encoder='tiny',dropout=.3,**kwargs)


def test_missing_pixels_cannot_affect_proposed():
    torch.manual_seed(1)
    model=small_model().eval()
    x=torch.randn(2,6,3,16,16); m=torch.tensor([[1,0,1,0,1,1]]*2,dtype=torch.bool)
    t=torch.arange(6).float().repeat(2,1)
    altered=x.clone(); altered[~m]=10000
    with torch.no_grad():
        assert torch.allclose(model(x,m,t),model(altered,m,t),atol=1e-7)


def test_gap_effect_and_ablation():
    torch.manual_seed(2)
    model=small_model().eval(); x=torch.randn(1,4,3,16,16)
    m=torch.ones(1,4,dtype=torch.bool); a=torch.arange(4).float()[None]; b=a.clone();b[:,2:]+=8
    with torch.no_grad(): assert not torch.allclose(model(x,m,a),model(x,m,b),atol=1e-7)
    model.blocks[0].use_gap=False
    with torch.no_grad(): assert torch.allclose(model(x,m,a),model(x,m,b),atol=1e-7)


def test_backward_and_all_missing():
    model=small_model(); x=torch.randn(2,6,3,16,16); m=torch.ones(2,6,dtype=torch.bool)
    t=torch.arange(6).float().repeat(2,1)
    loss=model(x,m,t).square().mean();loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.blocks[0].a_log.grad is not None
    with pytest.raises(ValueError):model(x,~m,t)


def test_exact_zoh_one_step():
    # Isolate a one-channel selective recurrence against a hand-computed scalar system.
    block=SelectiveBlock(width=1,state_size=1,dropout=0.).eval()
    with torch.no_grad():
        for p in block.parameters():p.zero_()
        block.in_proj.bias[:] = torch.tensor([1.,1.])
        block.conv.weight[0,0,2]=1.
        block.parameters_proj.bias[:] = torch.tensor([0.,1.,1.])
        block.out_proj.weight.fill_(1.)
    x=torch.zeros(1,1,1);m=torch.ones(1,1,dtype=torch.bool);t=torch.zeros(1,1)
    result=block(x,m,t).item()
    u=torch.nn.functional.silu(torch.tensor(1.)).item()
    expected=.5*u*u # A=-1, delta=log(2), B=C=1, input and output gates=SiLU(1)
    assert result==pytest.approx(expected,abs=1e-6)


def test_dropout_leaves_bn_unchanged():
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.bn=nn.BatchNorm1d(8);self.drop=nn.Dropout(.5)
        def forward(self,x,m,t):return self.drop(self.bn(x)).mean(-1)
    model=Net().eval();before=model.bn.running_mean.clone()
    _,std=mc_predict(model,torch.ones(2,8),None,None,12)
    assert (std>0).all() and torch.equal(before,model.bn.running_mean)
    assert not model.training and not model.drop.training


@pytest.mark.parametrize('architecture',['r3d_18','r2plus1d_18'])
def test_baseline_forward(architecture):
    model=VideoBaseline(architecture).eval()
    with torch.no_grad():p=model(torch.randn(1,8,3,32,32))
    assert p.shape==(1,) and torch.isfinite(p).all()


def test_leading_gap_survives_removal_timestamps():
    torch.manual_seed(3)
    model=small_model().eval();x=torch.randn(1,4,3,16,16);m=torch.ones(1,4,dtype=torch.bool)
    t=torch.arange(4).float()[None]
    with torch.no_grad():assert not torch.allclose(model(x,m,t),model(x,m,t+5),atol=1e-7)


def test_build_model_aliases():
    from cardiac.models.factory import build_model
    baseline = build_model({'model': 'baseline', 'dropout': 0.2})
    assert isinstance(baseline, VideoBaseline)
    mamba = build_model({'model': 'mamba', 'width': 16, 'state_size': 4, 'layers': 1, 'encoder': 'tiny'})
    assert isinstance(mamba, GapSSM)

