import numpy as np
import pytest
from cardiac.data.missing import make_mask, handle_missing, RATIOS, PATTERNS, HANDLINGS
from cardiac.data.sampling import clip_indices, phase_indices
from cardiac.data.dataset import normalize


@pytest.mark.parametrize('ratio', RATIOS)
@pytest.mark.parametrize('pattern', PATTERNS)
def test_masks(ratio, pattern):
    phase = {'ed':8, 'es':20}
    mask = make_mask(32, ratio, pattern, 44, phase)
    assert (~mask).sum() == int(np.floor(32*ratio+.5))
    assert np.array_equal(mask, make_mask(32, ratio, pattern, 44, phase))
    if pattern != 'random' and ratio:
        assert np.all(np.diff(np.flatnonzero(~mask)) == 1)
    if pattern in ['ed','es'] and ratio:
        assert not mask[phase[pattern]]


def test_positions_and_parity():
    indices, phase = phase_indices(100, 20, 40)
    assert indices[phase['ed']] == 20 and indices[phase['es']] == 40
    assert np.all(np.diff(indices) == 2)
    with pytest.raises(ValueError, match='parity'): phase_indices(100,20,41)
    with pytest.raises(ValueError): phase_indices(100,40,20)
    with pytest.raises(ValueError): phase_indices(100,0,80)
    with pytest.raises(ValueError): phase_indices(30,10,28)


def test_clip_padding():
    idx = clip_indices(4,8,2)
    assert np.array_equal(idx,[0,2,3,3,3,3,3,3])
    assert np.array_equal(clip_indices(100),clip_indices(100))


def test_handling_edges():
    x = np.array([1,2,3,4,5], dtype=float)[:,None,None,None]
    mask = np.array([0,1,0,1,0],bool)
    times = np.array([0,2,4,6,8],float)
    expected = {'zero':[0,2,0,4,0], 'repeat':[0,2,2,4,4],
                'interpolate':[2,2,3,4,4], 'removal':[2,4]}
    for method in HANDLINGS:
        y,m,t = handle_missing(x,mask,times,method)
        assert np.allclose(y.flatten(),expected[method])
        assert np.array_equal(t, times[mask] if method=='removal' else times)
        assert np.array_equal(m, np.ones(2,bool) if method=='removal' else mask)
    assert np.array_equal(x.flatten(),[1,2,3,4,5])
    with pytest.raises(ValueError): handle_missing(x,np.zeros(5,bool),times,'zero')
    assert normalize(np.zeros((1,1,2,2),np.float32)).min() == -1


def test_repeat_is_causal_and_interpolation_irregular():
    x = np.array([1,100,7],dtype=float)[:,None,None,None]
    m = np.array([1,0,1],bool); times=np.array([0,1,6])
    y,_,_ = handle_missing(x,m,times,'interpolate')
    assert y[1].item() == 2
    y,_,_ = handle_missing(x,m,times,'repeat')
    assert y[1].item() == 1
