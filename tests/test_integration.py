import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cardiac.data.manifest import build_manifest
from cardiac.data.dataset import EchoDataset, verify_files
from cardiac.experiments.bank import build_bank
from cardiac.evaluation.reporting import summarize
from cardiac.training.config import load_config
from cardiac.utils.io import write_json, read_json

spec=importlib.util.spec_from_file_location('fixture',Path(__file__).parents[1]/'scripts/make_fixture.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

@pytest.fixture
def dataset(tmp_path):
    root=fixture.create(tmp_path/'data');manifest=tmp_path/'manifest.json'
    build_manifest(root,manifest,allow_subset=True,synthetic=True)
    return root,manifest

def test_manifest_and_bank(dataset,tmp_path):
    root,path=dataset;m=read_json(path)
    assert m['split_counts']=={'TRAIN':2,'VAL':2,'TEST':2}
    assert all(r['trace']=={'ed':4,'es':8} for r in m['records'])
    bank=build_bank(path,tmp_path/'bank.json',length=8)
    assert len(bank['entries'])==4 and bank['phase_exclusions']==[]
    assert bank==build_bank(path,tmp_path/'same.json',length=8)
    config_path=tmp_path/'config.json';write_json(config_path,{'length':8,'size':32})
    ds=EchoDataset(root,m,'TRAIN',load_config(config_path),train=True)
    a,b=ds[0],ds[0]
    assert np.array_equal(a['video'].numpy(),b['video'].numpy())
    assert a['video'].shape==(8,3,32,32)
    verify_files(root,m)

def test_duplicate_and_provenance(dataset,tmp_path):
    root,path=dataset
    with pytest.raises(ValueError,match='provenance'):build_manifest(root,tmp_path/'x.json',True,False)
    df=pd.read_csv(root/'FileList.csv');pd.concat([df,df.iloc[:1]],ignore_index=True).to_csv(root/'FileList.csv',index=False)
    with pytest.raises(ValueError,match='Duplicate'):build_manifest(root,tmp_path/'x.json',True,True)
    with pytest.raises(ValueError,match='metadata changed'):verify_files(root,read_json(path))

def test_corrupt_video(dataset,tmp_path):
    root,path=dataset;(root/'Videos/fixture_train_0.avi').write_bytes(b'broken')
    with pytest.raises(ValueError,match='data errors'):build_manifest(root,tmp_path/'bad.json',True,True)
    assert read_json(tmp_path/'bad.json')['errors']

def prediction_rows():
    return [{'model':model,'split':'TEST','cohort':'main','pattern':'random','handling':'zero',
             'ratio':ratio,'actual_ratio':ratio,'id':f'v{i}','target':10.+i,
             'prediction':10.+i+ratio,'mc_std':.2,'inference_seconds':.1,
             'bank_sha256':'fixed','provenance':'synthetic_software_test','preliminary':True}
            for model in ['a','b'] for ratio in [0.,.25] for i in range(2)]

def test_paired_reporting():
    df=pd.DataFrame(prediction_rows());result=summarize(df)
    assert len(result)==4
    assert result[result.ratio==.25].prediction_shift.eq(.25).all()
    with pytest.raises(ValueError):summarize(df.iloc[:-1])
    with pytest.raises(ValueError):summarize(pd.concat([df,df.iloc[:1]]))
    df.loc[0,'bank_sha256']='other'
    with pytest.raises(ValueError):summarize(df)

def test_config_errors(tmp_path):
    p=tmp_path/'c.json'
    for config in [{'size':0},{'dropout':0},{'unknown':2},{'workers':-1}]:
        write_json(p,config)
        with pytest.raises(ValueError):load_config(p)


def test_atomic_figure_export(tmp_path):
    import matplotlib.pyplot as plt
    from PIL import Image
    from cardiac.evaluation.reporting import save_plot
    fig,ax=plt.subplots();ax.plot([0,1],[2,3])
    path=tmp_path/'plot.png';save_plot(fig,path);plt.close(fig)
    with Image.open(path) as im:im.verify()
    assert not path.with_suffix('.png.tmp').exists()
