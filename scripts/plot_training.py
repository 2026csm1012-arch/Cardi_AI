import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cardiac.evaluation.reporting import save_plot
from cardiac.utils.io import read_json

p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True)
a=p.parse_args();fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for run in a.runs:
    df=pd.read_csv(Path(run)/'training.csv')
    axes[0].plot(df.epoch,df.train_mse_fraction,label=Path(run).name)
    axes[1].plot(df.epoch,df.val_mae_pp,label=Path(run).name)
axes[0].set(xlabel='Epoch',ylabel='Training MSE (EF fraction squared)')
axes[1].set(xlabel='Epoch',ylabel='Complete validation MAE (EF pp)')
for ax in axes:ax.legend();ax.grid(alpha=.2)
if any(read_json(Path(run)/'run.json')['provenance']=='synthetic_software_test' for run in a.runs):
    fig.suptitle('SYNTHETIC SOFTWARE TEST - NOT RESEARCH RESULTS')
Path(a.out).parent.mkdir(parents=True,exist_ok=True);save_plot(fig,Path(a.out))
