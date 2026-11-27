"""Tiny artificial moving disks. SOFTWARE TESTS ONLY; not physiological training data."""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from cardiac.utils.io import write_json


def create(root):
    root=Path(root);(root/'Videos').mkdir(parents=True,exist_ok=True)
    labels=[];traces=[]
    for split in ['TRAIN','VAL','TEST']:
        for i,ef in enumerate([40.,65.]):
            name=f'fixture_{split.lower()}_{i}'
            path=root/'Videos'/f'{name}.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30.,(32,32))
            if not writer.isOpened():raise RuntimeError('MJPG writer unavailable.')
            for t in range(24):
                frame=np.zeros((32,32,3),np.uint8)
                radius=5+i+int(2*np.cos(t*np.pi/6))
                cv2.circle(frame,(16,16),radius,(180,180,180),-1)
                writer.write(frame)
            writer.release()
            labels.append({'FileName':name,'EF':ef,'Split':split})
            for t,r in [(4,9),(8,5)]:
                # First line is long axis; remaining lines form a rectangle.
                for x1,y1,x2,y2 in [(16,16-r,16,16+r),(16-r,16-r,16+r,16-r),(16-r,16+r,16+r,16+r)]:
                    traces.append({'FileName':name+'.avi','X1':x1,'Y1':y1,'X2':x2,'Y2':y2,'Frame':t})
    for name,rows in [('FileList.csv',labels),('VolumeTracings.csv',traces)]:
        with open(root/name,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(root/'SYNTHETIC_FIXTURE.json',{'purpose':'software tests only','labels':'fixed artificial targets; no clinical meaning'})
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);create(p.parse_args().out)
