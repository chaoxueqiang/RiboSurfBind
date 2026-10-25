#!/usr/bin/env python3
"""Print or execute the paper's benchmark, ablation and sensitivity jobs."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.io import read_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['benchmarks','ablations','sensitivity','case_models'],required=True)
    parser.add_argument('--protocol',choices=['test18','hariboss','external'])
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,default=ROOT/'results/new_runs')
    parser.add_argument('--execute',action='store_true',help='Without this flag, print commands only')
    args=parser.parse_args()
    jobs=[]
    protocols=read_json(ROOT/'configs/protocols.json')
    if args.suite=='benchmarks':
        for protocol in ([args.protocol] if args.protocol else protocols):
            for seed in (args.seeds or protocols[protocol]['paper_seeds']):
                for fold in (range(1,5) if protocol=='hariboss' else [1]):
                    jobs.append((f'benchmarks/{protocol}/set{fold}/seed{seed}',
                                 ['--protocol',protocol,'--set',str(fold),'--seed',str(seed)]))
    elif args.suite=='ablations':
        for variant in read_json(ROOT/'configs/ablations.json'):
            jobs.append((f'ablations/{variant}',['--protocol','test18','--seed','1','--ablation',variant]))
    elif args.suite=='sensitivity':
        for parameter,values in read_json(ROOT/'configs/sensitivity.json').items():
            for value in values:
                jobs.append((f'sensitivity/{parameter}/{value}',
                             ['--protocol','test18','--seed','1','--'+parameter.replace('_','-'),str(value)]))
    else:
        for protocol,fold in [('test18',1),('hariboss',2),('external',1)]:
            for variant in ['att_only','surface_only']:
                jobs.append((f'case_models/{protocol}/{variant}',
                             ['--protocol',protocol,'--set',str(fold),'--seed','1','--ablation',variant]))
    for name,options in jobs:
        output=args.output/name
        command=[sys.executable,str(ROOT/'train.py')]+options+['--device',args.device,'--output',str(output)]
        if (output/'result.json').exists():
            print('SKIP completed:',name,flush=True)
            continue
        print(shlex.join(command),flush=True)
        if args.execute:
            subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':
    main()
