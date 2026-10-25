#!/usr/bin/env python3
"""Train the published model under one explicit benchmark protocol."""
import argparse
from pathlib import Path
from utils.reproducibility import seed_all
from utils.io import ROOT, load_config, read_json
from data_process.dataset import load_protocol
from utils.engine import fit

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',choices=['test18','hariboss','external','custom'],required=True)
    parser.add_argument('--config',default='configs/trial20.json')
    parser.add_argument('--set',dest='set_id',type=int,choices=[1,2,3,4],default=1)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--data-root',type=Path,help='Directory containing processed/ packages and splits/ (default: repository data/)')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--ablation',choices=['att_only','surface_only','semantic_only','spatial_only','no_relation_gate'])
    parser.add_argument('--attention-depth',type=int,choices=range(1,7))
    parser.add_argument('--semantic-topk',type=int)
    parser.add_argument('--spatial-knn',type=int)
    parser.add_argument('--epochs',type=int,help='Explicit non-paper schedule override, for development only')
    parser.add_argument('--threads',type=int,default=2)
    args = parser.parse_args()
    if args.protocol != 'hariboss' and args.set_id != 1:
        parser.error('--set applies only to HARIBOSS')
    if args.protocol == 'custom' and not args.data_root:
        parser.error('Custom training requires an explicit --data-root')
    config = load_config(args.config)
    if args.ablation:
        config.update(read_json(ROOT/'configs/ablations.json')[args.ablation])
    for name in ('attention_depth','semantic_topk','spatial_knn','epochs'):
        if getattr(args,name) is not None:
            config[name] = getattr(args,name)
    if config['semantic_topk']<1 or config['spatial_knn']<1 or config['epochs']<1:
        parser.error('Depth, neighbors and epochs must be positive')
    config.update(protocol=args.protocol,set_id=args.set_id,seed=args.seed,
                  data_root=str(args.data_root.resolve()) if args.data_root else None,
                  ablation=args.ablation)
    import torch
    torch.set_num_threads(args.threads)
    output = args.output or ROOT/'results/new_runs'/args.protocol/f'set{args.set_id}'/f'seed{args.seed}'
    datasets,paths,settings = load_protocol(args.protocol,args.set_id,args.data_root)
    result = fit(config,datasets,paths,settings,output,args.device,args.resume,args.smoke_only)
    print(result['status'])

if __name__=='__main__':
    main()
