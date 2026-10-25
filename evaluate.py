#!/usr/bin/env python3
"""Evaluate a published or newly trained checkpoint at the fixed threshold."""
import argparse
import json
from pathlib import Path
from utils.reproducibility import seed_all
from utils.io import load_config,write_json
from data_process.dataset import load_protocol,load_group
from utils.engine import make_model,load_weights,evaluate_model
from utils.checkpoints import evaluation_set

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',choices=['test18','hariboss','external','custom'],required=True)
    parser.add_argument('--set',dest='set_id',type=int,choices=[1,2,3,4],help='HARIBOSS set (defaults to checkpoint metadata)')
    parser.add_argument('--split',choices=['test18','test','internal_test','jl10','tl12','validation'])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--data-root',type=Path,help='Directory containing processed/ packages and splits/ (default: repository data/)')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=2)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    config_path = args.config or args.checkpoint.parent/'config.json'
    config = load_config(config_path)
    sibling = args.checkpoint.parent/'config.json'
    checkpoint_config = load_config(sibling) if args.config and sibling.is_file() else None
    try:
        args.set_id = evaluation_set(config,args.protocol,args.set_id,checkpoint_config)
    except ValueError as error:
        parser.error(str(error))
    seed_all(int(config.get('seed',1)))
    split = args.split or {'test18':'test18','hariboss':'test','external':'tl12','custom':'test'}[args.protocol]
    if split in ('jl10','tl12'):
        if args.protocol!='external':
            parser.error('JL10/TL12 belong to the external benchmark protocol')
        graphs,paths = load_group(split,data_root=args.data_root)
        batch_size = 4
    else:
        datasets,paths,settings = load_protocol(args.protocol,args.set_id,args.data_root)
        if split not in datasets:
            parser.error('Split is not present in this protocol')
        graphs = datasets[split]
        batch_size = 1 if split=='validation' else settings['test_batch_size']
    model = make_model(config,paths,args.device)
    load_weights(model,args.checkpoint)
    args.output.mkdir(parents=True,exist_ok=True)
    result = evaluate_model(model,graphs,args.device,batch_size,args.output/f'{split}_predictions.npz')
    write_json(args.output/f'{split}_metrics.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
