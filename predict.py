#!/usr/bin/env python3
"""Predict binding probabilities for prepared, potentially unlabelled RNAs."""
import argparse
import csv
from pathlib import Path
from utils.reproducibility import seed_all
from utils.io import load_config
from data_process.dataset import load_record,load_package_graphs
from utils.engine import make_model,load_weights

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--data',type=Path,help='Prepared self-contained RNA .pt package')
    inputs.add_argument('--records',type=Path,help='Optional raw-preparation NPZ record or directory')
    parser.add_argument('--embedding-dir',type=Path)
    parser.add_argument('--attention-dir',type=Path)
    parser.add_argument('--surface-dir',type=Path)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    config = load_config(args.config or args.checkpoint.parent/'config.json')
    seed_all(int(config.get('seed',1)))
    if args.data:
        if any(value is not None for value in (args.embedding_dir,args.attention_dir,args.surface_dir)):
            parser.error('--data is self-contained; do not also supply cache directories')
        graphs = load_package_graphs(args.data,require_labels=False)
        input_paths = {}
    else:
        if any(value is None for value in (args.embedding_dir,args.attention_dir,args.surface_dir)):
            parser.error('--records requires --embedding-dir, --attention-dir and --surface-dir')
        paths = [args.records] if args.records.is_file() else sorted(args.records.glob('*.npz'))
        if not paths:
            parser.error('No input records found')
        graphs = [load_record(path,args.embedding_dir,args.attention_dir,args.surface_dir,require_labels=False) for path in paths]
        input_paths = {'attention':args.attention_dir,'surface':args.surface_dir}
    model = make_model(config,input_paths,args.device).eval()
    load_weights(model,args.checkpoint)
    args.output.mkdir(parents=True,exist_ok=True)
    for graph in graphs:
        with torch.no_grad():
            probabilities = model(graph.to(args.device)).cpu().tolist()
        with (args.output/f'{graph.node_feature_record_id}.csv').open('w',newline='',encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['record_id','sequence_position','residue_id','nucleotide','probability','predicted_label'])
            for index,(base,prob,resid) in enumerate(zip(graph.rna_sequence,probabilities,graph.residue_ids),1):
                writer.writerow([graph.node_feature_record_id,index,resid,base,repr(prob),int(prob>.5)])
        print(graph.node_feature_record_id,len(probabilities),flush=True)

if __name__=='__main__':
    main()
