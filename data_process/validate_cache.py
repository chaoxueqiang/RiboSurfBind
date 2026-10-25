#!/usr/bin/env python3
"""Audit inputs; optionally repair only the established curvature-Inf case."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--dataset-root',type=Path,help='Intermediate records/ERNIE/surface directory')
    inputs.add_argument('--package',type=Path,help='Self-contained processed .pt package')
    parser.add_argument('--output',type=Path,help='Optional validation report (input files are not modified)')
    parser.add_argument('--repair-curvature-output',type=Path,help='Write repaired copies here; never edit source caches')
    args=parser.parse_args()
    import numpy as np
    from data_process.dataset import load_record
    from utils.io import write_json
    if args.package:
        if args.repair_curvature_output:
            parser.error('Curvature repair operates on intermediate caches, not packed inputs')
        from data_process.packed import read_package
        payload=read_package(args.package,require_labels=False)
        result=dict(status='PASS',**payload['counts'])
        if args.output:
            write_json(args.output,result)
        print(result)
        return
    repairs=[]
    surface_dir=args.dataset_root/'surface'
    if args.repair_curvature_output:
        if args.repair_curvature_output.resolve()==surface_dir.resolve():
            parser.error('Repair output must differ from source surface directory')
        args.repair_curvature_output.mkdir(parents=True,exist_ok=True)
        for source in sorted(surface_dir.glob('*.npz')):
            with np.load(source,allow_pickle=False) as payload:
                arrays={key:payload[key].copy() for key in payload.files}
            xyz=arrays['surface_xyz'].astype(np.float64)
            chemistry=arrays['chemical_features']
            curvature=arrays['curvature']
            bad=~np.isfinite(curvature);good=np.isfinite(curvature)
            if bad.any():
                if not good.any() or not np.array_equal(bad,~np.isfinite(chemistry[:,-1])) or not np.isfinite(xyz).all():
                    raise ValueError('Unsupported non-finite pattern: '+source.name)
                for index in np.flatnonzero(bad):
                    donor=np.flatnonzero(good)[int(np.argmin(((xyz[good]-xyz[index])**2).sum(1)))]
                    curvature[index]=curvature[donor];chemistry[index,-1]=curvature[donor]
            if not all(np.isfinite(arrays[key]).all() for key in ['surface_xyz','surface_nuv','chemical_features']):
                raise ValueError('Invalid surface: '+source.name)
            np.savez_compressed(args.repair_curvature_output/source.name,**arrays)
            repairs.append(dict(record_id=source.stem,repaired_vertices=int(bad.sum())))
        surface_dir=args.repair_curvature_output
    count=nodes=0
    for path in sorted((args.dataset_root/'records').glob('*.npz')):
        graph=load_record(path,args.dataset_root/'embeddings',args.dataset_root/'attention',surface_dir,require_labels=False)
        with np.load(surface_dir/f'{graph.node_feature_record_id}.npz',allow_pickle=False) as surface:
            n=int(graph.num_nodes)
            owner=surface['owner_residue']
            if not np.isfinite(surface['chemical_features']).all() or owner.min()<0 or owner.max()>=n:
                raise ValueError('Invalid surface mapping: '+path.name)
        count+=1;nodes+=graph.num_nodes
    if not count:
        parser.error('No records found')
    result=dict(status='PASS',records=count,nodes=nodes,curvature_repairs=repairs)
    if args.output:
        write_json(args.output,result)
    print('PASS',count,'records',nodes,'nucleotides')

if __name__=='__main__':
    main()
