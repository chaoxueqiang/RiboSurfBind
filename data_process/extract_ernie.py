#!/usr/bin/env python3
"""Extract frozen final-layer ERNIE states and RNA-renormalized attention."""
import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Dataset root containing embeddings/ and attention/')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/ernie_rna/ERNIE-RNA_pretrain.pt')
    parser.add_argument('--ernie-repo',type=Path,default=ROOT/'third_party/ERNIE-RNA-official')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    import numpy as np
    import torch
    from utils.io import sha256_file
    from data_process.attention import validate_attention_matrix
    sequences={}
    for path in sorted(args.records.glob('*.npz')):
        with np.load(path,allow_pickle=False) as raw:
            sequence=str(raw['sequence'].item())
            sequences.setdefault(sequence,[]).append(str(raw['record_id'].item()))
    if not sequences:
        parser.error('No canonical NPZ records found')
    if not args.checkpoint.is_file():
        parser.error('ERNIE checkpoint not found')
    sys.path.insert(0,str(args.ernie_repo.resolve()))
    from src.utils import ErnieRNAOnestage,load_pretrained_ernierna,prepare_input_for_ernierna
    from extract_embedding import seq_to_index
    pretrained=load_pretrained_ernierna(str(args.checkpoint),{'data':str(args.ernie_repo/'src/dict')})
    model=ErnieRNAOnestage(pretrained.encoder).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for folder in ('embeddings','attention'):
        (args.output/folder).mkdir(parents=True,exist_ok=True)
    checkpoint_sha=sha256_file(args.checkpoint)
    for sequence,ids in sorted(sequences.items()):
        n=len(sequence);key='seq_'+hashlib.sha256(sequence.encode()).hexdigest()
        indices,lengths=seq_to_index([sequence]);tokens=int(lengths[0])+2
        one,two=prepare_input_for_ernierna(indices[0],int(lengths[0]))
        with torch.no_grad():
            _,raw_attention,extra=model.sentence_encoder(
                one.to(args.device),twod_tokens=two.to(device=args.device,dtype=torch.float32),
                is_twod=True,extra_only=True,masked_only=False)
        embedding=extra['inner_states'][-1].transpose(0,1)[0,1:n+1].float().cpu()
        att=torch.softmax(raw_attention[-1][0,:,:tokens,:tokens],dim=-1)[:,1:n+1,1:n+1]
        att=att/att.sum(-1,keepdim=True).clamp_min(torch.finfo(att.dtype).tiny)
        att=att.mean(0);att=att/att.sum(-1,keepdim=True).clamp_min(torch.finfo(att.dtype).tiny)
        att=att.float().cpu();validate_attention_matrix(att,n,key)
        if tuple(embedding.shape)!=(n,768):
            raise ValueError('Invalid ERNIE residue embedding')
        torch.save(dict(sequence=sequence,sequence_key=key,embedding=embedding,record_ids=ids,
                        checkpoint_sha256=checkpoint_sha),args.output/'embeddings'/f'{key}.pt')
        torch.save(dict(sequence=sequence,sequence_key=key,sequence_sha256=key[4:],num_nodes=n,attention=att,
                        cache_schema_version='ernie_attention_postsoftmax_v2',
                        attention_semantics='per_head_postsoftmax_rna_renormalized_mean',
                        raw_source_semantics='ERNIE pre-softmax attention bias transformed to probability',
                        special_token_policy='softmax_full_then_drop_bos_eos_then_rna_renormalize',
                        source_model='official ERNIE-RNA',source_checkpoint=args.checkpoint.name,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        ernie_repo_commit='43bc06de1088ed03ffd7de918ad4b2c2a3346a43',checkpoint_sha256=checkpoint_sha),
                   args.output/'attention'/f'{key}.pt')
        print(key,n,flush=True)

if __name__=='__main__':
    main()
