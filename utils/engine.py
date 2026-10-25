"""Shared Trial20 engine. Validation uses one RNA per batch, mean graph loss.

Test metrics are pooled across nucleotides. Tests never select a checkpoint or
a threshold. Published checkpoints are selected by minimum validation loss.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from models import RiboSurfBind
from data_process.dataset import load_group
from utils.io import write_json
from utils.metrics import calculate_metrics
from utils.reproducibility import seed_all, rng_state, restore_rng

def load_weights(model, path):
    payload = torch.load(path,map_location='cpu',weights_only=False)
    state = payload.get('model_state_dict',payload) if isinstance(payload,dict) else payload
    model.load_state_dict(state,strict=True)

def make_model(config, paths, device):
    # Packed benchmarks carry their own inputs. Explicit cache directories are
    # accepted only for the optional raw-record prediction/preparation path.
    return RiboSurfBind(config,paths.get('attention'),paths.get('surface')).to(device)

def learning_rate(epoch,config):
    if epoch <= config['warmup_epochs']:
        return config['warmup_start_lr'] + (config['max_lr']-config['warmup_start_lr'])*epoch/config['warmup_epochs']
    progress = (epoch-config['warmup_epochs'])/max(1,config['epochs']-config['warmup_epochs'])
    return config['final_lr'] + .5*(config['max_lr']-config['final_lr'])*(1+math.cos(math.pi*progress))

def training_loss(model, batch, criterion, config):
    probabilities, logits = model(batch,return_classifier_logits=True)
    main = criterion(logits,batch.y.float())
    aux = model.last_branch_aux_logits
    weight = float(config['surface_aux_weight'])
    surface = logits.new_zeros(())
    if weight:
        valid = aux['surface_mask'].reshape(-1)>0
        if not bool(valid.any()):
            raise ValueError('Surface auxiliary loss has no valid residues')
        surface = criterion(aux['surface'][valid],batch.y.float()[valid])
    # The original Train60 runner explicitly included 0 * ATT auxiliary BCE.
    # Its zero gradients still create Adam state / weight-decay updates for
    # that auxiliary head. HARIBOSS's runner omitted the term. Preserve both
    # historical optimizer trajectories instead of assuming None == zero.
    if config.get('protocol','test18') == 'test18' and weight:
        attention = criterion(aux['attention'],batch.y.float())
        return main + 0.0*attention + weight*surface, probabilities
    return main + weight*surface, probabilities

@torch.no_grad()
def evaluate_model(model, graphs, device, batch_size=16, output=None):
    model.eval()
    records = []
    losses = []
    criterion = torch.nn.BCEWithLogitsLoss()
    for batch in DataLoader(graphs,batch_size=batch_size,shuffle=False):
        batch = batch.to(device)
        scores, logits = model(batch,return_classifier_logits=True)
        ptr = batch.ptr.detach().cpu().tolist()
        ids = batch.node_feature_record_id
        for rid,start,end in zip(ids,ptr[:-1],ptr[1:]):
            labels = batch.y[start:end]
            losses.append(float(criterion(logits[start:end],labels.float()).cpu()))
            records.append(dict(record_id=rid,labels=labels.cpu().numpy(),probabilities=scores[start:end].cpu().numpy()))
    y = np.concatenate([r['labels'] for r in records])
    p = np.concatenate([r['probabilities'] for r in records])
    result = calculate_metrics(y,p)
    result['loss_mean_graph'] = float(np.mean(losses))
    result['mean_graph_mcc'] = float(np.mean([calculate_metrics(r['labels'],r['probabilities'])['mcc'] for r in records]))
    result['records'] = len(records)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output,record_ids=np.asarray([r['record_id'] for r in records],dtype=str),
                            offsets=np.asarray([0]+list(np.cumsum([len(r['labels']) for r in records]))),
                            labels=y,probabilities=p)
    return result

def fit(config,datasets,paths,settings,output,device,resume=None,smoke_only=False):
    output = Path(output)
    if (output/'result.json').exists() and not resume:
        raise FileExistsError('Completed result exists; use another --output directory')
    output.mkdir(parents=True,exist_ok=True)
    seed_all(int(config['seed']),bool(config['strict_determinism']))
    generator = torch.Generator().manual_seed(int(config['seed']))
    train_loader = DataLoader(datasets['train'],batch_size=int(config['batch_size']),shuffle=True,generator=generator)
    model = make_model(config,paths,device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config['max_lr'],weight_decay=config['weight_decay'])
    criterion = torch.nn.BCEWithLogitsLoss()
    best = {'validation_loss':float('inf'),'epoch':None}
    start = 1
    if resume:
        state = torch.load(resume,map_location='cpu',weights_only=False)
        if state['config'] != config:
            raise ValueError('Resume configuration differs from the original run')
        model.load_state_dict(state['model_state_dict'],strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        # Optimizer moments must reside with the corresponding parameters.
        for parameter,moments in optimizer.state.items():
            for key,value in moments.items():
                if torch.is_tensor(value) and key != 'step':
                    moments[key] = value.to(parameter.device)
        best = state['best']
        start = state['epoch']+1
        restore_rng(state['rng'],generator)
    elif (output/'metrics.jsonl').exists():
        raise FileExistsError('Partial run exists; use --resume or a new --output')
    write_json(output/'config.json',config)
    write_json(output/'data_manifest.json',{split:[g.node_feature_record_id for g in graphs] for split,graphs in datasets.items()})
    if smoke_only:
        model.train()
        batch = next(iter(train_loader)).to(device)
        loss,_ = training_loss(model,batch,criterion,config)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad_norm'])
        optimizer.step()
        write_json(output/'smoke_test.json',dict(status='PASS',loss=float(loss.detach().cpu()),gradient_norm=float(norm.cpu())))
        return {'status':'smoke_only'}
    for epoch in range(start,int(config['epochs'])+1):
        lr = float(learning_rate(epoch,config))
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss,_ = training_loss(model,batch,criterion,config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad_norm'])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate_model(model,datasets['validation'],device,batch_size=1)
        row = dict(epoch=epoch,lr=lr,train_loss=float(np.mean(losses)),validation=validation)
        with (output/'metrics.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n')
        if validation['loss_mean_graph'] < best['validation_loss']:
            best = dict(validation_loss=validation['loss_mean_graph'],epoch=epoch)
            torch.save(model.state_dict(),output/'best_model.pt')
        state = dict(epoch=epoch,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                     config=config,best=best,rng=rng_state(generator))
        temporary = output/'resume.pt.tmp'
        torch.save(state,temporary)
        temporary.replace(output/'resume.pt')
        print(f"epoch={epoch:03d} lr={lr:.8g} train={row['train_loss']:.6f} val={validation['loss_mean_graph']:.6f}",flush=True)
    load_weights(model,output/'best_model.pt')
    result = dict(status='complete',checkpoint_selection='minimum_validation_mean_graph_BCE',best=best,
                  threshold=.5,threshold_operator='>',splits={})
    for split,graphs in datasets.items():
        if split != 'train':
            bs = 1 if split=='validation' else settings['test_batch_size']
            result['splits'][split] = evaluate_model(model,graphs,device,bs,output/f'predictions/{split}.npz')
    if config['protocol']=='external':
        for split in ('jl10','tl12'):
            graphs,_ = load_group(split,data_root=config.get('data_root'))
            result['splits'][split] = evaluate_model(model,graphs,device,4,output/f'predictions/{split}.npz')
    write_json(output/'result.json',result)
    return result
