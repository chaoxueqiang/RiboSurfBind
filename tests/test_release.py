"""Fast CPU checks; no GPU, KeOps compilation or reference code required."""
import csv
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch_geometric.data import Data
from models import RiboSurfBind
from models.semantic_spatial import GCNIIConvolution,SemanticSpatialBranch,run_batched_attention_branch
from utils.io import load_config,read_json,sha256_file
from utils.metrics import calculate_metrics
from utils.engine import learning_rate,fit
from data_process.dataset import load_protocol,load_group,load_package_graphs
from data_process.packed import read_package
from data_process.attention import load_batch_attention
from models.surface import load_batch_surface
from torch_geometric.data import Batch

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=torch.nn.Linear(1,4)
        self.output=torch.nn.Linear(4,1)
        self.att=torch.nn.Linear(4,1)
        self.surface=torch.nn.Linear(4,1)
    def forward(self,data,return_classifier_logits=False):
        x=torch.nn.functional.dropout(torch.relu(self.hidden(data.x)),.3,self.training)
        logits=self.output(x).squeeze(-1)
        self.last_branch_aux_logits=dict(attention=self.att(x).squeeze(-1),surface=self.surface(x).squeeze(-1),surface_mask=torch.ones_like(logits)[:,None])
        probability=torch.sigmoid(logits)
        return (probability,logits) if return_classifier_logits else probability

class ReleaseTests(unittest.TestCase):
    def test_package_inventory_and_counts(self):
        expected={'train60':(60,3054,950),'test18':(18,610,207),
                  'hariboss':(353,35114,4958),'hariboss_tr60':(432,22432,7085),
                  'tl12':(12,515,183),'jl10':(10,821,264)}
        manifest=read_json(ROOT/'data/manifest.json')
        for name,(records,nodes,positives) in expected.items():
            path=ROOT/'data/processed'/f'{name}.pt'
            self.assertEqual(sha256_file(path),manifest['packages'][name]['sha256'])
            package=read_package(path)
            self.assertEqual((package['counts']['records'],package['counts']['nucleotides'],
                              package['counts']['positives']),(records,nodes,positives))
    def test_all_ordered_protocol_splits(self):
        for protocol,sets in [('test18',[1]),('hariboss',[1,2,3,4]),('external',[1])]:
            for set_id in sets:
                datasets,paths,_=load_protocol(protocol,set_id)
                filename=(f'hariboss_set{set_id}' if protocol=='hariboss' else
                          'train60' if protocol=='test18' else 'hariboss_tr60')
                expected=read_json(ROOT/'data/splits'/f'{filename}.json')
                for split,graphs in datasets.items():
                    self.assertEqual([g.node_feature_record_id for g in graphs],expected[split])
                self.assertNotIn('attention',paths)
                self.assertNotIn('surface',paths)
        for group in ['tl12','jl10']:
            graphs,_=load_group(group)
            # load_group uses package order. Manifests additionally preserve
            # the original explicit evaluation order for external benchmarks.
            expected=read_json(ROOT/'data/splits'/f'{group}.json')[group]
            self.assertEqual([g.node_feature_record_id for g in graphs],expected)
    def test_packed_variable_size_batch_and_rng(self):
        graphs=load_package_graphs(ROOT/'data/processed/test18.pt')
        chosen=[min(graphs,key=lambda g:g.num_nodes),max(graphs,key=lambda g:g.num_nodes)]
        before=torch.get_rng_state().clone()
        batch=Batch.from_data_list(chosen).to('cpu')
        with patch('data_process.attention.load_attention_record',side_effect=AssertionError('Unexpected disk attention dependency')), \
             patch('models.surface.np.load',side_effect=AssertionError('Unexpected disk surface dependency')):
            matrices,ptr=load_batch_attention(batch,None,'cpu')
            bounds,surfaces=load_batch_surface(batch,None,'cpu')
        self.assertEqual(bounds,ptr.tolist())
        self.assertEqual(len(batch.rna_inputs),2)
        for i,graph in enumerate(chosen):
            self.assertTrue(torch.equal(matrices[i],graph.rna_inputs.attention))
            for key in ['xyz','nuv','chemistry','atom_name_index','owner']:
                self.assertTrue(torch.equal(surfaces[i][key],graph.rna_inputs.surface[key]))
            self.assertLess(int(surfaces[i]['owner'].max()),graph.num_nodes)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_published_initialization_and_rng(self):
        # Golden hashes captured from the pre-cleanup model in the pinned
        # torch 2.2 environment. Compatibility layers are intentionally kept.
        torch.manual_seed(1)
        model=RiboSurfBind(load_config(),'unused','unused')
        digest=hashlib.sha256()
        for key,value in model.state_dict().items():
            digest.update(key.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.numpy().tobytes())
        self.assertEqual(digest.hexdigest(),'ff4f8e816589d8b52c5aaed7e4b0ed9df5d57db5f44fcfb043702586c44b1d64')
        self.assertEqual(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                         'e8e6d6345fc028a38b4e07a5998707fb8cdea952491aedd27552f1747b413269')
    def test_gcnii_published_equation(self):
        torch.manual_seed(23)
        conv=GCNIIConvolution(4)
        x=torch.randn(5,4);h0=torch.randn(5,4)
        adjacency=torch.softmax(torch.randn(5,5),dim=-1)
        alpha=.5;lamda=1.;layer=3
        support=(1-alpha)*torch.matmul(adjacency,x)+alpha*h0
        theta=min(1.0,math.log(lamda/layer+1.0))
        expected=theta*torch.mm(support,conv.weight)+(1-theta)*support
        expected=expected+x
        self.assertTrue(torch.equal(conv(x,adjacency,h0,lamda,alpha,layer),expected))
        message=torch.matmul(adjacency,x)
        self.assertTrue(torch.equal(conv(x,None,h0,lamda,alpha,layer,neighbor_message=message),expected))
    def test_attention_returns_only_final_states(self):
        config=load_config()
        torch.manual_seed(29)
        branch=SemanticSpatialBranch(config)
        x=torch.randn(7,768);positions=torch.randn(7,3)
        attention=[torch.softmax(torch.randn(n,n),dim=-1) for n in [3,4]]
        ptr=torch.tensor([0,3,7])
        branch.train()
        torch.manual_seed(31)
        a=run_batched_attention_branch(branch,x,attention,ptr,positions)
        rng=torch.get_rng_state()
        torch.manual_seed(31)
        b=run_batched_attention_branch(branch,x,attention,ptr,positions)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(tuple(a.shape),(7,128))
    def test_fixed_threshold(self):
        m=calculate_metrics([0,1,1,0],[.5,.5,.9,.1])
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']),(1,0,1,2))
    def test_learning_rate(self):
        config=load_config()
        self.assertAlmostEqual(learning_rate(10,config),config['max_lr'],places=18)
        self.assertEqual(learning_rate(100,config),config['final_lr'])
    def test_all_published_checkpoints_load_strictly(self):
        manifest=read_json(ROOT/'docs/artifact_manifest.json')
        self.assertEqual(len(manifest),55)
        for row in manifest:
            path=ROOT/row['target']
            self.assertEqual(sha256_file(path),row['sha256'])
            model=RiboSurfBind(read_json(path.parent/'config.json'),'unused','unused')
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False),strict=True)
    def test_external_published_means(self):
        with (ROOT/'results/published_metrics.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        for split,expected in [('tl12',[.8746362499177035,.8284313160038097,.6688524590163935,.6268727098921524]),
                               ('jl10',[.796055709700234,.6211389644588876,.5378787878787878,.3980454112673963])]:
            values=[r for r in rows if r['experiment']=='external' and r['split']==split]
            self.assertEqual(len(values),5)
            for metric,value in zip(['auc','precision','recall','mcc'],expected):
                self.assertAlmostEqual(np.mean([float(r[metric]) for r in values]),value,places=14)
    def test_epoch_boundary_resume(self):
        # Validate engine RNG/optimizer/loader restoration with a lightweight
        # dropout model without KeOps. The separate GPU-check scope is
        # documented in docs/REPRODUCIBILITY.md.
        graphs=[]
        for i in range(4):
            graphs.append(Data(x=torch.tensor([[.2],[.8]]),y=torch.tensor([0.,1.]),node_feature_record_id=f'toy{i}'))
        datasets={'train':graphs[:2],'validation':[graphs[2]],'test18':[graphs[3]]}
        config=dict(load_config(),epochs=2,batch_size=1,seed=3,protocol='test18',set_id=1,data_root=None)
        settings={'test_batch_size':1}
        paths={'attention':'unused','surface':'unused'}
        with tempfile.TemporaryDirectory() as work:
            full=Path(work)/'full';partial=Path(work)/'partial'
            with patch('utils.engine.make_model',side_effect=lambda *a:ToyModel()):
                fit(config,datasets,paths,settings,full,'cpu')
                def interrupt(epoch,cfg):
                    if epoch==2:
                        raise InterruptedError('simulated epoch-boundary interruption')
                    return learning_rate(epoch,cfg)
                with patch('utils.engine.learning_rate',side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        fit(config,datasets,paths,settings,partial,'cpu')
                fit(config,datasets,paths,settings,partial,'cpu',resume=partial/'resume.pt')
            a=torch.load(full/'resume.pt',weights_only=False)
            b=torch.load(partial/'resume.pt',weights_only=False)
            for key,value in a['model_state_dict'].items():
                self.assertTrue(torch.equal(value,b['model_state_dict'][key]),key)
            self.assertEqual((full/'metrics.jsonl').read_text(),(partial/'metrics.jsonl').read_text())

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
