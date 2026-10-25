#!/usr/bin/env python3
"""Plot HARIBOSS bars and sensitivity curves from archived numerical results."""
import argparse
import csv
import json
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/reproduced_figures')
    parser.add_argument('--only',choices=['all','hariboss'],default='all')
    parser.add_argument('--alignment-helper',type=Path,help='Optional external audit_panel_alignment.py for export QA')
    args=parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    args.output.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((ROOT/'results/published_metrics.csv').open()))
    seeds=set(json.loads((ROOT/'configs/protocols.json').read_text())['hariboss']['paper_seeds'])
    for fold in range(1,5):
        selected=[r for r in rows if r['experiment']==f'hariboss_set{fold}']
        if len(selected)!=len(seeds) or {int(r['seed']) for r in selected}!=seeds:
            raise ValueError('HARIBOSS results do not match the configured paper seeds')
    alignment=None
    if args.alignment_helper:
        spec=importlib.util.spec_from_file_location('figure_alignment',args.alignment_helper)
        alignment=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=alignment
        spec.loader.exec_module(alignment)
    def save_figure(fig,name):
        fig.canvas.draw()
        if alignment:
            alignment.require_matplotlib_panel_alignment(fig,
                json_out=str(args.output/(name+'.alignment.json')),
                tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        fig.savefig(args.output/(name+'.png'),dpi=600)
        fig.savefig(args.output/(name+'.pdf'))
        fig.savefig(args.output/(name+'.svg'))
    # Export tables from raw values with one final formatting step.
    with (args.output/'metrics_three_decimals.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({key:(f'{float(value):.3f}' if key in ('auc','precision','recall','mcc') else value) for key,value in row.items()})
    baseline=list(csv.DictReader((ROOT/'results/figure_sources/hariboss_bars.csv').open(encoding='utf-8-sig')))
    # Baselines are literature values; only our five actual runs are aggregated.
    def draw_hariboss(ax,metric):
        methods=list(dict.fromkeys(r['Method'] for r in baseline))
        colors=['#CFD0DC','#ADB8CF','#7D9EB5','#B5C8C0','#D3BFA5','#AC657E']
        width=.8/len(methods)
        for index,method in enumerate(methods):
            values=[];errors=[]
            for fold in range(1,5):
                if method=='RiboSurfBind':
                    v=[float(r[metric]) for r in rows if r['experiment']==f'hariboss_set{fold}']
                    values.append(float(np.mean(v)));errors.append(float(np.std(v,ddof=1)))
                else:
                    row=next(r for r in baseline if r['Method']==method and r['Split']==f'Set{fold}')
                    values.append(float(row[metric.upper()]));errors.append(0.)
            ax.bar(np.arange(4)+(index-(len(methods)-1)/2)*width,values,width=width,
                   color=colors[index%len(colors)],label=method,
                   yerr=errors if method=='RiboSurfBind' else None,
                   capsize=2,error_kw={'elinewidth':.7,'capthick':.7})
        ax.set_xticks(range(4),[f'Set{i}' for i in range(1,5)])
        ax.legend(frameon=False,ncol=3,fontsize=7,bbox_to_anchor=(.5,1.25),loc='upper center')
        ax.set_ylabel(metric.upper());ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(0,1)
    note='RiboSurfBind: mean ± sample SD, n = 5 seeds; baselines: reported values.'
    for metric in ('auc','mcc'):
        fig,ax=plt.subplots(figsize=(7.2,3.6))
        draw_hariboss(ax,metric)
        fig.subplots_adjust(left=.085,right=.985,bottom=.18,top=.76)
        fig.text(.5,.025,note,ha='center',fontsize=6.5)
        save_figure(fig,'hariboss_'+metric)
        plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(7.2,6.8))
    for label,ax,metric in zip(('a','b'),axes,('auc','mcc')):
        draw_hariboss(ax,metric)
        ax.annotate(label,xy=(0,1),xycoords='axes fraction',xytext=(-25,18),
                    textcoords='offset points',fontweight='bold',fontsize=9)
    fig.subplots_adjust(left=.085,right=.985,bottom=.09,top=.88,hspace=.55)
    fig.text(.5,.02,note,ha='center',fontsize=6.5)
    save_figure(fig,'hariboss_auc_mcc')
    plt.close(fig)
    (args.output/'hariboss_figure_metadata.json').write_text(json.dumps({
        'seeds':sorted(seeds),'center':'arithmetic mean of unrounded per-run metrics',
        'spread':'sample SD, ddof=1','baseline_uncertainty':'not available; no error bars inferred',
        'source':'results/published_metrics.csv','baseline_source':'results/figure_sources/hariboss_bars.csv',
        'significance_test':None},indent=2)+'\n')
    if args.only=='hariboss':
        return
    fig,axes=plt.subplots(1,3,figsize=(11,3.2))
    for ax,parameter in zip(axes,['attention_depth','semantic_topk','spatial_knn']):
        selected=[r for r in rows if r['experiment'].startswith('sensitivity_'+parameter+'_')]
        selected.sort(key=lambda r:int(r['experiment'].rsplit('_',1)[1]))
        x=[int(r['experiment'].rsplit('_',1)[1]) for r in selected]
        for metric,color in [('auc','#577D9C'),('mcc','#B16F80')]:
            ax.plot(x,[float(r[metric]) for r in selected],'-o',label=metric.upper(),color=color,markersize=4)
        ax.set_xlabel(parameter.replace('_',' '));ax.set_xticks(x);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False);fig.tight_layout()
    save_figure(fig,'sensitivity')
    plt.close(fig)
    print(args.output)

if __name__=='__main__':
    main()
