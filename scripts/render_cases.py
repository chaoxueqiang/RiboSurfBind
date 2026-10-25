#!/usr/bin/env python3
"""Re-render the four published cases with native PyMOL surfaces and sticks.

Scenes embed real structures and the original camera. Per-residue colors are
reapplied from audited predictions, not inferred from apparent surface area.
Run this script in the optional PyMOL environment.
"""
import argparse
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/reproduced_cases')
    parser.add_argument('--width',type=int,default=1000)
    parser.add_argument('--validate-only',action='store_true',help='Check scenes/cameras/colors without ray tracing')
    args=parser.parse_args()
    from pymol import cmd
    cases=json.loads((ROOT/'configs/cases.json').read_text())
    rows=list(csv.DictReader((ROOT/'results/cases/inputs/residue_predictions.csv').open(encoding='utf-8-sig')))
    rgb={'TP':[.09,.80,.09],'FP':[.80,.47,.84],'FN':[.20,.20,.87],'TN':[.8,.8,.8],'ligand':[.96,.83,.16]}
    args.output.mkdir(parents=True,exist_ok=True)
    report=[]
    for case in cases:
        name=case['name'];selected=[r for r in rows if r['rna']==name]
        view=None
        for model in ['ATT','Surface','Full']:
            cmd.reinitialize()
            cmd.load(str(ROOT/'results/cases/scenes'/f'{name}_{model}.pse'))
            if view is None:
                view=cmd.get_view()
            cmd.set_view(view)
            for color,value in rgb.items():
                cmd.set_color(color,value)
            cmd.color('TN','rna')
            for outcome in ['TP','FP','FN']:
                parts=[]
                for row in selected:
                    if row[model+'_outcome']==outcome:
                        chain,number,insert=row['residue_key'].split(':')
                        parts.append(f'(chain {chain} and resi {number}{insert})')
                if parts:
                    cmd.color(outcome,'rna and ('+' or '.join(parts)+')')
            cmd.hide('everything','all');cmd.show('surface','rna');cmd.show('sticks','rna')
            cmd.set('transparency',.28,'rna');cmd.set('surface_color',-1,'rna')
            cmd.set('stick_radius',.13,'rna');cmd.show('sticks','ligand')
            cmd.color('ligand','ligand');cmd.set('stick_radius',.23,'ligand')
            cmd.set('max_threads',2);cmd.bg_color('white')
            observed=[]
            cmd.iterate('rna','observed.append((chain,resi,color))',space={'observed':observed})
            color_ids={cmd.get_color_index(k):k for k in ['TP','FP','FN','TN']}
            actual={(chain,resi):color_ids[color] for chain,resi,color in observed}
            expected={}
            for row in selected:
                chain,number,insert=row['residue_key'].split(':')
                expected[(chain,number+insert)]=row[model+'_outcome']
            if actual!=expected:
                raise AssertionError((name,model,'structure/color mapping differs from predictions'))
            if not args.validate_only:
                cmd.png(str(args.output/f'{name}_{model}.png'),width=args.width,height=round(args.width*1.05),dpi=300,ray=1)
            report.append(dict(case=name,model=model,residue_colors_verified=True,view=list(view)))
            print(name,model,'PASS',flush=True)
    (args.output/'render_audit.json').write_text(json.dumps(report,indent=2))
    if args.validate_only:
        return
    # Figure labels contain names only; performance numbers remain in CSV.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for index,pair in enumerate([cases[:2],cases[2:]],1):
        fig,axes=plt.subplots(2,3,figsize=(9,6.8))
        for i,case in enumerate(pair):
            for j,(model,title) in enumerate([('ATT','ATT-only'),('Surface','Surface-only'),('Full','RiboSurfBind')]):
                axes[i,j].imshow(plt.imread(args.output/f"{case['name']}_{model}.png"))
                axes[i,j].axis('off');axes[i,j].set_xlabel(case['name'])
                axes[i,j].text(.5,-.025,case['name'],ha='center',transform=axes[i,j].transAxes)
                if i==0:
                    axes[i,j].set_title(title)
        fig.tight_layout()
        fig.savefig(args.output/f'cases_{index}.png',dpi=300,bbox_inches='tight')
        fig.savefig(args.output/f'cases_{index}.pdf',bbox_inches='tight')
        plt.close(fig)

if __name__=='__main__':
    main()
