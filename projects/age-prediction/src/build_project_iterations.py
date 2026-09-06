"""Compose both complete research iterations without overwriting their source reports."""
from pathlib import Path
import re,json,shutil
R=Path(__file__).resolve().parents[1]; C=R/'cycles/fewshot'
original=(R/'cycles/iteration1-report.html').read_text();new=(C/'report.html').read_text()
base='/projects/age-prediction/'
caption=(R/'figures/design-review/caption.txt').read_text().strip()
fig=f'<figure class="fig"><a href="{base}figures/design-review/iteration1_integration.svg"><img src="{base}figures/design-review/iteration1_integration.svg" alt="Iteration 1: microbiome integration performance, repeat consistency and study heterogeneity" loading="lazy"></a><figcaption><strong>Iteration 1 / Integration analysis.</strong> {caption} <a href="{base}figures/design-review/iteration1_integration.pdf">PDF</a> / <a href="{base}figures/design-review/iteration1_integration.svg">SVG</a> / <a href="{base}figures/fig4_metadata_microbiome_integration.pdf">Original figure</a></figcaption></figure>'
original,n=re.subn(r'<figure class="fig">(?:(?!</figure>).)*fig4_metadata_microbiome_integration(?:(?!</figure>).)*</figure>',lambda m:fig,original,flags=re.S)
assert n==1,n
# Use editable vector assets at every display width.
for name in ['original','new']:
 val=locals()[name];val=re.sub(r'(<img src="[^"]+)\.png"',r'\1.svg"',val);locals()[name]=val
css='''.age-study .iteration-heading{font-family:Helvetica,Arial,sans-serif;font-size:25px;line-height:1.3;letter-spacing:-.02em;margin:64px 0 20px;padding-top:24px;border-top:2px solid #272727;scroll-margin-top:30px}.age-study .iteration-heading small{display:block;font-family:ui-monospace,monospace;font-size:12px;letter-spacing:.08em;color:#666;margin-bottom:12px;text-transform:uppercase}.age-study .iteration-nav{display:flex;gap:12px;flex-wrap:wrap;padding:18px 0;border-block:1px solid #d2d2ca;margin-bottom:30px;font-size:13px}.age-study figure{width:min(960px,calc(100vw - 70px));max-width:none;margin:30px 0!important;padding:14px!important;box-sizing:border-box;background:white}.age-study figure img{width:100%;height:auto;display:block}.age-study figcaption{max-width:850px;font-size:12px;line-height:1.6;margin-top:14px}.age-study h3::before,.age-study h4::before{content:none!important}.age-study .age-project>h3:first-child{margin-top:20px}.age-study .table-scroll{overflow-x:auto}@media(max-width:720px){.age-study figure{width:100%;padding:7px!important}.age-study .iteration-heading{font-size:22px}}'''
def clean_article(fragment):
    fragment=re.sub(r'<h3>Code and data availability</h3>.*?(?=<h3>|</div>\s*$)', '', fragment, flags=re.S|re.I)
    fragment=re.sub(r'<li>Roh et al\. Integration of Blood Methylome.*?</li>', '', fragment, flags=re.S)
    fragment=re.sub(r'<a\b[^>]*>\s*(?:PDF|SVG|Original figure)\s*</a>', '', fragment, flags=re.I)
    fragment=re.sub(r'<a\b[^>]*>(<img\b[^>]*>)</a>', r'\1', fragment)
    fragment=fragment.replace('\u00b7','/').replace('•','/')
    fragment=re.sub(r'(?:\s*/\s*)+(?=</figcaption>)','',fragment)
    fragment=re.sub(r'<h[234][^>]*>\s*(?:#|\.|\s)*</h[234]>','',fragment)
    return fragment
original=clean_article(original)
new=clean_article(new)

body=f'''<style>{css}</style><div class="age-study"><nav class="iteration-nav" aria-label="Research iterations"><a href="{base}#iteration-1">Iteration 1 / Blood methylation and microbiome</a><a href="{base}#iteration-2">Iteration 2 / Few-shot adaptation</a></nav><section id="iteration-1"><h2 class="iteration-heading"><small>Iteration 1</small>Blood methylation and microbiome integration</h2>{original}</section><section id="iteration-2"><h2 class="iteration-heading"><small>Iteration 2</small>Few-shot adaptation across methylation cohorts</h2><p>The second iteration investigates adaptation under limited target-cohort labels, following the multimodal experiments above. Both iterations retain their complete results, methods and reproducibility files.</p>{new}</section></div>'''
title='Molecular age prediction: multimodal integration and few-shot learning'
body=body.replace('\u00b7','/').replace('•','/')
(R/'report.html').write_text(body)
style=(C/'style.css').read_text()
(R/'index.html').write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} | Kun Hyung Roh</title><style>{style}</style></head><body><main><nav><a href="/#projects">← Projects</a> / <a href="/#age-prediction">Portfolio reader</a></nav><div class="reader-inner"><div class="meta">Kun Hyung Roh / September 2026</div><h1>{title}</h1><article class="article">{body}</article></div></main></body></html>')
meta={'order':3,'slug':'age-prediction','tag':'computational biology','title':title,'date':'Sep 2026','desc':'Two research iterations: blood methylation and microbial age prediction, followed by few-shot neural adaptation across external cohorts. Complete experiments, biological interpretation, model comparisons and reproducible analyses.','tech':['DNA methylation','microbiome','meta-learning','PyTorch'],'hero':'projects/age-prediction/figures/design-review/iteration1_integration.svg','heroCaption':'Iteration 1: microbial integration, repeated validation and source-study effects. Iteration 2 extends the project to few-shot epigenetic age prediction.','url':'/#age-prediction'}
(R/'meta.json').write_text(json.dumps(meta,indent=2))
print('Composed both full iterations:',body.count('<figure'), 'figures')
