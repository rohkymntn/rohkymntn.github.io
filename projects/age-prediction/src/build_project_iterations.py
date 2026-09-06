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

def formalize_iteration1(fragment):
    replacements = [
        (r'<h4>Paired methylation-clock and gut microbiome pilot</h4><p>.*?</p>',
         '<h4>Early integration degrades prediction in a paired methylation-clock and gut-microbiome cohort</h4><p>The paired analysis used the HI-SEED <a href="https://doi.org/10.6084/m9.figshare.26487670">release</a>, comprising 123 participants with concordant identifiers across metadata and microbial abundance tables. Across three repeated five-fold validations, calibrated Horvath scores achieved an MAE of 4.87 years, genus-level microbial profiles achieved 12.59 years and early feature integration achieved 9.53 years. Relative to calibrated methylation scores, integration increased MAE by 4.66 years (95% conditional interval, 3.35–6.03). The analysis uses a pretrained methylation-clock score rather than raw methylation measurements and has not been externally validated.</p>'),
        (r'<h4>Clock-preserving residual fusion limits loss of predictive accuracy</h4><p>.*?</p><p>.*?</p>',
         '<h4>Clock-preserving residual integration recovers unimodal predictive accuracy</h4><p>A subsequent formulation retained an unpenalized clock-calibration term and estimated an independent correction from clock-orthogonalized microbial features. Across identical outer partitions, residual integration achieved an MAE of 4.848 years, compared with 4.851 years for inner-selected clock calibration and 4.870 years for ordinary least-squares calibration. Relative to early feature integration (MAE, 9.529 years), the post hoc paired reduction was 4.68 years (95% conditional interval, 3.42–6.07), equivalent to a 49.1% reduction.</p><p>The prespecified contrast with selected clock calibration was 0.004 years (95% conditional interval, −0.061 to 0.070). Although a microbial correction was selected in 12 of 15 outer training partitions, it did not produce a measurable held-out improvement. Residual integration therefore preserved the predictive accuracy of the methylation component but did not establish independent microbial complementarity.</p>'),
        (r'<h4>Exploratory pathway analysis</h4><p>.*?</p>',
         '<h4>Probe-aware pathway analysis identifies no FDR-significant enrichment</h4><p>The 500 selected methylation probes mapped to 472 unique genes. Reactome analysis tested 1,333 pathways against a background of 14,475 genes represented on the measured array. No pathway passed Benjamini–Hochberg correction at FDR &lt; 0.05. The nominally ranked pathways in Figure 3 are therefore presented as descriptive annotations rather than pathway-level evidence of biological ageing. Complete results are provided in the <a href="/projects/age-prediction/results/reactome_probe_aware_enrichment.csv">enrichment table</a>.</p>'),
        (r'<h3>Discussion</h3>.*?(?=<p>For component errors)',
         '<h3>Discussion</h3><p>Skin microbial features contributed predictive information beyond study, body site and sex in repeated participant-held-out validation. The aggregate improvement was reproducible across three partition seeds but heterogeneous across source studies, with the largest cohort accounting for most of the observed reduction. The result therefore supports conditional predictive value within the sampled study mixture rather than study-independent transportability.</p><p>The paired HI-SEED analysis further showed that multimodal performance depends on model formulation. Direct concatenation degraded a calibrated methylation predictor, whereas residual integration preserved its accuracy by restricting microbial features to a correction term. The absence of improvement over clock calibration indicates that model architecture alone cannot establish complementary biological information.</p>'),
        (r'<p>The unpaired cohorts do not identify.*?</p>\s*<p>Additional limitations.*?</p>',
         '<p>The unpaired blood and skin cohorts do not identify the within-person covariance or joint bias terms. Consequently, their component errors cannot be used to infer empirical blood–skin fusion performance. Evaluation of this model class requires paired specimens, prespecified weight estimation in training or calibration data and comparison with both unimodal components in an untouched test cohort.</p><p>Additional limitations include dependence on processed public matrices, ambiguous identifiers in one excluded microbiome study, restricted age coverage in the external methylation cohort and unresolved assay and study effects. The outcome is chronological age; no inference concerning biological age or clinical utility is supported.</p>'),
    ]
    for pattern, replacement in replacements:
        fragment, count = re.subn(pattern, replacement, fragment, count=1, flags=re.S)
        assert count == 1, pattern
    return fragment
original=clean_article(original)
new=clean_article(new)
original=formalize_iteration1(original)

body=f'''<style>{css}</style><div class="age-study"><nav class="iteration-nav" aria-label="Research iterations"><a href="{base}#iteration-1">Iteration 1 / Blood methylation and microbiome</a><a href="{base}#iteration-2">Iteration 2 / Few-shot adaptation</a></nav><section id="iteration-1"><h2 class="iteration-heading"><small>Iteration 1</small>Blood methylation and microbiome integration</h2>{original}</section><section id="iteration-2"><h2 class="iteration-heading"><small>Iteration 2</small>Few-shot adaptation across methylation cohorts</h2><p>This analysis examines adaptation of DNA-methylation age predictors under restricted target-cohort supervision. It extends the preceding multimodal study by evaluating representation learning and support-set model selection across independent blood-derived methylation cohorts.</p>{new}</section></div>'''
title='Molecular age prediction: multimodal integration and few-shot learning'
body=body.replace('\u00b7','/').replace('•','/')
(R/'report.html').write_text(body)
style=(C/'style.css').read_text()
(R/'index.html').write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} | Kun Hyung Roh</title><style>{style}</style></head><body><main><nav><a href="/#projects">← Projects</a> / <a href="/#age-prediction">Portfolio reader</a></nav><div class="reader-inner"><div class="meta">Kun Hyung Roh / September 2026</div><h1>{title}</h1><article class="article">{body}</article></div></main></body></html>')
meta={'order':3,'slug':'age-prediction','tag':'computational biology','title':title,'date':'Sep 2026','desc':'Evaluation of DNA-methylation and microbial age predictors, residual multimodal integration and few-shot adaptation across external cohorts.','tech':['DNA methylation','microbiome','meta-learning','PyTorch'],'hero':'projects/age-prediction/figures/design-review/iteration1_integration.svg','heroCaption':'Participant-held-out evaluation of microbial integration, repeated-validation stability and source-study heterogeneity.','url':'/#age-prediction'}
(R/'meta.json').write_text(json.dumps(meta,indent=2))
print('Composed both full iterations:',body.count('<figure'), 'figures')
