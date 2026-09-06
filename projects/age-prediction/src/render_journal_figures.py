"""Regenerate every displayed figure under the common journal style."""
from pathlib import Path
import runpy,sys
R=Path(__file__).resolve().parents[1]
import journal_style
for name in ['figures.py','figure_architecture.py','figure_biology.py','figure_protected.py','figure_protected_architecture.py','summarize_hiseed.py','figure_design_review.py']:
 print('Rendering',name,flush=True)
 runpy.run_path(str(R/'src'/name),run_name='__main__')
print('Rendering few-shot plates',flush=True)
runpy.run_path(str(R/'cycles/fewshot/src/figures.py'),run_name='__main__')
