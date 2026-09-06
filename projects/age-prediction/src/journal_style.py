"""Shared Helvetica / final-publication-size rendering for portfolio figures.
Uses figures4papers colour conventions and Nature's 183 mm width / 5–7 pt text.
"""
import re
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib import font_manager
font_manager.findfont('Helvetica',fallback_to_default=False)
plt.rcParams.update({'font.family':'Helvetica','font.size':7,'axes.labelsize':7,'axes.titlesize':7,'legend.fontsize':6,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'custom','mathtext.rm':'Helvetica','mathtext.it':'Helvetica:italic','mathtext.bf':'Helvetica:bold','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6})
_save=Figure.savefig

def journal_save(self,*args,**kwargs):
 if not getattr(self,'_journal_styled',False):
  width,height=self.get_size_inches();factor=min((183/25.4)/width,(165/25.4)/height)
  self.set_size_inches(width*factor,height*factor)
  for text in self.findobj(match=Text):
   text.set_fontfamily('Helvetica');text.set_fontsize(max(5.5,min(7,text.get_fontsize()*factor)))
   text.set_text(text.get_text().replace('·','/').replace('•','/').replace('→',' / '))
  from matplotlib.patches import Patch, FancyArrowPatch
  for patch in self.findobj(match=Patch):
   patch.set_linewidth(max(.35,min(.7,patch.get_linewidth()*factor)))
   if isinstance(patch,FancyArrowPatch): patch.set_mutation_scale(patch.get_mutation_scale()*factor)
  for ax in self.axes:
   for spine in ax.spines.values():spine.set_linewidth(.6)
   ax.tick_params(width=.6,length=2.5,pad=2)
   for line in ax.lines:
    line.set_linewidth(max(.5,min(1,line.get_linewidth()*factor)))
    line.set_markersize(line.get_markersize()*factor)
   for collection in ax.collections:
    if hasattr(collection,'get_sizes'): collection.set_sizes(collection.get_sizes()*factor**2)
    collection.set_linewidths([max(.4,min(.8,w*factor)) for w in collection.get_linewidths()])
  self._journal_styled=True
 kwargs.update(dpi=450,facecolor='white')
 return _save(self,*args,**kwargs)
Figure.savefig=journal_save
