"""Architecture figure using figures4papers palette, typography and vector conventions.
Reference: ChenLiu-1996/figures4papers, commit 565e6b97a9609e14ac07bee83dcb94589034fe27.
Original diagram code; no third-party scientific results or artwork are copied.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import journal_style
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle,Rectangle
ROOT=Path(__file__).resolve().parents[1]
P={'blue':'#0F4D92','secondary':'#3775BA','green':'#DDF3DE','teal':'#42949E','rose':'#F6CFCB','gray':'#CFCECE','ink':'#272727','muted':'#767676'}
plt.rcParams.update({'font.family':'Helvetica','font.size':15,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(14.4,7.8),facecolor='white');axes=fig.subplots(2,1)
fig.subplots_adjust(left=.035,right=.985,top=.95,bottom=.06,hspace=.34)

def box(a,x,y,w,h,title,detail,fill='white',edge=None):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='square,pad=0.014',facecolor=fill,edgecolor=edge or P['gray'],linewidth=1.4))
 a.text(x+w/2,y+h*.67,title,ha='center',va='center',fontweight='bold',fontsize=14,color=P['ink'])
 a.text(x+w/2,y+h*.29,detail,ha='center',va='center',fontsize=11.5,color=P['muted'],linespacing=1.4)
def arrow(a,start,end,c=None,rad=0):
 a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=15,lw=1.7,color=c or P['blue'],connectionstyle=f'arc3,rad={rad}',shrinkA=2,shrinkB=3))
def label(a,letter,title,subtitle):
 a.set(xlim=(0,14),ylim=(0,3.15));a.axis('off')
 a.text(0,3.0,letter,fontweight='bold',fontsize=22,color=P['ink'])
 a.text(.45,3.03,title,fontweight='normal',fontsize=14,color=P['ink'])
def matrix(a,x,y,c):
 for i in range(4):
  for j in range(5):
   a.add_patch(Rectangle((x+j*.085,y+i*.10),.066,.08,facecolor=c,alpha=.18+.13*((i+2*j)%5),lw=0))

a=axes[0];label(a,'a','Metadata-adjusted microbial age prediction','Evaluated using three repeated nested participant-held-out validations')
box(a,.15,1.63,2.3,.70,'Study / site / sex','Same skin participants',fill='#F4F4F4')
box(a,.15,.45,2.3,.82,'Skin microbiome','1,798 samples / 339 IDs',fill='#EDF6F7',edge=P['teal'])
box(a,3.0,1.63,2.7,.70,'Metadata regression','One-hot encoding → ridge',fill='#F4F4F4')
box(a,3.0,.45,2.7,.82,'Microbial preprocessing','Hellinger + prevalence filter',fill='#EDF6F7',edge=P['teal'])
box(a,6.30,.45,2.7,.82,'Residual regression','Random forest / 160 trees',fill=P['green'],edge=P['teal'])
arrow(a,(2.45,1.98),(3,1.98));arrow(a,(2.45,.86),(3,.86),P['teal']);arrow(a,(5.7,.86),(6.3,.86),P['teal'])
a.text(7.6,1.75,r'$r_{train}=y_{train}-\hat{y}_{metadata,train}$',ha='center',fontsize=14,color=P['blue'])
arrow(a,(5.7,1.98),(9.7,1.98));arrow(a,(9,.86),(9.7,1.72),P['teal'])
a.add_patch(Circle((10,1.85),.27,facecolor=P['blue'],edgecolor='white',lw=1.5));a.text(10,1.85,'+',color='white',fontsize=22,ha='center',va='center')
arrow(a,(10.27,1.85),(10.75,1.85))
box(a,10.8,1.39,2.85,.94,'Combined age estimate',r'$\hat{y}=\hat{y}_{metadata}+\hat{r}_{microbiome}$',fill='#EAF0F8',edge=P['blue'])
a=axes[1];label(a,'b','Blood–skin prediction-level fusion','Fitted component models; joint accuracy requires paired test specimens')
box(a,.15,1.63,2.3,.70,'Blood methylation','27,578 measured CpGs',fill='#EAF0F8',edge=P['blue'])
box(a,.15,.45,2.3,.82,'Skin microbiome','7,311 input ASVs',fill='#EDF6F7',edge=P['teal'])
box(a,3,1.63,2.7,.70,'CpG preprocessing','Impute → select → scale',fill='#EAF0F8',edge=P['blue'])
box(a,3,.45,2.7,.82,'Microbial preprocessing','Hellinger + prevalence filter',fill='#EDF6F7',edge=P['teal'])
box(a,6.3,1.63,2.7,.70,'Blood age predictor','Ridge / 500 CpGs / α = 100',fill='#EAF0F8',edge=P['blue'])
box(a,6.3,.45,2.7,.82,'Skin age predictor','Random forest / 160 trees',fill='#EDF6F7',edge=P['teal'])
for y,c in [(1.98,P['blue']),(.86,P['teal'])]:
 arrow(a,(2.45,y),(3,y),c);arrow(a,(5.7,y),(6.3,y),c);arrow(a,(9,y),(10.40,1.47),c)
box(a,10.45,1.0,3.2,.95,'Late-fusion estimate',r'$\hat{y}=0.5\hat{y}_{blood}+0.5\hat{y}_{skin}$',fill=P['green'],edge=P['blue'])
for fmt in ['png','pdf','svg']:fig.savefig(ROOT/f'figures/architecture.{fmt}',dpi=400,bbox_inches='tight',pad_inches=.08)
plt.close(fig)
