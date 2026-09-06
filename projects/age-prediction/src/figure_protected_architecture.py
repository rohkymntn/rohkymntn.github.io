"""Original vector diagram; figures4papers typography and palette conventions."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import journal_style
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
R=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'Helvetica','font.size':12,'svg.fonttype':'none','pdf.fonttype':42})
f,a=plt.subplots(figsize=(14,5));a.set(xlim=(0,14),ylim=(0,5));a.axis('off')
def box(x,y,w,title,sub,c):
 a.add_patch(FancyBboxPatch((x,y),w,.85,boxstyle='square,pad=.03',fc=c,ec='#CFCECE',lw=1.2))
 a.text(x+w/2,y+.59,title,ha='center',va='center',weight='bold',fontsize=11);a.text(x+w/2,y+.23,sub,ha='center',va='center',color='#666666',fontsize=9)
def arrow(x,y,u,v,col='#0F4D92'):
 a.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=15,lw=1.6,color=col,shrinkA=3,shrinkB=4))
a.text(.1,4.62,'Clock-preserving microbial residual fusion',fontsize=14,weight='normal')
box(.15,2.75,2.1,'Methylation clock','Released Horvath score','#EAF0F8')
box(3.0,2.75,3.2,'Clock calibration','OLS / Huber / median regression','#EAF0F8');arrow(2.25,3.17,3.,3.17)
box(.15,1.1,2.1,'Gut genera','Hellinger representation','#EDF6F7')
box(3.0,1.1,3.2,'Clock orthogonalization',r'$H_{\perp}=H-\widehat{E}_{linear}[H\mid c]$','#EDF6F7');arrow(2.25,1.52,3,1.52,'#42949E')
box(6.9,1.1,3.0,'Residual kernel ridge','Linear / RBF / α / strength s','#DDF3DE');arrow(6.2,1.52,6.9,1.52,'#42949E')
a.add_patch(Circle((10.6,3.17),.23,fc='#0F4D92'));a.text(10.6,3.17,'+',color='white',fontsize=20,ha='center',va='center');arrow(6.2,3.17,10.37,3.17);arrow(9.9,1.52,10.5,2.92,'#42949E')
box(11.2,2.75,2.55,'Predicted age',r'$\hat y=f(c)+s\,g(H_{\perp})$','#EAF0F8');arrow(10.83,3.17,11.2,3.17)
a.text(8.1,2.54,r'Training target: $y-f(c)$',ha='center',fontsize=11,color='#42949E')
for ext in ['png','svg','pdf']:f.savefig(R/f'figures/protected_fusion_architecture.{ext}',dpi=300,bbox_inches='tight')
