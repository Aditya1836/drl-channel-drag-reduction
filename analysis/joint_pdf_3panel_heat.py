import numpy as np, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'serif','pdf.fonttype':42,'axes.grid':False,
    'font.size':12,'axes.labelsize':15,'xtick.labelsize':11,'ytick.labelsize':11,
    'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})

UREF=0.09085; UTAU=0.00844
TIMES=['1000','2000','3000','4000','5000','6000','7500','8500','9000','9500','10000']

def read_internal(path):
    txt=open(path).read()
    m=re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', txt)
    kind,n=m.group(1),int(m.group(2))
    s=m.end(); e=txt.index(')\n;',s)
    a=np.fromstring(txt[s:e].replace('(',' ').replace(')',' '),sep=' ')
    return a.reshape(n,{'scalar':1,'vector':3}[kind])

def ref_states(base):
    y=read_internal(f'{base}/10000/Cy')[:,0]
    ylev=np.unique(np.round(y,10))
    ysel=ylev[np.argmin(np.abs(ylev-0.08333))]
    m=np.isclose(y,ysel)
    uu,vv=[],[]
    for t in TIMES:
        try: U=read_internal(f'{base}/{t}/U')
        except Exception: continue
        uu.append(U[m,0]); vv.append(U[m,1])
    return (np.concatenate(uu)-UREF)/UTAU, np.concatenate(vv)/UTAU

SUB='/proj/liu2/msc_thesis_aditya/Submission/Backward_Ctrl_Uctrl/results'
u0,v0=ref_states(f'{SUB}/uncontrolled')
u1,v1=ref_states(f'{SUB}/controlled')
d=np.loadtxt('Deter_BWD9500_b102/channel/run0_ep0/postProcessing/channelDRLControl/0/ActionState.dat',comments='#')
msk=d[:,0]>300
u2=d[msk,2::3].ravel(); v2=d[msk,3::3].ravel()

rng=[[-7,7],[-1.5,1.5]]; BINS=80
fig,axes=plt.subplots(1,3,figsize=(13.5,3.8),sharex=True,sharey=True,constrained_layout=True)
for ax,(u,v),lab in [(axes[0],(u0,v0),'(a) uncontrolled'),
                     (axes[1],(u1,v1),'(b) opposition control'),
                     (axes[2],(u2,v2),'(c) DRL control')]:
    H,ue,ve=np.histogram2d(u,v,bins=BINS,range=rng,density=True)
    pc=ax.pcolormesh(ue,ve,H.T/H.max(),cmap='inferno',vmin=0,vmax=1,shading='flat')
    ax.axhline(0,color='w',lw=0.4,alpha=0.4); ax.axvline(0,color='w',lw=0.4,alpha=0.4)
    ax.set_xlabel(r"$u'^{+}$"); ax.set_title(lab,fontsize=13)
axes[0].set_ylabel(r"$v'^{+}$")
cb=fig.colorbar(pc,ax=axes,shrink=0.9,pad=0.012)
cb.set_label('PDF / PDF$_{max}$')
fig.savefig('joint_pdf_3panel_heat.pdf', dpi=600); fig.savefig('joint_pdf_3panel_heat.png',dpi=600)
print('saved joint_pdf_3panel_heat')
