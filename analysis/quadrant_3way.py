import numpy as np, re, os, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','pdf.fonttype':42,'font.size':12,'axes.labelsize':13,
    'xtick.direction':'in','ytick.direction':'in'})
NU=4.7679e-5; UT0=0.008278
SUB='/proj/liu2/msc_thesis_aditya/Submission/Backward_Ctrl_Uctrl/results'
BASE='/proj/liu2/msc_thesis_aditya/DRL/Stats2_b102/channel/run0_ep0/postProcessing/planesY15'
TIMES=['4000','5000','6000','7500','8500','9000','9500','10000']

def rd(p):
    t=open(p).read(); m=re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(',t)
    k,n=m.group(1),int(m.group(2)); s=m.end(); e=t.index(')\n;',s)
    return np.fromstring(t[s:e].replace('(',' ').replace(')',' '),sep=' ').reshape(n,{'scalar':1,'vector':3}[k])

def quad(u,v):
    uv=u*v
    return np.array([uv[(u>0)&(v>0)].sum(),uv[(u<0)&(v>0)].sum(),
                     uv[(u<0)&(v<0)].sum(),uv[(u>0)&(v<0)].sum()])/len(uv)

tds=sorted([d for d in os.listdir(BASE) if os.path.isdir(f'{BASE}/{d}')],key=float)
tds=[t for t in tds if float(t)>300][::5]
r=vtk.vtkXMLPolyDataReader(); uu=[]; vv=[]; yp=None
for td in tds:
    f=f'{BASE}/{td}/y15plane.vtp'
    if not os.path.exists(f): continue
    r.SetFileName(f); r.Update(); out=r.GetOutput()
    if yp is None: yp=float(vtk_to_numpy(out.GetPoints().GetData())[:,1].mean())
    U=vtk_to_numpy(out.GetPointData().GetArray('U'))
    uu.append(U[:,0]-U[:,0].mean()); vv.append(U[:,1]-U[:,1].mean())
q_drl=quad(np.concatenate(uu),np.concatenate(vv))
print(f'DRL plane height y={yp:.5f} (y+={yp*UT0/NU:.1f} in uncontrolled wall units), frames={len(uu)}, t={tds[0]}..{tds[-1]}')

def ref(case):
    y=rd(f'{SUB}/{case}/10000/Cy')[:,0]; yl=np.unique(np.round(y,10))
    j=np.searchsorted(yl,yp); ya,yb=yl[j-1],yl[j]; w=(yp-ya)/(yb-ya); qa=[]; qb=[]; used=[]
    for t in TIMES:
        try: U=rd(f'{SUB}/{case}/{t}/U')
        except Exception: continue
        used.append(t)
        for yy,st in [(ya,qa),(yb,qb)]:
            m=np.isclose(y,yy); st.append(quad(U[m,0]-U[m,0].mean(),U[m,1]-U[m,1].mean()))
    print(f'{case}: snapshots {used}, layers y={ya:.5f},{yb:.5f}, weight {w:.2f}')
    return (1-w)*np.mean(qa,0)+w*np.mean(qb,0)

q_unc=ref('uncontrolled'); q_opp=ref('controlled')
names=['Q1 (u+,v+)','Q2 ejection','Q3 (u-,v-)','Q4 sweep']
T0=abs(q_unc.sum())
print(f'\ntotal <uv>:  unc {q_unc.sum():+.3e}  opp {q_opp.sum():+.3e}  drl {q_drl.sum():+.3e}')
print(f'in uncontrolled wall units (-<uv>/u_tau^2): unc {-q_unc.sum()/UT0**2:.3f}  opp {-q_opp.sum()/UT0**2:.3f}  drl {-q_drl.sum()/UT0**2:.3f}')
print(f'total relative to uncontrolled: opp {q_opp.sum()/q_unc.sum():.3f}  drl {q_drl.sum()/q_unc.sum():.3f}\n')
print('quadrant contribution / |<uv>|_unc      unc      opp      drl')
for i,n in enumerate(names): print(f'{n:14s}                        {q_unc[i]/T0:+.3f}   {q_opp[i]/T0:+.3f}   {q_drl[i]/T0:+.3f}')

x=np.arange(4); w=0.26
fig,ax=plt.subplots(figsize=(7.4,4.2),constrained_layout=True)
for k,(q,lab,c) in enumerate([(q_unc,'Uncontrolled','#7a8ca8'),(q_opp,'Opposition','#e8734a'),(q_drl,'DRL','#26315e')]):
    ax.bar(x+(k-1)*w,q/T0,w,label=lab,color=c,edgecolor='k',lw=0.7)
ax.axhline(0,color='k',lw=0.8); ax.set_xticks(x); ax.set_xticklabels(names,fontsize=10)
ax.set_ylabel(r"quadrant contribution / $|\langle u'v'\rangle|_{unc}$")
ax.legend(frameon=True,edgecolor='0.3')
fig.savefig('quadrant_3way.pdf'); fig.savefig('quadrant_3way.png',dpi=300)
print('\nsaved quadrant_3way')
