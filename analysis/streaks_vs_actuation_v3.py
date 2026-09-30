import numpy as np, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','pdf.fonttype':42,'font.size':12,'axes.labelsize':14})
CASE='Deter_BWD9500_b102/channel/run0_ep0'; UTAU=0.00844

def internal(path, vec):
    L=open(path).read().split('\n')
    for i,l in enumerate(L):
        if l.startswith('internalField'):
            n=int(L[i+1]); rows=L[i+3:i+3+n]
            return np.array([r.strip('()').split() for r in rows],float) if vec else np.array(rows,float)

def wall_patch(path, ncomp):
    t=open(path).read(); t=t[t.index('boundaryField'):]
    blk=re.search(r'\n\s*wall\s*\{(.*?)\n\s*\}', t, re.S).group(1)
    m=re.search(r'value\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(', blk)
    n=int(m.group(1)); body=blk[m.end():].split(';')[0]
    a=np.fromstring(body.replace('(',' ').replace(')',' '),sep=' ')
    return a[:n*ncomp].reshape(n,ncomp)

xw=wall_patch(f'{CASE}/1200/Cx',1)[:,0]; zw=wall_patch(f'{CASE}/1200/Cz',1)[:,0]
vw=wall_patch(f'{CASE}/1200/U',3)[:,1]            # applied wall velocity, face order
print(f'wall faces {len(xw)}  v_w mean {vw.mean():+.2e}  rms {vw.std()/UTAU:.3f} u_tau')

U=internal(f'{CASE}/1200/U',True)
cx=internal(f'{CASE}/1200/Cx',False); cy=internal(f'{CASE}/1200/Cy',False); cz=internal(f'{CASE}/1200/Cz',False)
yl=np.unique(np.round(cy,6)); ysel=yl[np.argmin(np.abs(yl-0.08333))]
m=np.isclose(cy,ysel,atol=1e-6)
xp,zp=cx[m],cz[m]; up=U[m,0]-U[m,0].mean(); vp=U[m,1]-U[m,1].mean()
idx=np.array([np.argmin((xp-a)**2+(zp-b)**2) for a,b in zip(xw,zw)])
print(f"r(u' above vent, v_w) = {np.corrcoef(up[idx],vw)[0,1]:+.3f}")
print(f"r(v' above vent, v_w) = {np.corrcoef(vp[idx],vw)[0,1]:+.3f}")

last=[l for l in open(f'{CASE}/postProcessing/channelDRLControl/0/ActionState.dat') if not l.startswith('#')][-1]
acts=np.array(last.split(),float)[1::3]; az=acts-acts.mean()
print(f"old pairing r(u', log action) = {np.corrcoef(up[idx],acts)[0,1]:+.3f}")
print(f'same values? r(sorted log, sorted patch) = {np.corrcoef(np.sort(az),np.sort(vw))[0,1]:.5f}')
ia,ib=np.argsort(az),np.argsort(vw); perm=np.empty(len(az),int); perm[ia]=ib
uq=acts>-0.95
print(f'unsaturated columns already in face order: {np.mean(perm[uq]==np.arange(len(az))[uq])*100:.1f} %')

fig,(a1,a2)=plt.subplots(2,1,figsize=(10,6.5),sharex=True,sharey=True,constrained_layout=True)
upp=up/UTAU; lim=3.0
c1=a1.tricontourf(xp,zp,upp,levels=np.linspace(-lim,lim,25),cmap='RdBu_r',extend='both')
a1.set_ylabel('$z/h$'); a1.set_title(r"(a) $u'$ at $y^+\approx15$, $t=1200$",fontsize=12)
fig.colorbar(c1,ax=a1,pad=0.01,label=r"$u'/u_\tau$",ticks=[-3,-2,-1,0,1,2,3])
w=vw/UTAU; wl=1.5
c2=a2.tricontourf(xw,zw,w,levels=np.linspace(-wl,wl,25),cmap='RdBu_r',extend='both')
a2.set_xlabel('$x/h$'); a2.set_ylabel('$z/h$')
a2.set_title(r'(b) applied wall velocity $v_w/u_\tau$, $t=1200$',fontsize=12)
fig.colorbar(c2,ax=a2,pad=0.01,label=r'$v_w/u_\tau$',ticks=[-1.5,-1,-0.5,0,0.5,1,1.5])
fig.savefig('streaks_vs_actuation_v3.pdf'); fig.savefig('streaks_vs_actuation_v3.png',dpi=300)
print('saved streaks_vs_actuation_v2')
