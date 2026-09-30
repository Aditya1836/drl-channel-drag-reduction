import numpy as np, re
CASE='Deter_BWD9500_b102/channel/run0_ep0'; UTAU=0.00844; UB=0.1335; RHO=1.0
Cf_unc=0.00765; DR=0.3114; Cf_drl=Cf_unc*(1-DR)

def wall_patch(path,ncomp):
    t=open(path).read(); t=t[t.index('boundaryField'):]
    blk=re.search(r'\n\s*wall\s*\{(.*?)\n\s*\}',t,re.S).group(1)
    m=re.search(r'value\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(',blk)
    n=int(m.group(1)); body=blk[m.end():].split(';')[0]
    return np.fromstring(body.replace('(',' ').replace(')',' '),sep=' ')[:n*ncomp].reshape(n,ncomp)

vw=wall_patch(f'{CASE}/1200/U',3)[:,1]
def internal(path,vec):
    L=open(path).read().split('\n')
    for i,l in enumerate(L):
        if l.startswith('internalField'):
            n=int(L[i+1]); rows=L[i+3:i+3+n]
            return np.array([r.strip('()').split() for r in rows],float) if vec else np.array(rows,float)
pc=internal(f'{CASE}/1200/p',False); cx=internal(f'{CASE}/1200/Cx',False); cy=internal(f'{CASE}/1200/Cy',False); cz=internal(f'{CASE}/1200/Cz',False)
xw=wall_patch(f'{CASE}/1200/Cx',1)[:,0]; zw=wall_patch(f'{CASE}/1200/Cz',1)[:,0]
m=np.isclose(cy,cy.min()); x1,z1,p1=cx[m],cz[m],pc[m]
idx=np.array([np.argmin((x1-a)**2+(z1-b)**2) for a,b in zip(xw,zw)])
p=p1[idx]; pp=p-p.mean()
print(f'p taken from first cell layer at y={cy.min():.5f} (zeroGradient wall), {m.sum()} cells')
P_pw=np.mean(pp*vw)
print(f'pressure term from 1200 snapshot: <p v_w>={np.mean(p*vw):.3e}  <p\' v_w>={P_pw:.3e}  (p mean {p.mean():.3e}, v_w mean {vw.mean():.1e})')

d=np.loadtxt(f'{CASE}/postProcessing/channelDRLControl/0/ActionState.dat',comments='#')
a=d[d[:,0]>300,1::3]; v=(a-a.mean(axis=1,keepdims=True))*UTAU
P_ke=0.5*RHO*np.mean(np.abs(v)**3)
half=0.5*RHO*UB**3; P_S=(Cf_unc-Cf_drl)*half; P_pump=Cf_unc*half; P_I=P_pw+P_ke
print(f'Cf_unc={Cf_unc}  Cf_drl={Cf_drl:.5f}  P_pump={P_pump:.3e}')
print(f'P_S={P_S:.3e}  P_ke={P_ke:.3e}  P_pw={P_pw:.3e}  P_I={P_I:.3e}  P_S/P_I={P_S/P_I:.1f}')
print(f'gross={P_S/P_pump*100:.1f} %  cost={P_I/P_pump*100:.1f} %  net={(P_S-P_I)/P_pump*100:.1f} %')

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','pdf.fonttype':42,'font.size':12,'axes.labelsize':13,
    'xtick.direction':'in','ytick.direction':'in','ytick.right':True})
fig,axes=plt.subplots(1,2,figsize=(9.6,4.0),constrained_layout=True)
ax=axes[0]; vals=[P_S,P_I,P_pw,P_ke]
labs=['$P_S$\n(saved)','$P_I$\n(input)',r"$\langle p' v_w\rangle$",r"$\frac{1}{2}\langle |v_w|^3\rangle$"]
bars=ax.bar(range(4),vals,0.6,color=['#26315e','#c23b22','#e8734a','#f2b134'],edgecolor='k',lw=0.8)
ax.set_yscale('log'); ax.set_ylim(1e-9,1e-5); ax.set_xticks(range(4)); ax.set_xticklabels(labs,fontsize=10)
ax.set_ylabel('power per unit wall area')
for b,vv in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,vv*1.25,f'{vv:.2e}',ha='center',fontsize=8.5)
ax.text(0.97,0.95,f'$P_S/P_I = {P_S/P_I:.0f}$',transform=ax.transAxes,ha='right',va='top',fontsize=13,bbox=dict(boxstyle='round',fc='w',ec='0.4'))
ax.text(0.02,0.97,'(a)',transform=ax.transAxes,fontsize=12,fontweight='bold',va='top')
ax=axes[1]; gross=P_S/P_pump*100; net=(P_S-P_I)/P_pump*100
bars=ax.bar([0,1],[gross,net],0.5,color=['#7a8ca8','#26315e'],edgecolor='k',lw=0.8)
ax.set_xticks([0,1]); ax.set_xticklabels(['gross saving\n(drag reduction)','net saving\n(minus actuation cost)'],fontsize=10)
ax.set_ylabel('% of uncontrolled pumping power'); ax.set_ylim(0,35)
for b,vv in zip(bars,[gross,net]): ax.text(b.get_x()+b.get_width()/2,vv+0.6,f'{vv:.1f}%',ha='center',fontsize=11)
ax.text(0.02,0.97,'(b)',transform=ax.transAxes,fontsize=12,fontweight='bold',va='top')
fig.savefig('power_budget_deter.pdf'); fig.savefig('power_budget_deter.png',dpi=300); print('saved power_budget_deter')
