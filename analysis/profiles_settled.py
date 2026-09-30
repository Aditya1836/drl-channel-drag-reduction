import numpy as np, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','pdf.fonttype':42,'font.size':12,'axes.labelsize':14,
    'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
NU=4.7679e-5; UB=0.1335
SUB='/proj/liu2/msc_thesis_aditya/Submission/Backward_Ctrl_Uctrl/results'
DRL='/proj/liu2/msc_thesis_aditya/DRL/Stats2_b102/channel/run0_ep0/4000'
PAIRS=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]      # xx xy xz yy yz zz

def read_field(path):
    txt=open(path).read()
    m=re.search(r'internalField\s+nonuniform\s+List<(scalar|vector|symmTensor)>\s*\n(\d+)\s*\n\(',txt)
    kind,n=m.group(1),int(m.group(2)); s=m.end(); e=txt.index(')\n;',s)
    a=np.fromstring(txt[s:e].replace('(',' ').replace(')',' '),sep=' ')
    return a.reshape(n,{'scalar':1,'vector':3,'symmTensor':6}[kind])

def wall_tau(tdir):
    txt=open(f'{tdir}/wallShearStressMean').read()
    m=re.search(r'wall\s*\{[^}]*?value\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(',txt,re.S)
    n=int(m.group(1)); body=txt[m.end():txt.index(';',m.end())]
    return np.fromstring(body.replace('(',' ').replace(')',' '),sep=' ')[:3*n].reshape(n,3)

def cell_stats(tdir,ydir=None):
    return (read_field(f'{ydir or tdir}/Cy')[:,0], read_field(f'{tdir}/UMean'),
            read_field(f'{tdir}/UPrime2Mean'), wall_tau(tdir))

def window(base,t1,t2):
    y,M1,R1,w1=cell_stats(f'{base}/{t1}',f'{base}/{t2}'); _,M2,R2,w2=cell_stats(f'{base}/{t2}')
    a,b=float(t1),float(t2)
    M=(b*M2-a*M1)/(b-a)
    UU1=np.stack([R1[:,k]+M1[:,i]*M1[:,j] for k,(i,j) in enumerate(PAIRS)],1)
    UU2=np.stack([R2[:,k]+M2[:,i]*M2[:,j] for k,(i,j) in enumerate(PAIRS)],1)
    UU=(b*UU2-a*UU1)/(b-a)
    R=np.stack([UU[:,k]-M[:,i]*M[:,j] for k,(i,j) in enumerate(PAIRS)],1)
    return y,M,R,(b*w2-a*w1)/(b-a)

def profiles(y,M,R,tw):
    yl=np.unique(np.round(y,10)); p={k:np.zeros(len(yl)) for k in ['U','uu','vv','ww','uv']}
    for j,yy in enumerate(yl):
        m=np.isclose(y,yy)
        p['U'][j]=M[m,0].mean(); p['uu'][j]=R[m,0].mean(); p['uv'][j]=R[m,1].mean()
        p['vv'][j]=R[m,3].mean(); p['ww'][j]=R[m,5].mean()
    return yl,p,np.sqrt(abs(tw[:,0].mean()))

data={'Uncontrolled':profiles(*window(f'{SUB}/uncontrolled','4000','10000')),
      'Opposition':  profiles(*window(f'{SUB}/controlled','4000','10000')),
      'DRL':         profiles(*cell_stats(DRL))}
ut0=data['Uncontrolled'][2]
print('windows: Uncontrolled and Opposition [4000,10000];  DRL [500,4000]\n')
for name,(yl,p,ut) in data.items():
    tot=(-p['uv']+NU*np.gradient(p['U'],yl))/ut**2
    print(f'{name:13s} u_tau={ut:.6f} Re_tau={ut/NU:.1f} Cf={2*(ut/UB)**2:.5f} '
          f'DR={(1-(ut/ut0)**2)*100:5.2f} %  stress-balance max dev={np.max(np.abs(tot-(1-yl))):.3f}')
for name,(yl,p,ut) in data.items():
    yp=yl*ut/NU
    print(f'\n{name}: U+ at y+=100: {np.interp(100,yp,p["U"]/ut):.2f}')
    for c,lab in [('uu',"u'rms"),('vv',"v'rms"),('ww',"w'rms")]:
        r=np.sqrt(np.maximum(p[c],0)); j=np.argmax(r)
        print(f'  {lab}: peak {r[j]/ut:.3f} (own) = {r[j]/ut0:.3f} (unc u_tau) at y+={yp[j]:.1f}, y/h={yl[j]:.3f}')
    uv=-p['uv']; j=np.argmax(uv)
    print(f"  -<u'v'>: peak {uv[j]/ut**2:.3f} (own) = {uv[j]/ut0**2:.3f} (unc u_tau) at y+={yp[j]:.1f}, y/h={yl[j]:.3f}")

col={'Uncontrolled':'#7a8ca8','Opposition':'#e8734a','DRL':'#26315e'}
fig,ax=plt.subplots(figsize=(6,4.5),constrained_layout=True)
yr=np.logspace(0,np.log10(180),100)
ax.plot(yr[yr<12],yr[yr<12],'k:',lw=1,label=r'$U^+{=}y^+$')
ax.plot(yr[yr>25],np.log(yr[yr>25])/0.41+5.2,'k--',lw=1,label='log law')
for n,(yl,p,ut) in data.items(): ax.semilogx(yl*ut/NU,p['U']/ut,color=col[n],lw=1.8,label=n)
ax.set_xlabel(r'$y^+$'); ax.set_ylabel(r'$U^+$'); ax.set_xlim(0.8,200); ax.legend(fontsize=10)
fig.savefig('profiles_Uplus_settled.pdf'); fig.savefig('profiles_Uplus_settled.png',dpi=300)

fig,ax=plt.subplots(figsize=(6,4.5),constrained_layout=True)
ls={'uu':'-','vv':'--','ww':':'}
for n,(yl,p,ut) in data.items():
    for c,l in ls.items(): ax.plot(yl*ut/NU,np.sqrt(np.maximum(p[c],0))/ut,l,color=col[n],lw=1.6)
for c,l in ls.items(): ax.plot([],[],l,color='k',label={'uu':r"$u'^+_{rms}$",'vv':r"$v'^+_{rms}$",'ww':r"$w'^+_{rms}$"}[c])
for n in data: ax.plot([],[],'-',color=col[n],label=n)
ax.set_xlabel(r'$y^+$'); ax.set_ylabel(r'rms$^+$'); ax.set_xlim(0,180); ax.legend(fontsize=9,ncol=2)
fig.savefig('profiles_rms_settled.pdf'); fig.savefig('profiles_rms_settled.png',dpi=300)

fig,ax=plt.subplots(figsize=(6,4.5),constrained_layout=True)
for n,(yl,p,ut) in data.items(): ax.plot(yl*ut/NU,-p['uv']/ut**2,color=col[n],lw=1.8,label=n)
ax.set_xlabel(r'$y^+$'); ax.set_ylabel(r"$-\langle u'v'\rangle^+$"); ax.set_xlim(0,180); ax.legend(fontsize=10)
fig.savefig('profiles_uv_settled.pdf'); fig.savefig('profiles_uv_settled.png',dpi=300)
print('\nsaved profiles_*_settled')
