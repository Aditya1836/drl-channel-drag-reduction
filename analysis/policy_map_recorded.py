import numpy as np, zipfile, io, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','pdf.fonttype':42,'font.size':12,'axes.labelsize':14,
    'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
z=zipfile.ZipFile('IC_BWD_9500/model/checkpoints/batch_102.zip')
sd=torch.load(io.BytesIO(z.read('policy.pth')),map_location='cpu')
W1=sd['actor.latent_pi.0.weight'].numpy(); b1=sd['actor.latent_pi.0.bias'].numpy()
W2=sd['actor.mu.weight'].numpy(); b2=sd['actor.mu.bias'].numpy()
d=np.loadtxt('Deter_BWD9500_b102/channel/run0_ep0/postProcessing/channelDRLControl/0/ActionState.dat',comments='#')
d=d[d[:,0]>300]; A=d[:,1::3]; e=(A-A.mean(axis=1,keepdims=True)).ravel()
u=d[:,2::3].ravel(); v=d[:,3::3].ravel(); off=A.mean()
ulo,uhi=np.percentile(u,[0.5,99.5]); vlo,vhi=np.percentile(v,[0.5,99.5])
B=100; rng=[[ulo,uhi],[vlo,vhi]]
N,ue,ve=np.histogram2d(u,v,bins=B,range=rng)
S,_,_=np.histogram2d(u,v,bins=B,range=rng,weights=e)
P,_,_=np.histogram2d(u,v,bins=B,range=rng,weights=(e>0).astype(float))
ok=N>=20
mean_e=np.where(ok,S/np.maximum(N,1),np.nan); share=np.where(ok,100*P/np.maximum(N,1),np.nan)
print(f'bins with at least 20 states: {ok.sum()} of {B*B}  ({ok.mean()*100:.1f} % of the map area)')
n=401; U,V=np.meshgrid(np.linspace(ulo,uhi,n),np.linspace(vlo,vhi,n))
EFF=(np.tanh(np.tanh(np.stack([U.ravel(),V.ravel()],1)@W1.T+b1)@W2.T+b2)[:,0]-off).reshape(n,n)
fig,ax=plt.subplots(1,2,figsize=(11,4.4),sharey=True,constrained_layout=True)
lim=np.nanpercentile(np.abs(mean_e),99)
c0=ax[0].pcolormesh(ue,ve,mean_e.T,cmap='RdBu_r',vmin=-lim,vmax=lim)
fig.colorbar(c0,ax=ax[0],pad=0.01,label='mean effective action')
c1=ax[1].pcolormesh(ue,ve,share.T,cmap='RdBu_r',vmin=0,vmax=100)
fig.colorbar(c1,ax=ax[1],pad=0.01,label='share of blowing [%]')
for a_,t in zip(ax,['(a) recorded effective action','(b) recorded share of blowing']):
    a_.contour(U,V,EFF,levels=[0.0],colors='k',linewidths=1.6)
    a_.set_xlabel(r"$u'^{+}$"); a_.set_title(t,fontsize=12); a_.set_facecolor('0.85')
ax[0].set_ylabel(r"$v'^{+}$")
fig.savefig('policy_map_recorded.pdf'); fig.savefig('policy_map_recorded.png',dpi=300)
print('saved policy_map_recorded')
