import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile, io, torch

plt.rcParams.update({'font.family':'serif','pdf.fonttype':42,
    'font.size':12,'axes.labelsize':14,'xtick.labelsize':11,'ytick.labelsize':11,
    'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})

# ---- Data 1: the 33 numbers ----
z = zipfile.ZipFile('IC_BWD_9500/model/checkpoints/batch_102.zip')
sd = torch.load(io.BytesIO(z.read('policy.pth')), map_location='cpu')
W1 = sd['actor.latent_pi.0.weight'].numpy(); b1 = sd['actor.latent_pi.0.bias'].numpy()
W2 = sd['actor.mu.weight'].numpy();          b2 = sd['actor.mu.bias'].numpy()

# ---- Data 2 & 3: offset and bounds from the deterministic run's log ----
f = 'Deter_BWD9500_b102/channel/run0_ep0/postProcessing/channelDRLControl/0/ActionState.dat'
acts, us, vs = [], [], []
for line in open(f):
    if line.startswith('#'): continue
    r = np.array(line.split(), float)
    if r[0] <= 300: continue
    acts.append(r[1::3]); us.append(r[2::3]); vs.append(r[3::3])
acts = np.concatenate(acts); us = np.concatenate(us); vs = np.concatenate(vs)
offset = acts.mean()
u_lo, u_hi = np.percentile(us, [0.5, 99.5])
v_lo, v_hi = np.percentile(vs, [0.5, 99.5])
print(f'log entries: {acts.size} actions | offset (mean action) = {offset:+.4f}')
print(f'bounds: u in [{u_lo:.2f}, {u_hi:.2f}]   v in [{v_lo:.2f}, {v_hi:.2f}]')

# ---- the sweep: 601x601 pairs through the frozen 33 ----
n = 601
u = np.linspace(u_lo, u_hi, n); v = np.linspace(v_lo, v_hi, n)
U, V = np.meshgrid(u, v)
S = np.stack([U.ravel(), V.ravel()], axis=1)
MU = np.tanh((np.tanh(S @ W1.T + b1)) @ W2.T + b2).reshape(n, n)
EFF = MU - offset            # subtract the average (offset is negative -> adds 0.68)
OPP = -V                     # panel (a): opposition, phi = -v'

# verification
s = np.array([0.5, -0.3])
fp = float(np.tanh(W2 @ np.tanh(W1 @ s + b1) + b2))
print(f'verification mu(0.5,-0.3) = {fp:.8f}   (runtime fingerprint 0.25842038)')

# ---- Sonoda-binary painting ----
from matplotlib.colors import ListedColormap
cmap2 = ListedColormap(['#3b6fb6', '#c0392b'])   # blue / red
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, constrained_layout=True)
for ax, F, title in [(axes[0], OPP, '(a) opposition control'),
                     (axes[1], EFF, '(b) learned policy (effective)')]:
    ax.contourf(U, V, np.sign(F), levels=[-1.5, 0, 1.5], cmap=cmap2)
    ax.contour(U, V, F, levels=[0.0], colors='k', linewidths=1.6)
    ax.set_xlabel(r"$u'^{+}$")
    ax.set_title(title, fontsize=12)
axes[0].set_ylabel(r"$v'^{+}$")
axes[1].plot(0.5, -0.3, marker='o', ms=5, mfc='w', mec='k', mew=1.0)
fig.savefig('policy_map_sonoda_settled.png', dpi=300)
fig.savefig('policy_map_sonoda_settled.pdf')
print('saved policy_map_sonoda')
