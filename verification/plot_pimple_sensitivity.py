#!/usr/bin/env python3
"""
PIMPLE corrector-loop sensitivity at 68k baseline.
Compares nc13, nc14, nc22, nc32 against Kim, Moin & Moser (1987).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

BASE       = "/proj/liu2/msc_thesis_aditya/Submission"
MKM_DIR    = f"{BASE}/Existing_Literature/channel/dataset/chandata/chan180/profiles"
OUTPUT_DIR = f"{BASE}/pimple_sensitivity"

CASES = {
    "nc13": f"{BASE}/DRL_baseline_nc13_68k",
    "nc14": f"{BASE}/DRL_baseline_nc14_68k",
    "nc22": f"{BASE}/opp_new2_10k/results/uncontrolled",
    "nc32": f"{BASE}/DRL_baseline_nc32_68k",
    "bwd23": f"{BASE}/Backward_Ctrl_Uctrl/results/uncontrolled",
}

nu, delta, Re_tau = 4.7679e-05, 1.0, 180
u_tau = Re_tau * nu / delta

COL_DNS = "black"
LBL_DNS = "Kim, Moin & Moser (1987)"
STYLE = {
    "nc13": dict(color="#1E2761", ls="-",  label=r"nc13: $n_{\rm outer}=1,\;n_{\rm corr}=3$"),
    "nc14": dict(color="#4A9D7F", ls="--", label=r"nc14: $n_{\rm outer}=1,\;n_{\rm corr}=4$"),
    "nc22": dict(color="#7E5EA0", ls=":",  label=r"nc22: $n_{\rm outer}=2,\;n_{\rm corr}=2$"),
    "nc32": dict(color="#C9A961", ls="-.", label=r"nc32: $n_{\rm outer}=3,\;n_{\rm corr}=2$"),
    "bwd23": dict(color="#C0392B", ls="-",  label=r"nc23: $n_{\rm outer}=2,\;n_{\rm corr}=3$"),
}

print("=" * 64)
print(" PIMPLE corrector sensitivity (68k baseline)")
print(" Reference: Kim, Moin & Moser (1987), Reτ = 180")
print("=" * 64)
print(f"  u_tau = {u_tau:.6f} m/s\n")

def latest_graphs(case_dir):
    g = os.path.join(case_dir, "graphs")
    if not os.path.isdir(g):
        return None
    dirs = []
    for d in os.listdir(g):
        try: dirs.append((float(d), d))
        except ValueError: continue
    if not dirs: return None
    dirs.sort()
    return os.path.join(g, dirs[-1][1])

def load_xy(fp):
    d = np.loadtxt(fp); return d[:, 0], d[:, 1]
def load_dns_means(fp):
    d = np.loadtxt(fp, comments='#')
    return {'y_plus': d[:, 1], 'U_plus': d[:, 2]}
def load_dns_rey(fp):
    d = np.loadtxt(fp, comments='#')
    return {'y_plus': d[:, 1], 'uu_plus': d[:, 2], 'vv_plus': d[:, 3],
            'ww_plus': d[:, 4], 'uv_plus': d[:, 5]}

def load_case(case_dir):
    g = latest_graphs(case_dir)
    if g is None:
        raise SystemExit(f"no graphs/ folder in {case_dir}")
    print(f"  {case_dir}\n    latest = {os.path.basename(g)}")
    y, U     = load_xy(f"{g}/Uf.xy")
    _, u_rms = load_xy(f"{g}/u.xy")
    _, v_rms = load_xy(f"{g}/v.xy")
    _, w_rms = load_xy(f"{g}/w.xy")
    _, uv    = load_xy(f"{g}/uv.xy")
    return dict(
        y_plus = y * u_tau / nu,
        U_plus = U / u_tau,
        u_plus = u_rms / u_tau,
        v_plus = v_rms / u_tau,
        w_plus = w_rms / u_tau,
        uv_plus= uv / u_tau**2,
    )

print("Loading cases...")
data = {tag: load_case(d) for tag, d in CASES.items()}
print("\nLoading DNS reference...")
dns_m = load_dns_means(f"{MKM_DIR}/chan180.means")
dns_r = load_dns_rey  (f"{MKM_DIR}/chan180.reystress")
v_dns = np.sqrt(dns_r['vv_plus'])

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times', 'CMU Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 13, 'axes.labelsize': 16, 'legend.fontsize': 10.5,
    'legend.frameon': True, 'legend.framealpha': 1.0,
    'legend.edgecolor': 'black', 'legend.fancybox': False,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'xtick.major.size': 6, 'xtick.minor.size': 3,
    'ytick.major.size': 6, 'ytick.minor.size': 3,
    'axes.linewidth': 1.0, 'lines.linewidth': 1.6,
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# 01 — Mean velocity
fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogx(dns_m['y_plus'], dns_m['U_plus'], 'o', color=COL_DNS, mfc='none',
            markeredgewidth=1.0, markersize=5, label=LBL_DNS)
for tag, d in data.items():
    s = STYLE[tag]
    ax.semilogx(d['y_plus'], d['U_plus'], s['ls'], color=s['color'], label=s['label'])
yp = np.logspace(0.5, 2.3, 50)
ax.semilogx(yp, (1/0.41)*np.log(yp) + 5.2, ':', color='grey', lw=1.0,
            label=r'$U^+ = \frac{1}{\kappa}\ln y^+ + B$')
ax.set_xlabel(r'$y^+$'); ax.set_ylabel(r'$U^+$')
ax.set_xlim([0.3, 200]); ax.grid(False)
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/01_mean_velocity_pimple.pdf")
fig.savefig(f"{OUTPUT_DIR}/01_mean_velocity_pimple.png", dpi=300)
print("Saved: 01_mean_velocity_pimple"); plt.close()

# 02 — v'_rms+
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(dns_r['y_plus'], v_dns, 'o', color=COL_DNS, mfc='none',
        markeredgewidth=1.0, markersize=5, label=LBL_DNS)
for tag, d in data.items():
    s = STYLE[tag]
    ax.plot(d['y_plus'], d['v_plus'], s['ls'], color=s['color'], label=s['label'])
ax.set_xlabel(r'$y^+$'); ax.set_ylabel(r"$v'^+_{rms}$")
ax.set_xlim([0, 180]); ax.set_ylim([0, 1.2])
ax.grid(False); ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/02_v_rms_pimple.pdf")
fig.savefig(f"{OUTPUT_DIR}/02_v_rms_pimple.png", dpi=300)
print("Saved: 02_v_rms_pimple"); plt.close()

# 03 — Reynolds shear
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(dns_r['y_plus'], -dns_r['uv_plus'], 'o', color=COL_DNS, mfc='none',
        markeredgewidth=1.0, markersize=5, label=LBL_DNS)
for tag, d in data.items():
    s = STYLE[tag]
    ax.plot(d['y_plus'], -d['uv_plus'], s['ls'], color=s['color'], label=s['label'])
ax.set_xlabel(r'$y^+$'); ax.set_ylabel(r"$-\overline{u'v'}^+$")
ax.set_xlim([0, 180]); ax.set_ylim([0, 1.0])
ax.grid(False); ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/03_reynolds_shear_pimple.pdf")
fig.savefig(f"{OUTPUT_DIR}/03_reynolds_shear_pimple.png", dpi=300)
print("Saved: 03_reynolds_shear_pimple"); plt.close()

# SUMMARY
def peak(yp, val):
    i = np.argmax(np.abs(val))
    return val[i], yp[i]

print("\n" + "=" * 64)
print(" SUMMARY")
print("=" * 64)
print(f"\n  DNS               : U+_c = {dns_m['U_plus'][-1]:.3f}, "
      f"peak v'_rms+ = {v_dns.max():.3f}, "
      f"peak -u'v'+ = {(-dns_r['uv_plus']).max():.3f}")
for tag, d in data.items():
    v_pk, _ = peak(d['y_plus'], d['v_plus'])
    uv_pk, _ = peak(d['y_plus'], -d['uv_plus'])
    print(f"  {tag:17s} : U+_c = {d['U_plus'][-1]:.3f}, "
          f"peak v'_rms+ = {v_pk:.3f}, "
          f"peak -u'v'+ = {uv_pk:.3f}")

print("\n" + "=" * 64)
print(f"PDFs + PNGs in: {OUTPUT_DIR}")
print("=" * 64)
