import re
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("/proj/liu2/msc_thesis_aditya/styles/plotstyle.mplstyle")

# Logs
CN_LOG = "/proj/liu2/msc_thesis_aditya/Submission/test_perturb_noise_nc13_68k/slurm-extend-4305779.out"
BWD_UNC_LOG = "/proj/liu2/msc_thesis_aditya/Submission/Backward_Ctrl_Uctrl/results/uncontrolled/log.pimpleFoam"
BWD_CTRL_LOG = "/proj/liu2/msc_thesis_aditya/Submission/Backward_Ctrl_Uctrl/results/controlled/log.pimpleFoam"
OUT = "/proj/liu2/msc_thesis_aditya/Submission/residual_comparison"

T_START_COMPARE = 10000

def parse_first_residual_per_step(log_path):
    rows = []
    re_time = re.compile(r'^Time = ([\d.e+-]+)')
    re_res = re.compile(r'Solving for (\w+), Initial residual = ([\d.eE+-]+)')
    current_time = None
    cur = {'p': None, 'Ux': None, 'Uy': None, 'Uz': None}
    with open(log_path) as f:
        for line in f:
            tm = re_time.match(line)
            if tm:
                if current_time is not None:
                    rows.append((current_time, cur['p'], cur['Ux'], cur['Uy'], cur['Uz']))
                current_time = float(tm.group(1))
                cur = {'p': None, 'Ux': None, 'Uy': None, 'Uz': None}
                continue
            rm = re_res.search(line)
            if rm and current_time is not None:
                field, val = rm.group(1), float(rm.group(2))
                if field in cur and cur[field] is None:
                    cur[field] = val
        if current_time is not None:
            rows.append((current_time, cur['p'], cur['Ux'], cur['Uy'], cur['Uz']))
    arr = np.array([r for r in rows if r[1] is not None], dtype=float)
    return arr

print("Parsing CN case (slurm-extend-4305779.out)...")
cn = parse_first_residual_per_step(CN_LOG)
print(f"  Got {len(cn)} timesteps, time range [{cn[0,0]:.1f}, {cn[-1,0]:.1f}]")

print("Parsing backward uncontrolled log...")
bwd_unc = parse_first_residual_per_step(BWD_UNC_LOG)
print(f"  Got {len(bwd_unc)} timesteps, time range [{bwd_unc[0,0]:.1f}, {bwd_unc[-1,0]:.1f}]")

print("Parsing backward controlled log...")
bwd_ctrl = parse_first_residual_per_step(BWD_CTRL_LOG)
print(f"  Got {len(bwd_ctrl)} timesteps, time range [{bwd_ctrl[0,0]:.1f}, {bwd_ctrl[-1,0]:.1f}]")

mask_bwd_unc = bwd_unc[:, 0] >= T_START_COMPARE
mask_bwd_ctrl = bwd_ctrl[:, 0] >= T_START_COMPARE
bwd_unc_f = bwd_unc[mask_bwd_unc]
bwd_ctrl_f = bwd_ctrl[mask_bwd_ctrl]
print(f"\nAfter filtering backward to t >= {T_START_COMPARE}:")
print(f"  bwd unc:  {len(bwd_unc_f)} samples")
print(f"  bwd ctrl: {len(bwd_ctrl_f)} samples")

field_names = ['p', 'Ux', 'Uy', 'Uz']
field_cols = [1, 2, 3, 4]
COL_CN = "#1E5288"
COL_BWD_UNC = "#6B7A99"
COL_BWD_CTRL = "#E76F51"

for fname, col in zip(field_names, field_cols):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(cn[:, 0], cn[:, col], '.', ms=0.5, alpha=0.5, color=COL_CN,
                label='CN (nOuter=1)')
    ax.semilogy(bwd_unc_f[:, 0], bwd_unc_f[:, col], '.', ms=0.5, alpha=0.5, color=COL_BWD_UNC,
                label='Backward Uncontrolled (nOuter=2)')
    ax.semilogy(bwd_ctrl_f[:, 0], bwd_ctrl_f[:, col], '.', ms=0.5, alpha=0.5, color=COL_BWD_CTRL,
                label='Backward Opposition (nOuter=2)')
    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(f'Initial residual ${fname}$')
    ax.set_xlim(T_START_COMPARE, 20000)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='black',
              markerscale=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/residual_{fname}_compare.png", dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Saved: residual_{fname}_compare.png")

print("\n" + "="*84)
print(f"RESIDUAL STATISTICS  (Window: t = [{T_START_COMPARE}, 20000])")
print("="*84)
print(f"{'Field':<6}{'CN (nOuter=1)':<28}{'Bwd Unc (nOuter=2)':<28}{'Bwd Ctrl (nOuter=2)'}")
print(f"{'':<6}{'mean        max':<28}{'mean        max':<28}{'mean        max'}")
print("-"*84)
for fname, col in zip(field_names, field_cols):
    vals_cn = cn[:, col][~np.isnan(cn[:, col])]
    vals_bwd_unc = bwd_unc_f[:, col][~np.isnan(bwd_unc_f[:, col])]
    vals_bwd_ctrl = bwd_ctrl_f[:, col][~np.isnan(bwd_ctrl_f[:, col])]
    print(f"{fname:<6}"
          f"{vals_cn.mean():<11.2e} {vals_cn.max():<14.2e}"
          f"{vals_bwd_unc.mean():<11.2e} {vals_bwd_unc.max():<14.2e}"
          f"{vals_bwd_ctrl.mean():<11.2e} {vals_bwd_ctrl.max():.2e}")
print("="*84)
print(f"\nPlots saved to: {OUT}")
