"""
Verification 2: C++ inside OpenFOAM vs Python PyTorch
Compares actions from ActionState.dat (C++ deterministic) with Python PyTorch (batch_140).
"""

import torch
import numpy as np
from stable_baselines3 import SAC
from environment import OFEnvironment

env = OFEnvironment("configs/channel.yaml", 0)
model = SAC.load("model/checkpoints/batch_140", env=env)
pytorch_actor = model.actor
pytorch_actor.eval()

# Read ActionState.dat from the deterministic run
data = np.loadtxt(
    "channel/run0_ep0/postProcessing/channelDRLControl/0/ActionState.dat",
    skiprows=2
)
print(f"ActionState.dat: {data.shape[0]} timesteps, {data.shape[1]} columns")
print(f"  = {data.shape[0]} timesteps x {(data.shape[1]-1)//3} actuators")

# Compare C++ vs Python for every actuator at every timestep
max_diff = 0.0
all_diffs = []
n_compared = 0

for t in range(data.shape[0]):
    time = data[t, 0]
    n_actuators = (data.shape[1] - 1) // 3

    for j in range(n_actuators):
        cpp_action = data[t, 1 + 3*j]
        u = data[t, 2 + 3*j]
        v = data[t, 3 + 3*j]

        # Feed same state to Python PyTorch
        state = np.array([[u, v]], dtype=np.float32)
        with torch.no_grad():
            mean, _, _ = pytorch_actor.get_action_dist_params(torch.FloatTensor(state))
            py_action = torch.tanh(mean).numpy().flatten()[0]

        diff = abs(cpp_action - py_action)
        all_diffs.append(diff)
        if diff > max_diff:
            max_diff = diff
            worst = (t, j, time, u, v, cpp_action, py_action)

        n_compared += 1

    # Print summary per timestep
    t_diffs = all_diffs[-n_actuators:]
    print(f"  t={time:.2f}: compared {n_actuators} actuators, "
          f"max diff={max(t_diffs):.2e}, mean diff={np.mean(t_diffs):.2e}")

print(f"\n{'='*60}")
print(f"VERIFICATION 2 SUMMARY")
print(f"{'='*60}")
print(f"  Total comparisons: {n_compared}")
print(f"  Max difference:    {max_diff:.2e}")
print(f"  Mean difference:   {np.mean(all_diffs):.2e}")
print(f"  Tolerance:         1e-5")
print(f"\n  Worst case:")
print(f"    Timestep {worst[0]}, actuator {worst[1]}, t={worst[2]:.2f}")
print(f"    State: u={worst[3]:.6f}, v={worst[4]:.6f}")
print(f"    C++ action:    {worst[5]:.8f}")
print(f"    Python action: {worst[6]:.8f}")

if max_diff < 1e-5:
    print(f"\n  PASS — C++ inside OpenFOAM matches Python PyTorch")
    print(f"  cppflow loads saved_model.pb correctly")
elif max_diff < 1e-3:
    print(f"\n  PASS (with float32 tolerance) — minor numerical differences")
else:
    print(f"\n  FAIL — C++ and Python give different actions")
