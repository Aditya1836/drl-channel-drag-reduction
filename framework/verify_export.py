import sys
sys.path = [p for p in sys.path if "setuptools-rust" not in p]
sys.path.insert(0, "/home/x_adipa/.local/lib/python3.11/site-packages")
"""
Verification 1: Python PyTorch vs Python TensorFlow
Proves export_actor.py converts PyTorch weights to TensorFlow correctly.

Step 1: Load Model 1 (PyTorch checkpoint: batch_140.zip)
        Load Model 2 (TensorFlow SavedModel from export_actor.py)
Step 2: Generate 100 random states
Step 3: Deterministic test — must match exactly (tolerance 1e-5)
Step 4: Stochastic test — means over 10000 samples must match (tolerance 0.02)
"""

import torch
import numpy as np
import tensorflow as tf
from stable_baselines3 import SAC
from environment import OFEnvironment
from utils.export_actor import export_actor

# ============================================================
# Step 1: Load both models
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Loading both models")
print("=" * 60)

env = OFEnvironment("configs/channel.yaml", 0)
model = SAC.load("model/checkpoints/batch_140", env=env)
pytorch_actor = model.actor
pytorch_actor.eval()
print("  Model 1 (PyTorch): loaded from model/checkpoints/batch_140.zip")

export_actor(model.actor, env)
tf_model = tf.saved_model.load(str(env.base_policy_dir))
print(f"  Model 2 (TensorFlow): loaded from {env.base_policy_dir}")

# ============================================================
# Step 2: Generate random test observations
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Generating random test observations")
print("=" * 60)

n_tests = 100
np.random.seed(42)
test_states = np.random.randn(n_tests, env.n_states).astype(np.float32)
print(f"  Generated {n_tests} random states, shape: {test_states.shape}")

# ============================================================
# Step 3: Deterministic test
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Deterministic test — Model1(state, True) == Model2(state, 1.0)")
print("=" * 60)

max_diff = 0.0
all_diffs = []

for i in range(n_tests):
    state = test_states[i:i+1]

    with torch.no_grad():
        state_torch = torch.FloatTensor(state)
        mean_actions, log_std, kwargs = pytorch_actor.get_action_dist_params(state_torch)
        pt_action = torch.tanh(mean_actions).numpy().flatten()[0]

    state_tf = tf.constant(state, dtype=tf.float32)
    tf_action = tf_model(state_tf, tf.constant([1.0])).numpy().flatten()[0]

    diff = abs(pt_action - tf_action)
    all_diffs.append(diff)
    if diff > max_diff:
        max_diff = diff

    if i < 5:
        print(f"  State {i}: PyTorch={pt_action:.8f}  TensorFlow={tf_action:.8f}  diff={diff:.2e}")

print(f"\n  Results over {n_tests} random states:")
print(f"    Max difference:  {max_diff:.2e}")
print(f"    Mean difference: {np.mean(all_diffs):.2e}")
print(f"    Tolerance:       1e-5")

if max_diff < 1e-5:
    print("    PASS — Deterministic outputs match exactly")
else:
    print("    FAIL — Deterministic outputs differ!")

# ============================================================
# Step 4: Stochastic test
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Stochastic test — means over 10000 samples must match")
print("=" * 60)

n_stochastic_samples = 10000
n_test_states_stochastic = 10

all_mean_diffs = []

for i in range(n_test_states_stochastic):
    state = test_states[i:i+1]

    # PyTorch stochastic
    pt_actions = []
    with torch.no_grad():
        state_torch = torch.FloatTensor(state)
        for _ in range(n_stochastic_samples):
            mean_actions, log_std, kwargs = pytorch_actor.get_action_dist_params(state_torch)
            log_std_clamped = torch.clamp(log_std, -20.0, 2.0)
            std = torch.exp(log_std_clamped)
            noise = torch.randn_like(mean_actions)
            action = torch.tanh(mean_actions + std * noise)
            pt_actions.append(action.numpy().flatten()[0])

    # TensorFlow stochastic (deterministic=0.0)
    tf_actions = []
    state_tf = tf.constant(state, dtype=tf.float32)
    for _ in range(n_stochastic_samples):
        action = tf_model(state_tf, tf.constant([0.0])).numpy().flatten()[0]
        tf_actions.append(action)

    pt_mean = np.mean(pt_actions)
    tf_mean = np.mean(tf_actions)
    pt_std = np.std(pt_actions)
    tf_std = np.std(tf_actions)
    mean_diff = abs(pt_mean - tf_mean)
    all_mean_diffs.append(mean_diff)

    print(f"  State {i}: PT mean={pt_mean:.6f} std={pt_std:.6f} | "
          f"TF mean={tf_mean:.6f} std={tf_std:.6f} | "
          f"mean_diff={mean_diff:.6f}")

max_mean_diff = max(all_mean_diffs)
avg_mean_diff = np.mean(all_mean_diffs)

print(f"\n  Results over {n_test_states_stochastic} states x {n_stochastic_samples} samples:")
print(f"    Max mean difference:  {max_mean_diff:.6f}")
print(f"    Avg mean difference:  {avg_mean_diff:.6f}")
print(f"    Tolerance:            0.02")

if max_mean_diff < 0.02:
    print("    PASS — Stochastic distributions match")
else:
    print("    CHECK — Stochastic means differ (may need more samples)")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("VERIFICATION 1 SUMMARY (trained model: batch_140)")
print("=" * 60)
det_pass = max_diff < 1e-5
sto_pass = max_mean_diff < 0.02
print(f"  Deterministic test:  {'PASS' if det_pass else 'FAIL'} (max diff = {max_diff:.2e})")
print(f"  Stochastic test:     {'PASS' if sto_pass else 'CHECK'} (max mean diff = {max_mean_diff:.6f})")
if det_pass and sto_pass:
    print("\n  VERIFIED — export_actor.py converts PyTorch to TensorFlow correctly")
    print("  PyTorch (batch_140.zip) == TensorFlow (saved_model.pb)")
else:
    print("\n  Verification FAILED — check export_actor.py for bugs")
