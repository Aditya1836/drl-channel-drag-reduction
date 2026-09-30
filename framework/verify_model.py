#!/usr/bin/env python3
"""Verify the TF SavedModel in a case dir matches the expected batch.
Usage: python3 verify_model.py <case_dir> <expected_batch>
"""
import sys, os, hashlib
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: python3 verify_model.py <case_dir> <expected_batch>")
    sys.exit(1)

case = sys.argv[1].rstrip('/')
batch = int(sys.argv[2])
os.chdir(case)
sys.path.insert(0, case)

import tensorflow as tf
import numpy as np
import torch
from stable_baselines3 import SAC
from environment import OFEnvironment

# 1. Check the file is fresh (modified after setup)
pb_file = "channel/base/model/saved_model.pb"
if not os.path.exists(pb_file):
    print(f"ERROR: {pb_file} does not exist")
    sys.exit(1)

mtime = os.path.getmtime(pb_file)
size = os.path.getsize(pb_file)
md5 = hashlib.md5(open(pb_file, 'rb').read()).hexdigest()
print(f"File:     {pb_file}")
print(f"Size:     {size:,} bytes")
print(f"Mtime:    {mtime}")
print(f"MD5:      {md5}")

# 2. Load it and check the canonical output
m = tf.saved_model.load("channel/base/model")
obs = tf.constant([[0.5, -0.3]], dtype=tf.float32)
det = tf.constant(1.0)
deployed_output = m(obs, det).numpy()[0, 0]
print(f"\nDeployed output at (0.5, -0.3) det:  {deployed_output:.8f}")

# 3. Compute the expected output independently from batch_N.zip
env = OFEnvironment(Path("configs/channel.yaml"), 0)
model = SAC.load(f"model/checkpoints/batch_{batch}.zip", env=env, device=torch.device("cpu"))
actor = model.actor
obs_t = torch.tensor([[0.5, -0.3]], dtype=torch.float32)
with torch.no_grad():
    expected_output = actor(obs_t, deterministic=True)[0].numpy()[0, 0]
print(f"Expected output from batch_{batch}.zip:    {expected_output:.8f}")

# 4. Compare
diff = abs(deployed_output - expected_output)
if diff < 1e-5:
    print(f"\n✅ MATCH: deployed model is batch_{batch} (diff={diff:.2e})")
else:
    print(f"\n❌ MISMATCH: deployed model is NOT batch_{batch} (diff={diff:.2e})")
    print("   The TF SavedModel was NOT replaced with the trained weights!")
    sys.exit(1)
