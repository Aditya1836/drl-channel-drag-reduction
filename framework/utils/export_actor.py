"""
Export SB3 SAC actor (PyTorch) as TensorFlow SavedModel.

Guastoni architecture: Actor has 1 hidden layer with 8 nodes.

NOTE: cppflow cannot handle tf.bool correctly. Using float instead.
"""

import torch
import numpy as np
import tensorflow as tf
from pathlib import Path


def export_actor(actor, env):
    policy_dir = env.base_policy_dir
    policy_dir.mkdir(parents=True, exist_ok=True)

    actor.eval()
    weights = {}
    for name, param in actor.named_parameters():
        weights[name] = param.detach().cpu().numpy()

    w0 = weights['latent_pi.0.weight'].T
    b0 = weights['latent_pi.0.bias']
    w_mu = weights['mu.weight'].T
    b_mu = weights['mu.bias']
    w_log_std = weights['log_std.weight'].T
    b_log_std = weights['log_std.bias']

    actor.train()

    n_states = env.observation_space.shape[0]

    class PolicyModel(tf.Module):
        def __init__(self):
            super().__init__()
            self.w0 = tf.Variable(w0, dtype=tf.float32, name='w0')
            self.b0 = tf.Variable(b0, dtype=tf.float32, name='b0')
            self.w_mu = tf.Variable(w_mu, dtype=tf.float32, name='w_mu')
            self.b_mu = tf.Variable(b_mu, dtype=tf.float32, name='b_mu')
            self.w_log_std = tf.Variable(w_log_std, dtype=tf.float32, name='w_log_std')
            self.b_log_std = tf.Variable(b_log_std, dtype=tf.float32, name='b_log_std')

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, n_states], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32)
        ])
        def __call__(self, args_0, deterministic):
            x = tf.matmul(args_0, self.w0) + self.b0
            x = tf.math.tanh(x)
            mean = tf.matmul(x, self.w_mu) + self.b_mu
            log_std = tf.matmul(x, self.w_log_std) + self.b_log_std
            log_std = tf.clip_by_value(log_std, -20.0, 2.0)
            std = tf.exp(log_std)
            noise = tf.random.normal(tf.shape(mean))
            action = tf.math.tanh(mean + (1.0 - deterministic) * std * noise)
            return action

    model = PolicyModel()
    tf.saved_model.save(model, str(policy_dir))
    print(f"  Policy exported to {policy_dir} (TensorFlow SavedModel)")

    loaded = tf.saved_model.load(str(policy_dir))
    test_state = tf.constant([[0.5, -0.3]], dtype=tf.float32)
    test_det = tf.constant(1.0)
    test_output = loaded(test_state, test_det)
    print(f"  Verification: input={test_state.numpy()}, output={test_output.numpy()}")
