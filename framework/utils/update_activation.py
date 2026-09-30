"""
Helper to convert activation function string from YAML to PyTorch module.
"""

import torch.nn as nn


def update_activation_function(policy_kwargs):
    """
    Convert the 'activation_fn' string in policy_kwargs to actual PyTorch module.

    Args:
        policy_kwargs: dict from sac.yaml with 'activation_fn' as string

    Returns:
        Updated policy_kwargs with activation_fn as torch.nn module
    """
    activation_map = {
        "Tanh": nn.Tanh,
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "ELU": nn.ELU,
        "GELU": nn.GELU,
    }

    if "activation_fn" in policy_kwargs:
        act_name = policy_kwargs["activation_fn"]
        if act_name in activation_map:
            policy_kwargs["activation_fn"] = activation_map[act_name]
        else:
            raise ValueError(
                f"Unknown activation function: {act_name}. "
                f"Available: {list(activation_map.keys())}"
            )

    return policy_kwargs
