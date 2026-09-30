"""
Learning rate scheduler for SB3.
"""


def linear_schedule(initial_value, final_value):
    """
    Linear learning rate schedule.

    Args:
        initial_value: starting learning rate
        final_value: ending learning rate

    Returns:
        schedule function compatible with SB3
    """
    def func(progress_remaining):
        """
        Progress remaining goes from 1.0 (start) to 0.0 (end).
        """
        return final_value + progress_remaining * (initial_value - final_value)

    return func
