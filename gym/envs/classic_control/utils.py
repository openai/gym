"""
Utility functions used for classic control environments.
"""

from typing import Optional, SupportsFloat, Union


def verify_number_and_cast(x: SupportsFloat) -> float:
    """Verify parameter is a single number and cast to a float."""
    try:
        x = float(x)
    except (ValueError, TypeError):
        raise ValueError(f"Your input must support being cast to a float: {x}")
    return x


def maybe_parse_reset_bounds(
    options: Optional[dict], default_low: float, default_high: float
) -> Union[float, float]:
    """
    This function can be called during a reset() to customize the sampling
    ranges for setting the initial state distributions.

    Args:
      options: (Optional) options passed in to reset().
      default_low: Default lower limit to use, if none specified in options.
      default_high: Default upper limit to use, if none specified in options.
      limit_low: Lowest allowable value for user-specified lower limit.
      limit_high: Highest allowable value for user-specified higher limit.

    Returns:
      Lower and higher limits.
    """
    if options is None:
        return default_low, default_high

    low = options.get("low") if "low" in options else default_low
    high = options.get("high") if "high" in options else default_high
    # We expect only numerical inputs.
    low = verify_number_and_cast(low)
    high = verify_number_and_cast(high)
    if low > high:
        raise ValueError("Lower bound must be lower than higher bound.")
    return low, high
