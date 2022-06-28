"""
Utility functions used for classic control environments.
"""

from typing import Optional, Union
import numpy as np


def verify_number(x: Union[int, float, np.ndarray]) -> bool:
    """Verify parameter is a single number."""
    if type(x) == int or type(x) == float:
        # A single value that is either an int or a float.
        return True
    if type(x) == np.ndarray:
        if ((np.issubdtype(x.dtype, np.floating) or
             np.issubdtype(x.dtype, np.integer)) and
            len(x.shape) == 0):
            # A numpy single value that is either an int or a float.
            return True
    return False


def maybe_parse_reset_bounds(options: Optional[dict],
                             default_low: float,
                             default_high: float) -> Union[float, float]:
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

    low = options.get('low') if 'low' in options else default_low
    high = options.get('high') if 'high' in options else default_high
    # We expect only numerical inputs.
    assert verify_number(low)
    assert verify_number(high)
    assert low <= high
    return low, high
