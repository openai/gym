"""Duration module."""

import operator

from matplotlib import _api


class Duration:
    """Class Duration in development."""

    allowed = ["ET", "UTC"]

    def __init__(self, frame, seconds):
        """
        Create a new Duration object.

        = ERROR CONDITIONS
        - If the input frame is not in the allowed list, an error is thrown.

        = INPUT VARIABLES
        - frame     The frame of the duration.  Must be 'ET' or 'UTC'
        - seconds  The number of seconds in the Duration.
        """
        _api.check_in_list(self.allowed, frame=frame)
        self._frame = frame
        self._seconds = seconds

    def frame(self):
        """Return the frame the duration is in."""
        return self._frame

    def __abs__(self):
        """Return the absolute value of the duration."""
        return Duration(self._frame, abs(self._seconds))

    def __neg__(self):
        """Return the negative value of this Duration."""
        return Duration(self._frame, -self._seconds)

    def seconds(self):
        """Return the number of seconds in the Duration."""
        return self._seconds

    def __bool__(self):
        return self._seconds != 0

    def __eq__(self, rhs):
        return self._cmp(rhs, operator.eq)

    def __ne__(self, rhs):
        return self._cmp(rhs, operator.ne)

    def __lt__(self, rhs):
        return self._cmp(rhs, operator.lt)

    def __le__(self, rhs):
        return self._cmp(rhs, operator.le)

    def __gt__(self, rhs):
        return self._cmp(rhs, operator.gt)

    def __ge__(self, rhs):
        return self._cmp(rhs, operator.ge)

    def _cmp(self, rhs, op):
        """
        Compare two Durations.

        = INPUT VARIABLES
        - rhs     The Duration to compare against.
        - op      The function to do the comparison

        = RETURN VALUE
        - Returns op(self, rhs)
        """
        self.checkSameFrame(rhs, "compare")
        return op(self._seconds, rhs._seconds)

    def __add__(self, rhs):
        """
        Add two Durations.

        = ERROR CONDITIONS
        - If the input rhs is not in the same frame, an error is thrown.

        = INPUT VARIABLES
        - rhs     The Duration to add.

        = RETURN VALUE
        - Returns the sum of ourselves and the input Duration.
        """
        # Delay-load due to circular dependencies.
        import matplotlib.testing.jpl_units as U

        if isinstance(rhs, U.Epoch):
            return rhs + self

        self.checkSameFrame(rhs, "add")
        return Duration(self._frame, self._seconds + rhs._seconds)

    def __sub__(self, rhs):
        """
        Subtract two Durations.

        = ERROR CONDITIONS
        - If the input rhs is not in the same frame, an error is thrown.

        = INPUT VARIABLES
        - rhs     The Duration to subtract.

        = RETURN VALUE
        - Returns the difference of ourselves and the input Duration.
        """
        self.checkSameFrame(rhs, "sub")
        return Duration(self._frame, self._seconds - rhs._seconds)

    def __mul__(self, rhs):
        """
        Scale a UnitDbl by a value.

        = INPUT VARIABLES
        - rhs     The scalar to multiply by.

        = RETURN VALUE
        - Returns the scaled Duration.
        """
        return Duration(self._frame, self._seconds * float(rhs))

    def __rmul__(self, lhs):
        """
        Scale a Duration by a value.

        = INPUT VARIABLES
        - lhs     The scalar to multiply by.

        = RETURN VALUE
        - Returns the scaled Duration.
        """
        return Duration(self._frame, self._seconds * float(lhs))

    def __str__(self):
        """Print the Duration."""
        return "%g %s" % (self._seconds, self._frame)

    def __repr__(self):
        """Print the Duration."""
        return "Duration('%s', %g)" % (self._frame, self._seconds)

    def checkSameFrame(self, rhs, func):
        """
        Check to see if frames are the same.

        = ERROR CONDITIONS
        - If the frame of the rhs Duration is not the same as our frame,
          an error is thrown.

        = INPUT VARIABLES
        - rhs     The Duration to check for the same frame
        - func    The name of the function doing the check.
        """
        if self._frame != rhs._frame:
            raise ValueError(
                f"Cannot {func} Durations with different frames.\n"
                f"LHS: {self._frame}\n"
                f"RHS: {rhs._frame}")
