from isint_ufunc import isint

class TestIsInt:
    def test_scalars(self):
        """
        Verify that a small sample of each scalar type works properly.
        """
    def test_multidims(self):
        """
        Verify that operates on multidimensional arrays.
        """
    def test_ints(self):
        """
        Check for all the integer types.
        """
    def test_nans(self):
        """
        Verify all the valid nans.
        """
    def test_unpseudonormal(self):
        """
        Verify all the bad inputs for long double
        (only if float128 is actually 80-bit extended).
        """
    def test_infs(self):
        """
        Test infinities.
        """
    def test_true(self):
        """
        Verify normal true values.
        """
    def test_false(self):
        """
        Verify normal false values.
        """"
