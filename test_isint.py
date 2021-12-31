import numpy as np
import pytest

from isint_ufunc import isint


def get_dtypes(label, bits):
    return [getattr(np, name) for n in bits if hasattr(np, name := f'{label}{n}')]

float_dtypes = get_dtypes('float', [16, 32, 64, 80, 96, 128, 256])
cplx_dtypes = get_dtypes('complex', [64, 128, 160, 192, 256, 512])
int_dtypes = get_dtypes('int', [8, 16, 32, 64, 128, 256])
uint_dtypes = get_dtypes('uint', [8, 16, 32, 64, 128, 256])

# This does not have to be overly complete
odd_dtypes = [np.bytes_, np.string_, np.unicode_, np.object_, np.void,
              np.timedelta64, np.datetime64]


class TestIsInt:
    float_dtypes = (np.half, np.single, np.double)
    complex_dtypes = (np.complex64, np.complex128, np.complex256)

    def test_objects(self):
        """
        Check common python types.
        """
        assert isint(False), 'Python bool'
        assert isint(3), 'Python int'
        assert isint(3.0), 'Python float'
        assert not isint(-3.1), 'Python float'
        with pytest.raises(TypeError):
            isint('abc'), 'Python string'

    def test_scalars(self):
        """
        Verify that a small sample of each scalar type works properly.
        """
        for dtype in int_dtypes + float_dtypes:
            for k in np.arange(-5, 6):
                n = dtype(k)
                assert isint(n), f'{k}: {n.dtype.name}({n})'

        for dtype in uint_dtypes:
            i = np.iinfo(dtype)
            for k in np.arange(i.max - 5, i.max + 1):
                n = dtype(k)
                assert isint(n), f'{k}: {n.dtype.name}({n})'
            for k in np.arange(0, 6):
                n = dtype(k)
                assert isint(n), f'{k}: {n.dtype.name}({n})'

        for dtype in float_dtypes:
            i = np.finfo(dtype)
            assert isint(i.min), f'{i.min.dtype.name}({i.min})'
            assert isint(i.max), f'{i.max.dtype.name}({i.max})'
            assert not isint(i.resolution), f'{i.resolution.dtype.name}({i.resolution})'
            assert not isint(-i.resolution), f'{i.resolution.dtype.name}(-{i.resolution})'

    def test_zerodims(self):
        """
        Test 2D array with a zero dimension.
        """
        for a in (np.empty((3, 0), dtype=int), np.empty((0, 5), dtype=float)):
            v = isint(a)
            assert v.dtype == np.bool_
            assert np.array_equal(v, np.empty(a.shape, dtype=bool))
        with pytest.raises(TypeError):
            isint(np.empty(0, dtype=np.string_))

    def test_multidims(self):
        """
        Verify that operates on multidimensional arrays.
        """
        shape = (80, 53, 17, 20)
        rng = np.random.default_rng(0xBEEF)
        for dtype in float_dtypes:
            i = np.finfo(dtype)
            x = rng.uniform(-1000, 1000, size=shape).astype(dtype)
            if dtype == np.float128:
                x[40:] *= rng.uniform(-1000, 1000, size=(40,) + shape[1:])
            x[:40, ...] = rng.integers(-1000, 1000, size=(40,) + shape[1:])
            rng.shuffle(x.ravel())

            result = isint(x)
            actual = ((x % 1) == 0)

            assert result.shape == shape, f'Multidim shape {i.dtype.name}'
            assert np.array_equal(result, actual), f'Multidim {i.dtype.name}'

        for dtype in int_dtypes + uint_dtypes:
            i = np.iinfo(dtype)
            x = rng.integers(i.min, i.max + 1, size=shape, dtype=dtype)

            result = isint(x)

            assert result.shape == shape, f'Multidim shape {i.dtype.name}'
            assert np.array_equiv(result, True), f'Multidim {i.dtype.name}'


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
        for dtype in float_dtypes:
            assert not isint(dtype('+inf')), f'+inf {dtype(0).dtype.name}'
            assert not isint(dtype('-inf')), f'-inf {dtype(0).dtype.name}'

    def test_true(self):
        """
        Verify normal true values.
        """

    def test_false(self):
        """
        Verify normal false values.
        """

    def test_complex(self):
        """
        Verify that only real values pass.
        """
 
    def test_odd(self):
        """
        Check string, unicode, object, datetime, timedela.
        """
        for dtype in odd_dtypes:
            x = np.zeros((3, 3), dtype=dtype)
            with pytest.raises(TypeError):
                isint(x)


if __name__ == '__main__':
    inst = TestIsInt()
    for name in dir(inst):
        if name.startswith('test_') and callable(value := getattr(inst, name)):
            value()

