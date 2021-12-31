import numpy as np
import pytest

from isint_ufunc import isint


def annotated(cls):
    def annotate(fn):
        def wrapper(*args, **kwargs):
            print('Starting', fn.__name__)
            return fn(*args, **kwargs)
        return wrapper
    for name, value in list(cls.__dict__.items()):
        if name.startswith('test_') and callable(value):
            setattr(cls, name, annotate(value))
    return cls

@annotated
class TestIsInt:
    int_dtypes = (np.bool_, np.int8, np.uint8, np.int16, np.uint16,
                  np.int32, np.uint32, np.int64, np.uint64)
    float_dtypes = (np.half, np.float32, np.float64, np.float128)
    complex_dtypes = (np.complex64, np.complex128, np.complex256)
    odd_dtypes = (np.bytes_, np.string_, np.unicode_, np.object_, np.void,
                  np.timedelta64, np.datetime64)

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
        for dtype in self.int_dtypes + self.float_dtypes:
            for k in np.arange(-5, 6):
                n = dtype(k)
                assert isint(n), f'{k}: {n.dtype.name}({n})'

        for dtype in self.float_dtypes:
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
        for dtype in self.float_dtypes:
            name = np.finfo(dtype).dtype.name
            x = rng.uniform(-1000, 1000, size=shape).astype(dtype)
            if dtype == np.float128:
                x[40:] *= rng.uniform(-1000, 1000, size=(40,) + shape[1:])
            x[:40, ...] = rng.integers(-1000, 1000, size=(40,) + shape[1:])
            rng.shuffle(x.ravel())

            result = isint(x)
            actual = ((x % 1) == 0)

            assert result.shape == shape, f'Multidim shape {name}'
            assert np.array_equal(result, actual), f'Multidim {name}'

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
        for dtype, pinf, ninf in zip(
                [np.float16, np.float32, np.float64],
                [np.uint16(0x7C00), np.uint32(0x7F800000), np.uint64(0x7FF0000000000000)],
                [np.uint16(0xFC00), np.uint32(0xFF800000), np.uint64(0xFFF0000000000000)]):
            for sbit in (0, 1):
                assert np.isinf(pinf.view(dtype))
                assert np.isinf(ninf.view(dtype))
                assert not isint(pinf.view(dtype)), f'+inf {dtype(0).name}'
                assert not isint(ninf.view(dtype)), f'-inf {dtype(0).name}'
        # Add ldbl

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
        for dtype in self.odd_dtypes:
            x = np.zeros((3, 3), dtype=dtype)
            with pytest.raises(TypeError):
                isint(x)


if __name__ == '__main__':
    inst = TestIsInt()
    for name in dir(inst):
        if name.startswith('test_') and callable(value := getattr(inst, name)):
            value()

