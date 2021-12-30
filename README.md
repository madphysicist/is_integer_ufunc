This is a sample implementation of a ufunc that checks if a floating point
value is an integer. It is inspired by my Stack Overflow question
https://stackoverflow.com/q/35042128/2988730, which has garnered a bit of
mild interest.

To build this package, run

    $ python setup.py build_ext --inplace
    $ cd isint_ufunc

Some preliminary benchmarks show that the double version of the function is
5x to 15x faster than using ``(x % 1) == 0``. Here is a simple timing test:

    In [0]: import numpy as np
    In [1]: from isint import isint

    In [2]: np.random.seed(0xBEEF)
    In [3]: x = np.random.rand(10000, 10000)
    In [4]: x[5000:, :] = np.random.randint(-1000, 1000, size=(5000, 10000))
    In [5]: np.random.shuffle(x)

    In [6]: np.array_equal(((x % 1) == 0), isint(x))
    Out[6]: True

    In [7]: %timeit isint(x)
    135 ms ± 1.02 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

    In [8]: %timeit (x % 1) == 0
    2.08 s ± 28.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

Setting the size to ``(1000, 1000)`` yields a 5x rather than a 15x improvement,
likely due to the smaller overhead imposed by the intermediate arrays.

Tests are available, currently in ``test_isint.py``.

