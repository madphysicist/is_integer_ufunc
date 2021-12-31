This is a sample implementation of a ufunc that checks if a floating point
value is an integer. It is inspired by my Stack Overflow question
https://stackoverflow.com/q/35042128/2988730, which has garnered a bit of
mild interest.

To build this package, run

    $ python setup.py build_ext --inplace
    $ cd isint_ufunc

Due to the small and experimental nature of this library, cleaning is largely
a manual process:

    $ rm -rf build/ isint_ufunc.* __pycache__/

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

A larger variety of benchmarks are available in ``benchmark_isint.py``.
Fairly comprehensive tests are available, currently in ``test_isint.py``.

# References:
-------------

- https://standards.ieee.org/standard/754-2019.html: IEEE 754-2019 - IEEE Standard for Floating-Point Arithmetic
- https://en.wikipedia.org/wiki/IEEE_754
- https://en.wikipedia.org/wiki/Half-precision_floating-point_format
- https://en.wikipedia.org/wiki/Single-precision_floating-point_format
- https://en.wikipedia.org/wiki/Double-precision_floating-point_format
- https://en.wikipedia.org/wiki/Extended_precision
- https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format


# Justifications:
-----------------

- https://stackoverflow.com/q/35042128/2988730: Numpy: Check if float array contains whole numbers
- https://stackoverflow.com/q/64044147/2988730: How to check if all elements of a numpy array have integer values? 
- https://stackoverflow.com/q/36505879/2988730: Convert float64 to int64 using "safe"
- https://stackoverflow.com/q/21583758/2988730: How to check if a float value is a whole number
- https://stackoverflow.com/q/934616/2988730: How do I find out if a numpy array contains integers?
- https://stackoverflow.com/q/6209008/2988730: Checking if float is equivalent to an integer value in python
- https://stackoverflow.com/q/52094533/2988730: Filter integers in numpy float array

