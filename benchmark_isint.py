from timeit import Timer
import numpy as np
from isint_ufunc import isint


SIZES = range(2, 9, 2)
ITYPES = (np.int8, np.int16, np.int32, np.int64)
UTYPES = (np.uint8, np.uint16, np.uint32, np.uint64)
FTYPES = (np.float16, np.float32, np.float64, np.float128)


class Bench(Timer):
    def autorange(self):
        super().autorange(self.callback)
        return self.time

    def callback(self, number, time):
        self.time = time / number

rng = np.random.default_rng(0xBEEF)


print(f'{"=":=^72}')
print(f'{"Speedup of isint vs (x % 1) == 0":^72s}')
print(f'{"=":=^72}')
print(f'{"Integers":^72s}')
print(*[12 * '-'] * 5, sep='-+-')
bs = '\\'
print(f'{bs.join(["dtype ", " N"]):>12s}', end='')
for size in SIZES:
    print(f' | {10**size:>12d}', end='')
print()
print(*[12 * '-'] * 5, sep='-+-')

for dtype in UTYPES + ITYPES:
    i = np.iinfo(dtype)
    print(f'{i.dtype.name:>12s}', end='')
    for size in SIZES:
        x = rng.integers(i.min, i.max, size=10**size, dtype=dtype)
        t0 = Bench('isint(x)', globals=globals()).autorange()
        t1 = Bench('(x % 1) == 0', globals=globals()).autorange()
        print(f' | {t1 / t0:>11.1f}x', end='')
    print()
print(*[12 * '='] * 5, sep='===')

print(f'{"Floats":^72s}')
print(*[12 * '-'] * 5, sep='-+-')
bs = '\\'
print(f'{bs.join(["dtype ", " N"]):>12s}', end='')
for size in SIZES:
    print(f' | {10**size:>12d}', end='')
print()
print(*[12 * '-'] * 5, sep='-+-')

for dtype in FTYPES:
    i = np.finfo(dtype)
    print(f'{i.dtype.name:>12s}', end='')
    for size in SIZES:
        half = 10**size // 2
        x = rng.uniform(-1000, 1000, size=10**size).astype(dtype)
        if dtype == np.float128:
            x[half:] *= rng.uniform(-1000, 1000, size=half)
        x[:half] = rng.integers(-1000, 1000, size=half)
        rng.shuffle(x)
        t0 = Bench('isint(x)', globals=globals()).autorange()
        t1 = Bench('(x % 1) == 0', globals=globals()).autorange()
        print(f' | {t1 / t0:>11.1f}x', end='')
    print()
print(*[12 * '='] * 5, sep='===')

