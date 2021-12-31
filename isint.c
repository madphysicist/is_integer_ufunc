#include <Python.h>
#include <math.h>
#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/halffloat.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "numpy/npy_common.h"
#include "isint.h"

/*
 * isint.c: ufunc wrapper for the `isint` functions.
 */


static PyMethodDef IsintMethods[] = {
    {NULL, NULL, 0, NULL}
};

/* The loop definitions must precede the PyMODINIT_FUNC. */


// TODO:
/**begin repeat
 * #type = half, float, double, longdouble#
 */
/*
static void @type@_isint(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    npy_intp n = dimensions[0];
    for(npy_intp i = 0; i < n; i++, in += in_step, out += out_step) {
        *((npy_bool *)out) = isint_@type@(*((npy_@type@ *)in));
    }
}
*/
/**end repeat */

static void half_isint(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    npy_intp n = dimensions[0];
    for(npy_intp i = 0; i < n; i++, in += in_step, out += out_step) {
        *((npy_bool *)out) = isint_half(*((npy_half *)in));
    }
}

static void float_isint(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    npy_intp n = dimensions[0];
    for(npy_intp i = 0; i < n; i++, in += in_step, out += out_step) {
        *((npy_bool *)out) = isint_float(*((npy_float *)in));
    }
}

static void double_isint(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    npy_intp n = dimensions[0];
    for(npy_intp i = 0; i < n; i++, in += in_step, out += out_step) {
        *((npy_bool *)out) = isint_double(*((npy_double *)in));
    }
}

static void longdouble_isint(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    npy_intp n = dimensions[0];
    for(npy_intp i = 0; i < n; i++, in += in_step, out += out_step) {
        *((npy_bool *)out) = isint_longdouble(*((npy_longdouble *)in));
    }
}

// TODO:
/**begin repeat
 * #type = float, double, longdouble#
 * #TYPE = FLOAT, DOUBLE, LONGDOUBLE#
 */
/*
static void c@type@_isint(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    npy_intp n = dimensions[0];
    for(npy_intp i = 0; i < n; i++, in += in_step, out += out_step) {
        *((npy_bool *)out) = ((((npy_c@type@ *)in)->imag == 0) && isint_@type@(((npy_c@type@ *)in)->real));
    }
}
*/
/**end repeat */
static void cfloat_isint(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    npy_intp n = dimensions[0];
    for(npy_intp i = 0; i < n; i++, in += in_step, out += out_step) {
        *((npy_bool *)out) = ((((npy_cfloat *)in)->imag == 0) && isint_float(((npy_cfloat *)in)->real));
    }
}

static void cdouble_isint(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    npy_intp n = dimensions[0];
    for(npy_intp i = 0; i < n; i++, in += in_step, out += out_step) {
        *((npy_bool *)out) = ((((npy_cdouble *)in)->imag == 0) && isint_double(((npy_cdouble *)in)->real));
    }
}

static void clongdouble_isint(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];
    npy_intp n = dimensions[0];
    for(npy_intp i = 0; i < n; i++, in += in_step, out += out_step) {
        *((npy_bool *)out) = ((((npy_clongdouble *)in)->imag == 0) && isint_longdouble(((npy_clongdouble *)in)->real));
    }
}

// Only one loop is necessary for integer types
static void int_isint(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    char *out = args[1];
    npy_intp out_step = steps[1];
    npy_intp n = dimensions[0];
    for(npy_intp i = 0; i < n; i++, out += out_step) {
        *((npy_bool *)out) = (npy_bool)true;
    }
}

/* This gives pointers to the loop functions */
PyUFuncGenericFunction funcs[] = {
    int_isint,
    int_isint,
    int_isint,
    int_isint,
    int_isint,
    int_isint,
    int_isint,
    int_isint,
    int_isint,
#ifdef NPY_INT128
    int_isint,
#endif // NPY_INT128
#ifdef NPY_UINT128
    int_isint,
#endif // NPY_UINT128
#ifdef NPY_INT256
    int_isint,
#endif // NPY_INT256
#ifdef NPY_UINT256
    int_isint,
#endif // NPY_UINT256
    half_isint,
    float_isint,
    double_isint,
    longdouble_isint,
    cfloat_isint,
    cdouble_isint,
    clongdouble_isint,
};

static char types[] = {
    NPY_BOOL, NPY_BOOL,
    NPY_INT8, NPY_BOOL,
    NPY_UINT8, NPY_BOOL,
    NPY_INT16, NPY_BOOL,
    NPY_UINT16, NPY_BOOL,
    NPY_INT32, NPY_BOOL,
    NPY_UINT32, NPY_BOOL,
    NPY_INT64, NPY_BOOL,
    NPY_UINT64, NPY_BOOL,
#ifdef NPY_INT128
    NPY_INT128, NPY_BOOL,
#endif // NPY_INT128
#ifdef NPY_UINT128
    NPY_UINT128, NPY_BOOL,
#endif // NPY_UINT128
#ifdef NPY_INT256
    NPY_INT256, NPY_BOOL,
#endif // NPY_INT256
#ifdef NPY_UINT256
    NPY_UINT256, NPY_BOOL,
#endif // NPY_UINT256
    NPY_HALF, NPY_BOOL,
    NPY_FLOAT, NPY_BOOL,
    NPY_DOUBLE, NPY_BOOL,
    NPY_LONGDOUBLE, NPY_BOOL,
    NPY_CFLOAT, NPY_BOOL,
    NPY_CDOUBLE, NPY_BOOL,
    NPY_CLONGDOUBLE, NPY_BOOL,
};

static void *data[sizeof(funcs) / sizeof(funcs[0])] = {NULL};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "isint_ufunc",
    NULL,
    -1,
    IsintMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_isint_ufunc(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if(m == NULL) return NULL;

    import_array();
    import_ufunc();
    import_umath();

    PyObject *isint = PyUFunc_FromFuncAndData(funcs, data, types,
                                              sizeof(funcs) / sizeof(funcs[0]),
                                              1, 1, PyUFunc_None, "isint",
                                              "isint_docstring", 0);

    if(PyModule_AddObject(m, "isint", isint)) {
        Py_XDECREF(isint);
        return NULL;
    }

    return m;
}
