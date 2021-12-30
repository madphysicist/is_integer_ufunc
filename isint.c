#include <Python.h>
#include <math.h>

#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"

#include "isint.h"

/*
 * isint.c: ufunc wrapper for the `isint` functions.
 */


static PyMethodDef IsintMethods[] = {
    {NULL, NULL, 0, NULL}
};

/* The loop definitions must precede the PyMODINIT_FUNC. */

static void long_double_isint(char **args, const npy_intp *dimensions,
                              const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    for(char *end = in + dimensions[0] * in_step; in < end; in += in_step, out += out_step) {
        *((npy_bool *)out) = isint_longdouble(*((long double *)in));
    }
}

static void double_isint(char **args, const npy_intp *dimensions,
                         const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    for(char *end = in + dimensions[0] * in_step; in < end; in += in_step, out += out_step) {
        *((npy_bool *)out) = isint_double(*((double *)in));
    }
}

static void float_isint(char **args, const npy_intp *dimensions,
                        const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    for(char *end = in + dimensions[0] * in_step; in < end; in += in_step, out += out_step) {
        *((npy_bool *)out) = isint_float(*((float *)in));
    }
}

static void half_isint(char **args, const npy_intp *dimensions,
                       const npy_intp *steps, void *data)
{
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    for(char *end = in + dimensions[0] * in_step; in < end; in += in_step, out += out_step) {
        *((npy_bool *)out) = isint_half(*((npy_half *)in));
    }
}

/*This gives pointers to the above functions*/
PyUFuncGenericFunction funcs[4] = {half_isint,
                                   float_isint,
                                   double_isint,
                                   long_double_isint};

static char types[8] = {NPY_HALF, NPY_BOOL,
                        NPY_FLOAT, NPY_BOOL,
                        NPY_DOUBLE,NPY_BOOL,
                        NPY_LONGDOUBLE, NPY_BOOL};
static void *data[4] = {NULL, NULL, NULL, NULL};

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
    import_umath();

    PyObject *isint = PyUFunc_FromFuncAndData(funcs, data, types, 4, 1, 1,
                                              PyUFunc_None, "isint",
                                              "isint_docstring", 0);

    PyObject *mod_dict = PyModule_GetDict(m);

    PyDict_SetItemString(mod_dict, "isint", isint);
    Py_DECREF(isint);

    return m;
}
