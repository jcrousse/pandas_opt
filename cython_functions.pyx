cdef int global_val = 0

def f(x):
    return x * (x - 1)


def integrate_f(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx


cdef double f_typed(double x) except? -2:
    return x * (x - 1)

cpdef double integrate_f_typed(double a, double b, int N):
    cdef int i
    cdef double s, dx
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_typed(a + i * dx)
    return s * dx

cimport numpy as np
import numpy as np
cpdef np.ndarray[double] apply_integrate_f(np.ndarray col_a, np.ndarray col_b,
                                           np.ndarray col_N):
    assert (col_a.dtype == np.float and col_b.dtype == np.float and col_N.dtype == np.int)
    cdef Py_ssize_t i, n = len(col_N)
    assert (len(col_a) == len(col_b) == n)
    cdef np.ndarray[double] res = np.empty(n)
    for i in range(len(col_a)):
        res[i] = integrate_f_typed(col_a[i], col_b[i], col_N[i])
    return res