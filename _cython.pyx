cimport cython
from cython.parallel import prange
from libc.math cimport sin, cos, atan2, asin
cimport numpy as np
cimport numpy.math
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def getmap(int w, int h, np.ndarray[np.double_t, ndim=2] r):
    cdef double u0 = w / 2.
    cdef double v0 = h / 2.
    cdef double u2t = 2 * np.math.PI / w
    cdef double v2p = np.math.PI / h
    cdef double t2u = 1 / u2t
    cdef double p2v = 1 / v2p

    cdef int u, v
    cdef double phi, cos_phi, theta
    cdef double xd, yd, zd, xs, ys, zs

    cdef np.ndarray[np.float32_t, ndim=3] xymap = np.empty((h, w, 2), np.float32)

    for v in xrange(h):
        phi = (v - v0) * v2p
        cos_phi = cos(phi)
        yd = sin(phi)
        for u in xrange(w):
            theta = (u - u0) * u2t
            xd = cos_phi * sin(theta)
            zd = cos_phi * cos(theta)

            xs = r[0, 0] * xd + r[0, 1] * yd + r[0, 2] * zd
            ys = r[1, 0] * xd + r[1, 1] * yd + r[1, 2] * zd
            zs = r[2, 0] * xd + r[2, 1] * yd + r[2, 2] * zd

            xymap[v, u, 0] = atan2(xs, zs) * t2u + u0
            xymap[v, u, 1] = asin(ys) * p2v + v0

    return xymap

@cython.boundscheck(False)
@cython.wraparound(False)
def getmapomp(int w, int h, np.ndarray[np.double_t, ndim=2] r):
    cdef double u0 = w / 2.
    cdef double v0 = h / 2.
    cdef double u2t = 2 * np.math.PI / w
    cdef double v2p = np.math.PI / h
    cdef double t2u = 1 / u2t
    cdef double p2v = 1 / v2p

    cdef int u, v
    cdef double phi, cos_phi, theta
    cdef double xd, yd, zd, xs, ys, zs

    cdef np.ndarray[np.float32_t, ndim=3] xymap = np.empty((h, w, 2), np.float32)

    for v in prange(h, nogil=True):
        phi = (v - v0) * v2p
        cos_phi = cos(phi)
        yd = sin(phi)
        for u in xrange(w):
            theta = (u - u0) * u2t
            xd = cos_phi * sin(theta)
            zd = cos_phi * cos(theta)

            xs = r[0, 0] * xd + r[0, 1] * yd + r[0, 2] * zd
            ys = r[1, 0] * xd + r[1, 1] * yd + r[1, 2] * zd
            zs = r[2, 0] * xd + r[2, 1] * yd + r[2, 2] * zd

            xymap[v, u, 0] = atan2(xs, zs) * t2u + u0
            xymap[v, u, 1] = asin(ys) * p2v + v0

    return xymap
