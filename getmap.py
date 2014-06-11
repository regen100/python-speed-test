import numpy as np
import scipy as sp
import scipy.weave
import _cython
from math import sin, cos, atan2, asin

def python(w, h, r):
    u0 = w / 2.
    v0 = h / 2.
    u2t = 2 * np.pi / w
    v2p = np.pi / h
    t2u = 1 / u2t
    p2v = 1 / v2p
    
    xymap = np.empty((h, w, 2), np.float32)  
    
    for v in xrange(h):
        phi = (v - v0) * v2p;
        cos_phi = cos(phi);
        yd = sin(phi);
        for u in xrange(w):
            theta = (u - u0) * u2t;
            xd = cos_phi * sin(theta);
            zd = cos_phi * cos(theta);
            
            xs, ys, zs = r.dot([xd, yd, zd])
            
            xymap[v, u, 0] = atan2(xs, zs) * t2u + u0;
            xymap[v, u, 1] = asin(ys) * p2v + v0;
            
    return xymap

def numpy(w, h, r):
    v, u = np.indices((h, w))
    theta = (u - w / 2.) * 2 * np.pi / w
    phi = (v - h / 2.) * np.pi / h
    cos_phi = np.cos(phi)
    x = cos_phi * np.sin(theta)
    y = np.sin(phi)
    z = cos_phi * np.cos(theta)
    
    ray = np.dstack((x, y, z))
    ray = ray.reshape(-1, 3).dot(r.T)
    ray.shape = (h, w, 3)
    
    x, y, z = np.dsplit(ray, 3)
    theta = np.arctan2(x, z)
    phi = np.arcsin(y)
    u = theta * w / 2 / np.pi + w / 2.
    v = phi * h / np.pi + h / 2.
    xymap = np.dstack((u, v)).astype(np.float32)
    
    return xymap

def weave(w, h, r):
    code = \
    """
        const double u0 = w / 2.;
        const double v0 = h / 2.;
        const double u2t = 2 * NPY_PI / w;
        const double v2p = NPY_PI / h;
        const double t2u = 1 / u2t;
        const double p2v = 1 / v2p;
        
        for(int v = 0; v < h; v++)
        {
            const double phi = (v - v0) * v2p;
            const double cos_phi = cos(phi);
            const double yd = sin(phi);
            for(int u = 0; u < w; u++)
            {
                const double theta = (u - u0) * u2t;
                const double xd = cos_phi * sin(theta);
                const double zd = cos_phi * cos(theta);
                
                const double xs = R2(0, 0) * xd + R2(0, 1) * yd + R2(0, 2) * zd;
                const double ys = R2(1, 0) * xd + R2(1, 1) * yd + R2(1, 2) * zd;
                const double zs = R2(2, 0) * xd + R2(2, 1) * yd + R2(2, 2) * zd;
                
                XYMAP3(v, u, 0) = atan2(xs, zs) * t2u + u0;
                XYMAP3(v, u, 1) = asin(ys) * p2v + v0;
            }
        }
    """
    w = int(w)
    h = int(h)
    xymap = np.empty((h, w, 2), np.float32)
    sp.weave.inline(code, ['w', 'h', 'r', 'xymap'], headers=['<numpy/npy_math.h>'])
    return xymap

def cython(w, h, r):
    return _cython.getmap(w, h, r)

def cythonomp(w, h, r):
    return _cython.getmapomp(w, h, r)

