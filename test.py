import getmap
import numpy as np
w = 3584
h = w / 2
r = np.array([
    [0.61566148, -0.78369395, 0.08236955],
    [0.78801075, 0.61228882, -0.06435415],
    [0, 0.10452846, 0.9945219],
])

def python():
    return getmap.python(w, h, r)

def numpy():
    return getmap.numpy(w, h, r)

def weave():
    return getmap.weave(w, h, r)

def cython():
    return getmap.cython(w, h, r)

def cythonomp():
    return getmap.cythonomp(w, h, r)
