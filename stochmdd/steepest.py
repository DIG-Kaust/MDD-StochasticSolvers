import numpy as np


def steepest_descent(G, d, niter=10, m0=None, a=None, tol=1e-10, savemhist=True):
    """Steepest descent  for minimizing d - Gm with square G. Can also solve 0.5||d - Gm||^2 
    by passing G.TG and G.Td
    """
    seta = True if a is None else False
    N,M = G.shape
    if m0 is None:
        m = np.zeros_like(d)
    else:
        m = m0.copy()
    if savemhist:
        mh = np.zeros((niter + 1, M))
    rh = np.zeros((niter + 1))
    ah = np.zeros(niter)
    if savemhist:
        mh[0] = m0.copy()
    r = d - G @ m
    rh[0] = np.linalg.norm(r)
    for i in range(niter):
        if seta:
            a = np.dot(r, r) / np.dot(r, G @ r)
        m = m + a*r
        if savemhist:
            mh[i + 1] = m.copy()
        ah[i] = a
        r = d - G @ m
        rh[i+1] = np.linalg.norm(r)
        if np.linalg.norm(r) < tol:
            break
    return m if not savemhist else mh[:i+2], rh[:i+2], ah[:i+1]

def steepest_descent1(G, d, niter=10, m0=None, a=None, tol=1e-10):
    """Steepest descent  for minimizing 0.5||d - Gm||^2
    """
    seta = True if a is None else False
    N,M = G.shape
    if m0 is None:
        m = np.zeros_like(d)
    else:
        m = m0.copy()
    mh = np.zeros((niter + 1, M))
    rh = np.zeros((niter + 1))
    ah = np.zeros(niter)
    mh[0] = m0.copy()
    r = G.H @ (d - G @ m)
    rh[0] = np.linalg.norm(r)
    for i in range(niter):
        if seta:
            a = np.dot(r, r) / np.dot(r, G.H @ G @ r)
        m = m + a*r
        mh[i + 1] = m.copy()
        ah[i] = a
        r = G.H @(d - G @ m)
        rh[i+1] = np.linalg.norm(r)
        if np.linalg.norm(r) < tol:
            break
    return mh[:i+2], rh[:i+2], ah[:i+1]