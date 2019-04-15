import numpy as np
import tensorflow as tf


def linesearch(f, x, fullstep, expected_improve_rate, kl_bound, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)[0]
    for stepfrac in (.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval, newkl, newent = f(xnew)
        if any(newkl) > kl_bound:
            newfval += np.inf
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return True, xnew
    return False, x

def cg(f_Ax, b, cg_iters=10, callback=None, residual_tol=1e-10):
    """
        Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    return x

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

def unflatten_params(flat_params, shapes):
    """
    :param shapes: Shapes of the flat parameters
    :param flat_params: Flat parameters
    :return: Unflattened Parameters
    """
    unflat_params = []
    start = 0
    for i, shape in enumerate(shapes):
         size = np.prod(shape)
         param = tf.reshape(flat_params[start:start + size], shape)
         unflat_params.append(param)
         start += size

    return unflat_params


