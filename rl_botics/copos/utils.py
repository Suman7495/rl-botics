import numpy as np
import tensorflow as tf
import scipy.optimize


def linesearch1(f, x, fullstep, expected_improve_rate, kl_bound, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)[0]
    for stepfrac in (.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval, newkl = f(xnew)
        # if newkl > kl_bound:
        #     newfval += np.inf
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


def flatgrad(y, x):
    """
    :param y: Function y
    :param x: List of variable x
    :return: Flattened gradients (dy/dx)
    """
    grads = tf.gradients(y, x)
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


def jvp(f, x, u, v):
    """
        Computes the Jacobian-Vector Product: (df/dx)u
        See: https://j-towns.github.io/2017/06/12/A-new-trick.html
        and: https://github.com/renmengye/tensorflow-forward-ad/issues/2
    :param f: Function to be differentiated
    :param x: Variable
    :param u: Vector to be multiplied with
    :param v: Dummy variable (type tf.placeholder)
    :return: Jacobian Vector Product: (df/dx)u
    """
    g = tf.gradients(f, x, grad_ys=v)
    jvp = tf.gradients(g, v, grad_ys=u)
    return jvp


def optimize_dual(dual, x0, bounds):
    """
        Compute COPOS Discrete dual optimization to return eta and omega
    :param dual: Dual function which takes as input x = [eta, omega]
    :param x0: [eta, omega]
    :param bounds: Limits for eta, omega. e.g. bounds = ((1e-12, 1e+8), (1e-12, 1e+8))
    :return: res (see scipy.optimize), eta, omega
    """
    # res = scipy.optimize.minimize(get_dual, x0,
    #                               method='SLSQP',
    #                               jac=True,
    #                               bounds=((1e-12, 1e+8), (1e-12, 1e+8)),
    #                               options={'ftol': 1e-12})

    res = scipy.optimize.minimize(dual, x0,
                                  method='SLSQP',
                                  jac=True,
                                  bounds=bounds,
                                  options={'ftol': 1e-12})
    return res, res.x[0], res.x[1]

