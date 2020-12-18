"""
APGD: Accelerated Proximal Gradient Descent.
Author: rkterence@zju.edu.cn
Written with python 3.7.2
"""

from numpy.linalg import norm  # the default input is 2-norm or frobenius norm.
import numpy as np
import warnings


def mixed_norm(X, p, q):
    """
    Calculate the L(p, q) norm of X.
    The input should be 2darray.
    """
    if not isinstance(X, np.ndarray):
        raise RuntimeError("Wrong input")
    if len(X.shape) == 1:
        return norm(norm(X, ord=p), ord=q)
    elif len(X.shape) == 2:
        return norm(norm(X, axis=1, ord=p), ord=q)
    else:
        raise RuntimeError("Unsupported dimension of X.")


def prox(constrain_type, constrain_lambda, x):
    def prox_l1(x, l):
        return np.sign(x) * np.maximum(np.abs(x) - l, 0)
    def prox_l2_1(x, l):
        return np.maximum(1-l/norm(x, axis=1), 0).reshape(-1, 1) * x

    if constrain_type is None:
        return x
    elif constrain_type == "l1":  # L_1 norm
        return prox_l1(x, constrain_lambda)
    elif constrain_type == 'l21':  # L_{2,1} norm
        return prox_l2_1(x, constrain_lambda)
    else:
        raise RuntimeError("Unsupported constrain type")


def g(constrain_type, x):
    """
    calculate the constrain part of cost function according to the type.
    """
    if constrain_type is None:
        return 0
    elif constrain_type == "l1": # L1 norm
        return np.sum(np.abs(x))
    elif constrain_type == 'l21':  # L_{2,1} norm
        return mixed_norm(x, 2, 1)


def line_search(f, constrain_type, constrain_lambda, grad, x, step, beta=0.5):
    while True:
        z = prox(constrain_type, constrain_lambda * step, x - step * grad(x))
        cost_hat = f(x) + np.dot(z.ravel() - x.ravel(), grad(x).ravel()) + 1/(2*step)*norm(z - x) + constrain_lambda * g(constrain_type, z)
        if cost_hat >= f(z) + constrain_lambda * g(constrain_type, z):
            break
        step *= beta
    return step, z


def apg(f, constrain_type, constrain_lambda, grad, x_init=None, 
                                  lipschitz=None, step=1, loop_tol=1e-6, max_iter=500,
                                  verbose=False):
    """
    Accelerated Proximal Gradient Method.
    :param f: cost function of f(x) in min{ f(x) + g(x) }
    :param constrain_type: type of cost function of non-smooth part g(x) in min { f(x) + \lambda g(x) }
    :param constrain_lambda: the coefficient of constrain part.
    :param grad: gradient function of f(x).
    :param x_init: the initial value of x
    :param lipschitz: lipschitz constant. if not given, line search method will be exploited.
    :param step: the initial step of line search.
    :param loop_tol: the tolerance of final result
    :param max_iter: maximum number of iterations
    :return: the final calculated value of x.
    """
    x = x_init
    x_old = np.zeros_like(x)   # x_old's initial value doesn't matter
    iter = 0
    while True:
        omega = iter / (iter + 3)
        y = x + omega * (x - x_old)
        if lipschitz is None:
            step, z = line_search(f, constrain_type, constrain_lambda, grad, y, step=step, beta=0.5)
        else:
            L_inv = 1 / lipschitz
            z = prox(constrain_type, constrain_lambda / L_inv, y - L_inv * grad(y))

        x_old = x
        x = z  # update x by z
        if norm(x - x_old) <= loop_tol:
            break
        if iter >= max_iter:
            warnings.warn("max_iter exceeded, if the Lipschitz constant is not given, "
                          "consider set it")
            break
        iter += 1
        if verbose:
            print('Iter: %d\tCurrent x: ' % (iter+1), end='')
            print(x)
    return x


# Below is the testing part
if __name__ == "__main__":
    def test_f1(x):
        return x[0]**2 + 2*x[1]**2 + 3*x[2]**2
    def test_grad1(x):
        return np.array([2*x[0], 4*x[1], 6*x[2]])
    def test_f2(x):
        return np.sum([1, 4, 1, 4] @ x.ravel()**2)
    def test_grad2(x):
        return np.array([[2, 8], [2, 8]]) * x
    def test_f3(x):
        return -200 * np.exp(-0.2 * norm(x))
    def test_grad3(x):
        return 40 * np.exp(-0.2 * norm(x)) / norm(x) * x
    # x_init = np.array([50, 2, 3])
    # x_init = np.array([[1, 100], [20, -43]])
    x_init = np.array([-10, 20])
    x = apg(f=test_f3, constrain_type=None, 
            constrain_lambda=0.1, grad=test_grad3,
            x_init=x_init, lipschitz=None, 
            step=10, loop_tol=1e-6, max_iter=2000,
            verbose=True)
