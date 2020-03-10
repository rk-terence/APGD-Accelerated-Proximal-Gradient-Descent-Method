"""
APGD: Accelerated Proximal Gradient Descent.
Author: rkterence@zju.edu.cn
"""

from numpy.linalg import norm
import numpy as np


def norm2(X):
    """
    根据X的形状计算frobenius范数或者二范数。
    """
    if len(X.shape) == 1:
        return norm(X, 2)
    elif len(X.shape) == 2:
        return norm(X, 'fro')
    else:
        print("Error input")
        raise


def mixed_norm(X, p, q):
    """
    Calculate the L(p, q) norm of X.
    """
    return norm([norm(row, p) for row in X], q)


def prox(constrain_type, constrain_lambda, x):
    def prox_l1(x, l):
        return np.sign(x) * np.maximum(np.abs(x) - l, 0)

    if constrain_type is None:
        return x
    elif constrain_type == "l1": # L1 norm
        return prox_l1(x, constrain_lambda)


def g(constrain_type, x):
    """
    calculate the constrain part of cost function according to the type.
    """
    if constrain_type is None:
        return 0
    elif constrain_type == "l1": # L1 norm
        return np.sum(np.abs(x))


def line_search(f, constrain_type, constrain_lambda, grad, x, step, beta=0.5):
    while True:
        z = prox(constrain_type, constrain_lambda * step, x - step * grad(x))
        cost_hat = f(x) + np.dot(z - x, grad(x)) + 1/(2*step)*norm2(z - x) + constrain_lambda * g(constrain_type, z)
        if cost_hat >= f(z) + constrain_lambda * g(constrain_type, z):
            break
        step *= beta
    return step, z


def accelerated_proximal_gradient(f, constrain_type, constrain_lambda, grad, x_init, lipschitz=None, step=1,
                                  loop_tol=1e-6, max_iter=500):
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
        print("Current x: ", x)
        if norm2(x - x_old) <= loop_tol:
            break
        if iter >= max_iter:
            print("max_iter exceeded...")
            break
        iter += 1
    print("Iter =", iter)
    return x


"""
Below is the testing part:
$$
cost = 10x_1^2 + 50x_2^2 + 0.1 \cdot \sum_{i=1}^2 |x|
$$ 
"""


def test_f(x):
    return 10*int(x[0])**2 + 50*int(x[1])**2


def test_grad(x):
    return np.array([20*x[0], 100*x[1]])


if __name__ == "__main__":
    x_init = np.array([50, 75], dtype='int64')  # int64 to avoid data overflow in numpy
    x = accelerated_proximal_gradient(f=test_f, constrain_type='l1', constrain_lambda=0.1, grad=test_grad,
                                      x_init=x_init, lipschitz=None, step=10, loop_tol=1e-6, max_iter=50000)
    print(x)
