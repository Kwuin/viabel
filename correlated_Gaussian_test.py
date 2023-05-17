from Correlated_Gaussian import correlated_gaussian
from Covariance_Matrix_Factory import diagonal_matrix, block_diagonal, block_random_diag, constant_diagonal_matrix, \
    random_covariance, toeplitz_cov, hetero_cov
import numpy as np
import cProfile
import pstats
from pstats import SortKey
import re

test_dimensions = [5, 10, 50, 100]


def constant_diagonal():
    for i in test_dimensions:
        mean = np.arange(1, i + 1)
        correlated_gaussian(i, mean, constant_diagonal_matrix(i, 5), "constant_diagonal", 1, hessian_approx=False)


def block_test():
    for i in test_dimensions:
        mean = np.arange(1, i + 1)
        correlated_gaussian(i, mean, 10000 * block_random_diag(i, i//3 + 1), "block_diagonal", 1, hessian_approx=False)


def toeplitz_test():
    for i in test_dimensions:
        mean = np.arange(1, i + 1)
        correlated_gaussian(i, mean, 10000 * toeplitz_cov(i, 0.8), "toeplitz_diagonal", 1, hessian_approx=False)


def random_test():
    for i in test_dimensions:
        mean = np.arange(1, i + 1)
        correlated_gaussian(i, mean, 10000 * random_covariance(i), "random", 1, hessian_approx=False)


def hetero_test():
    for i in test_dimensions:
        mean = np.arange(1, i + 1)
        correlated_gaussian(i, mean, hetero_cov(i, np.exp(np.arange(1,i + 1))), "hetero_exp", 1, hessian_approx=False)


if __name__ == '__main__':
    # D = 5
    # mean = np.arange(1, D + 1)
    # correlated_gaussian(D, mean, 100 * toeplitz_cov(D, 0.5), "toeplitz_diagonal", 1)
    D = 10
    mean = np.arange(1, D + 1)

    cProfile.run('correlated_gaussian(D, mean, 100 * toeplitz_cov(D, 0.5), "toeplitz_diagonal", 1, '
                 'hessian_approx=False)', 'restats')
    p = pstats.Stats('restats')
    p.strip_dirs().sort_stats(-1).print_stats()
    p.sort_stats(SortKey.CUMULATIVE).print_stats(50)

