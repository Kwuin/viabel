import profile
import cProfile
import pstats

from autograd import hessian_vector_product, make_hvp, numpy as np
from autograd.scipy.stats import multivariate_normal
from Covariance_Matrix_Factory import hetero_cov
import time

# mean = 3 * np.ones(D)
D = 5
mean = np.arange(1, D + 1)
cov = hetero_cov(5, np.arange(1, D + 1))


def log_gaussian(x):
    if x.ndim == 1:
        x = x[np.newaxis, :]
    log_pdf = multivariate_normal.logpdf(x, mean, cov)
    return log_pdf


def function(D, N):
    n = np.random.randn(100, D)
    m = np.ones(D)
    ahvp = make_hvp(log_gaussian)(m)
    b = np.array([ahvp[0](e) for e in n])
    return b


# b = np.array([bhvp(m,e) for e in n])
cProfile.run('function(5,100)', 'profile_stats')
p = pstats.Stats('profile_stats')
p.strip_dirs().sort_stats('cumulative').print_stats(10)


