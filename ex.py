import autograd.numpy as np
import autograd.scipy.stats.norm as norm
from viabel import bbvi, MFStudentT, MFGaussian, Model, ExclusiveKL, RGE
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import date
from datetime import datetime
import torch

from autograd import grad, hessian, jacobian, elementwise_grad, make_hvp, vector_jacobian_product, \
    hessian_vector_product

D = 2  # number of dimensions
log_sigma_stdev = 1.  # 1.35


def log_density(x):
    mu, log_sigma = x[:, 0], x[:, 1]
    sigma_density = norm.logpdf(log_sigma, 0, log_sigma_stdev)
    mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))
    return sigma_density + mu_density


mean = np.zeros(2)
stdev = np.ones(2)
ground_truth = np.concatenate([mean, stdev])


def log_gaussian(x):
    if x.ndim == 1:
        x = x[np.newaxis, :]
    log_pdf = norm.logpdf(x, mean, stdev)
    return np.sum(log_pdf, axis=-1)


init_var_param = np.concatenate([np.zeros(D), np.ones(D)])
approx = MFGaussian(D)
model = Model(log_gaussian)
obj_exclusivekl = ExclusiveKL(approx, model, 10)
obj_RGE = RGE(approx, model, 10)

lr = 0.05
iter = 30000

currrent_time = time.time()

resultsKL = bbvi(D, objective=obj_exclusivekl, learning_rate=lr, n_iters=iter, init_var_param=init_var_param,
                 fixed_lr=True)
time_cost_KL = time.time() - currrent_time

currrent_time = time.time()

resultsRGE = bbvi(D, objective=obj_RGE, learning_rate=lr, n_iters=iter, init_var_param=init_var_param,
                  fixed_lr=True)
time_cost_RGE = time.time() - currrent_time

# time
time_cost = np.round(time_cost_KL, 3), np.round(time_cost_RGE, 3)

# variance of gradients
var_grad = np.var(resultsKL['grad_history'][500:], axis=0), np.var(resultsRGE['grad_history'][500:], axis=0)

# reduction ratio of variance
variance_reduce = np.round(np.mean((var_grad[0] - var_grad[1]) / var_grad[0]), 5) * 100

# deviance of estimated parameters compared to the ground truth , L2 error
para_error = np.round(np.sum((resultsKL['opt_param'] - ground_truth) ** 2),3), \
             np.round(np.sum((resultsRGE['opt_param'] - ground_truth) ** 2),3)

sns.set_style('white')
sns.set_context('notebook', font_scale=2, rc={'lines.linewidth': 2})

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y%H:%M:%S")
t = time.time()
dt_string = dt_string + ""

# plt.plot(np.log(results['value_history']))

name = str(D) + " dimension diagonal Gaussian " + " lr=" + str(lr) + " iter=" + str(iter) + dt_string

f = open("records.txt", "a")

f.write("\n" + " results of " + name +"\n")
f.write("\n" + "                             KL       RGE " + "\n")
f.write(str(D) + "  dimensional diagonal Gaussian")
f.write("\n" + "time cost                 " + str(time_cost))
print("\n" + " var_grad of KL and RGE " + str(var_grad))
f.write("\n" + "variance reduction rate   " + str(variance_reduce))
f.write("\n" + "optimal parameter error   " + str(para_error))
f.close()
