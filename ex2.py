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


def testStandardGaussian(dim, lr=0.05, iter=30000):
    mean = np.zeros(dim)
    cov = np.ones(dim)

    def log_gaussian(x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        log_density = norm.logpdf(x, mean, stdev)
        return np.sum(log_density, axis=-1)

    init_var_param = np.concatenate([np.zeros(D), np.ones(D)])
    approx = MFGaussian(D)
    model = Model(log_gaussian)


    obj_exclusivekl = ExclusiveKL(approx, model, 10)
    obj_RGE = RGE(approx, model, 10)

    def result(obj):
        currrent_time = time.time()
        resultsKL = bbvi(D, objective=obj, learning_rate=lr, n_iters=iter, init_var_param=init_var_param,
                         fixed_lr=True)
        time_cost_KL = time.time() - currrent_time
        result_para = results['opt_param']













    dt_string = now.strftime("%d-%m-%Y%H:%M:%S")
    name = "" + dt_string
    f = open("records.txt", "a")

    var_grad_KL = np.var(resultsKL['grad_history'][500:], axis=0)
    var_grad_RGE = np.var(resultsRGE['grad_history'][500:], axis=0)
    f.write("\n" + "variance of RGE" + name + "\n")
    f.write(str(var_grad))

    f.close()




def log_density(x):
    mu, log_sigma = x[:, 0], x[:, 1]
    sigma_density = norm.logpdf(log_sigma, 0, log_sigma_stdev)
    mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))
    return sigma_density + mu_density


mean = np.zeros(2)
stdev = np.ones(2)




init_var_param = np.concatenate([np.zeros(D), np.ones(D)])
approx = MFGaussian(D)
model = Model(log_density)
obj_exclusivekl = ExclusiveKL(approx, model, 10)
obj_RGE = RGE(approx, model, 10)
currrent_time = time.time()

results = bbvi(D, objective=obj_RGE, learning_rate=0.05, n_iters=30000, init_var_param=init_var_param,
               fixed_lr=True)
time_cost = time.time() - currrent_time
print(time_cost)

sns.set_style('white')
sns.set_context('notebook', font_scale=2, rc={'lines.linewidth': 2})

print(results['grad_history'].shape)

plt.plot(results['variational_param_history'][:, 0], '-b', label='mean1', linewidth=0.5)
plt.plot(results['variational_param_history'][:, 1], '-r', label='mean2', linewidth=0.5)
plt.plot(results['variational_param_history'][:, 2], '-g', label='log scale1', linewidth=0.5)
plt.plot(results['variational_param_history'][:, 3], '-y', label='log scale2', linewidth=0.5)
plt.xlabel('iteration')
plt.ylabel('value')
plt.legend(loc=2, prop={'size': 6})

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y%H:%M:%S")
t = time.time()
dt_string = dt_string + ""
name = "parameter_history_lr=5e-2_RGE" + dt_string
plt.title("parameter history lr=5e-2 RGE")
plt.savefig(name)

plt.close()

# plt.plot(np.log(results['value_history']))
plt.plot(results['grad_history'][500:, 0], '-b', label='grad mean1', linewidth=0.5)
plt.plot(results['grad_history'][500:, 1], '-r', label='grad mean2', linewidth=0.5)
plt.plot(results['grad_history'][500:, 2], '-g', label='grad log scale1', linewidth=0.5)
plt.plot(results['grad_history'][500:, 3], '-y', label='grad log scale2', linewidth=0.5)
var_grad = np.var(results['grad_history'][500:], axis=0)

name = "grad_history_lr=5e-2_RGE" + dt_string
f = open("records.txt","a")
f.write("\n" +"variace of" + name + "\n")
f.write(str(var_grad))
f.close()


plt.title("grad history lr=5e-2 RGE")
plt.legend(loc=2, prop={'size': 6})


plt.savefig(name)
plt.close()

plt.plot(np.linalg.norm(results['grad_history'][500:], axis=1), label='norm grad ', linewidth=0.5)
plt.title("norm of grad history lr=5e-2 RGE ")
plt.legend(loc=2, prop={'size': 6})

name = "norm_grad_history_lr=5e-2_RGE" + dt_string
plt.savefig(name)
