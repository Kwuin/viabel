import autograd.numpy as np
import autograd.scipy.stats.norm as norm
from autograd.scipy.stats import multivariate_normal

from viabel import bbvi, MFStudentT, MFGaussian, Model, ExclusiveKL, RGE
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import date
from datetime import datetime
import os
import torch


from autograd import grad, hessian, jacobian, elementwise_grad, make_hvp, vector_jacobian_product, \
    hessian_vector_product


# number of dimensions


# A = np.random.rand(3, 3)
# A = 2 * np.ones([D, D])

# Create a symmetric matrix from the upper triangular part of A
# stdev = np.triu(A) + np.triu(A, 1).T


def correlated_gaussian(D, mean, cov, info, repeat, hessian_approx=False, KL=False):
    def log_gaussian(x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        log_pdf = multivariate_normal.logpdf(x, mean, cov)
        return log_pdf

    stdev = np.sqrt(1 / np.diag(np.linalg.inv(cov)))

    init_var_param = np.concatenate([2 * mean, 2 * np.log(stdev)])
    approx = MFGaussian(D)
    model = Model(log_gaussian)
    obj_RGE = RGE(approx, model, 10, hessian_approx=False)
    obj_RGE_approx = RGE(approx, model, 10, hessian_approx=True)

    obj_exclusivekl = ExclusiveKL(approx, model, 10)

    lr = 5e-1
    iter = 30000

    time_cost_record = []
    mean_error_record = []
    std_error_record = []
    iterations_record = []
    var_reduction_record = []
    elbo_record = []

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d%H:%M:%S")
    t = time.time()

    dt_string = dt_string + info
    file_name = "/Info" + info + ".txt"

    subdirectory = dt_string
    directory = os.path.join('plots', subdirectory)
    os.makedirs(directory)

    file_path = directory + file_name

    file = open(file_path, "a")
    # Write the data to the file
    file.write("mean" + str(mean) + "  ")
    file.write("cov" + str(cov) + "   ")
    file.write("stdev" + str(stdev))
    file.write("lr" + str(lr) + "   ")
    file.write("iteration  " + str(iter))
    file.close()

    for i in range(repeat):

        currrent_time = time.time()

        resultsRGE = bbvi(D, objective=obj_RGE_approx, learning_rate=lr, n_iters=iter, init_var_param=init_var_param,
                          fixed_lr=True, adaptive=True)
        time_cost_RGE = time.time() - currrent_time

        #if KL:
        currrent_time = time.time()

        resultsKL = bbvi(D, objective=obj_exclusivekl, learning_rate=lr, n_iters=iter, init_var_param=init_var_param,
                      fixed_lr=True, adaptive=True)

        time_cost_KL = time.time() - currrent_time

        # time
        time_cost = time_cost_KL, time_cost_RGE
        time_cost_record.append(time_cost)

        # print("Time" + str(time_cost))
        # variance of gradients

        var_grad = np.concatenate(
            [np.var(resultsKL['grad_history'][-100:], axis=0), np.var(resultsRGE['grad_history'][-100:], axis=0)])
        # print("max grad")
        # print(np.max(resultsRGE['grad_history']))
        # print(resultsRGE['grad_history'].shape)
        # print("var_grad" + str(var_grad))

        # reduction ratio of variance
        # print(np.mean(var_grad[:2 * D] - var_grad[2 * D:] / var_grad[:2 * D]))
        variance_reduce = np.mean((var_grad[:2 * D] - var_grad[2 * D:]) / var_grad[:2 * D]) * 100
        # print("Var Reduction " + str(variance_reduce) + "%")
        var_reduction_record.append(variance_reduce)

        # deviance of estimated parameters compared to the ground truth , L2 error
        mean_error = np.mean(((resultsKL['opt_param'][:D] - mean) / stdev) ** 2), \
                     np.mean(((resultsRGE['opt_param'][:D] - mean) / stdev) ** 2)

        # print(resultsRGE['opt_param'][D:])
        mean_error_record.append(mean_error)
        # print("mean error " + str(mean_error))

        scale_error = np.mean(((np.exp(resultsKL['opt_param'][D:]) - stdev) / stdev) ** 2), \
                      np.mean((((np.exp(resultsRGE['opt_param'][D:])) - stdev) / stdev) ** 2)

        std_error_record.append(scale_error)

        iterations_record.append(np.array([resultsKL['k_stopped'], resultsRGE['k_stopped']]))

        file = open(file_path, "a")
        # Write the data to the file

        plt.plot(resultsKL['value_history'], '-b', label='KL elbo', linewidth=0.5)
        plt.plot(resultsRGE['value_history'], '-r', label='RGE elbo', linewidth=0.5)
        file.write(" \n" + "KL elbo \n")
        file.write(str(resultsKL['value_history']))
        file.write(" \n" + "RGE elbo \n")
        file.write(str(resultsRGE['value_history']))

        plt.title("elbo history of" + str(D) + " dimension ")
        # plt.legend(loc=2, prop={'size': 6})

        filename = 'elbo_history  ' + str(i) + "  " + str(D) + 'dimension.png'
        filepath = os.path.join(directory, filename)

        plt.savefig(filepath)
        plt.close()

        for j in range(D):
            plt.plot(resultsRGE['variational_param_history'][:, j], label="RGE mean" + str(j), linewidth=0.5)
            file.write(" \n" + "RGE mean " + str(j) + " \n")
            file.write(str(resultsRGE['variational_param_history'][:, j]))

        plt.title("variational parameter history of RGE" + str(D) + " dimension ")
        # plt.legend(loc=2, prop={'size': 6})
        filename = "RGEparam_mean_history " + str(i) + "  " + str(D) + "dimension.png"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)
        plt.close()

        for j in range(D):
            plt.plot(resultsKL['variational_param_history'][:, j], label="exKL mean" + str(j), linewidth=0.5)
            file.write(" \n" + "KL mean " + str(j) + " \n")
            file.write(str(resultsKL['variational_param_history'][:, j]))

        plt.title("variational parameter history of exKL" + str(D) + " dimension ")
        # plt.legend(loc=2, prop={'size': 6})

        filename = "KLparam_mean_history " + str(i) + "  " + str(D) + "dimension.png"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)
        plt.close()

        for j in range(D, 2 * D):
            plt.plot(resultsKL['variational_param_history'][:, j], label="KL log scale" + str(j), linewidth=0.5)
            file.write(" \n" + " KL log scale " + str(j) + " \n")
            file.write(str(resultsKL['variational_param_history'][:, j]))

        plt.title("variational parameter history of KL" + str(D) + " dimension ")
        # plt.legend(loc=2, prop={'size': 6})
        filename = "KLparam_logscale_history " + str(i) + "  " + str(D) + "dimension.png"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)
        plt.close()

        for j in range(D, 2 * D):
            plt.plot(resultsRGE['variational_param_history'][:, j], label="RGE log scale" + str(j), linewidth=0.5)
            file.write(" \n" + " RGE log scale " + str(j) + " \n")
            file.write(str(resultsRGE['variational_param_history'][:, j]))

        plt.title("variational parameter history of RGE" + str(D) + " dimension ")
        # plt.legend(loc=2, prop={'size': 6})
        filename = "RGEparam_logscale_history " + str(i) + "  " + str(D) + "dimension.png"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)
        plt.close()

        for j in range(D):
            plt.plot(resultsKL['grad_history'][:, j], label="KL mean grad" + str(j), linewidth=0.5)
            file.write(" \n" + "KL mean grad " + str(j) + " \n")
            file.write(str(resultsRGE['grad_history'][:, j]))

        plt.title("mean grad history of KL" + str(D) + " dimension ")
        # plt.legend(loc=2, prop={'size': 6})

        filename = "KL_mean_grad_history " + str(i) + "  " + str(D) + "dimension.png"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)
        plt.close()

        # print(resultsRGE['grad_history'][:500,:])
        for j in range(D):
            plt.plot(resultsRGE['grad_history'][:, j], label="RGE mean grad" + str(j), linewidth=0.5)
            file.write(" \n" + "RGE mean grad" + str(j) + " \n")
            file.write(str(resultsRGE['grad_history'][:, j]))

        plt.title("mean grad history of RGE" + str(D) + " dimension ")
        # plt.legend(loc=2, prop={'size': 6})

        filename = "RGE_mean_grad_history " + str(i) + "  " + str(D) + "dimension.png"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)
        plt.close()

        for j in range(D, 2 * D):
            plt.plot(resultsKL['grad_history'][:, j], label="KL log scale grad" + str(j), linewidth=0.5)
            file.write(" \n" + "KL log scale grad " + str(j) + " \n")
            file.write(str(resultsKL['grad_history'][:, j]))

        plt.title("log scale grad history of KL" + str(D) + " dimension")
        # plt.legend(loc=2, prop={'size': 6})

        filename = "KL_log_scale_grad_history " + str(i) + "  " + str(D) + "dimension.png"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)
        plt.close()

        # print(resultsRGE['grad_history'][:500,:])
        for j in range(D, 2 * D):
            plt.plot(resultsRGE['grad_history'][:, j], label="RGE log scale grad" + str(j), linewidth=0.5)
            file.write(" \n" + "RGE log scale grad " + str(j) + " \n")
            file.write(str(resultsRGE['grad_history'][:, j]))

        plt.title("log scale grad history of RGE" + str(D) + " dimension ")
        # plt.legend(loc=2, prop={'size': 6})

        filename = "RGE_log_scale_grad_history " + str(i) + "  " + str(D) + "dimension.png"
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)
        plt.close()

        file.close()

        file = open(file_path, "a")
        # Write the data to the file
        file.write("paramter histories" + "\n")
        file.write(str(resultsKL['opt_param']) + "\n")
        file.write(str(resultsRGE['opt_param']) + "\n")
        file.close()

    time_cost_record = np.array(time_cost_record)
    mean_error_record = np.array(mean_error_record)
    std_error_record = np.array(std_error_record)

    # iterations_record = np.array(iterations_record)
    var_reduction_record = np.array(var_reduction_record)

    time_cost_record = np.round(np.mean(time_cost_record, axis=0), 3)
    mean_error_record = np.round(np.mean(mean_error_record, axis=0), 3)
    std_error_record = np.round(np.mean(std_error_record, axis=0), 3)
    # iterations_record = np.mean(iterations_record, axis=0)
    var_reduction_record = np.round(np.mean(var_reduction_record, axis=0), 3)

    print("average time cost" + str(time_cost_record))
    print("average mean error" + str(mean_error_record))
    print("average scale error" + str(std_error_record))
    print("average iteration time " + str(iterations_record))
    print("average varaince reduction" + str(var_reduction_record))

    file = open(file_path, "a")
    # Write the data to the file
    file.write("average time cost" + str(time_cost_record))
    file.write("average mean error" + str(mean_error_record))
    file.write("average scale error" + str(std_error_record))
    file.write("average iteration time " + str(iterations_record))
    file.write("average variance reduction" + str(var_reduction_record))
    file.close()


if __name__ == '__main__':
    D = 1

    # A = 5 * np.eye(D)
    A = np.arange(1, D + 1)
    B = 2 * np.ones([D, D])

    cov = np.triu(A) + np.triu(A, 1).T
    # mean = 3 * np.ones(D)
    mean = np.arange(1, D + 1)
    cov = A + B

    correlated_gaussian(D, mean, cov, "test_1_dim", 1)

#
# f = open("records.txt", "a")
#
# f.write("\n" + " results of " + name +"\n")
# f.write("\n" + "                             KL       RGE " + "\n")
# f.write(str(D) + "  dimensional diagonal Gaussian")
# f.write("\n" + "time cost                 " + str(time_cost))
# print("\n" + " var_grad of KL and RGE " + str(var_grad))
# f.write("\n" + "variance reduction rate   " + str(variance_reduce))
# f.write("\n" + "optimal parameter error   " + str(para_error))
# f.close()
