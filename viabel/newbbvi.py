import jax

from viabel import FASO, MFGaussian, ExclusiveKL, RMSProp
from viabel.models import SubModel


def simpleTest(dimension, *, n_iters=10000, num_mc_samples=10, log_prior=None,
               log_likelihood=None, subsample_size=1, dataset=None, learning_rate=0.01,
               RMS_kwargs=dict(), FASO_kwargs=dict(), RAABBVI_kwargs=dict()):
    model = SubModel(log_prior, log_likelihood, dataset, subsample_size)

    approx = MFGaussian(dimension)
    objective = ExclusiveKL(approx, model, num_mc_samples)

    init_var_param = approx.init_param()
    base_opt = RMSProp(learning_rate, diagnostics=True, **RMS_kwargs)

    opt = FASO(base_opt, **FASO_kwargs)

    opt_results = opt.optimize(n_iters, objective, init_var_param)
    opt_results['objective'] = objective
    return opt_results

# if __name__=='__main__':
