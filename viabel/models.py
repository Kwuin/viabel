from _utils import ensure_2d, vectorize_if_needed

from _utils import ensure_2d, vectorize_if_needed
import jax
import jax.numpy as jnp
from jax import grad, jit, random

__all__ = [
    'Model',
    'SubsamplingModel'
]


class Model(object):
    """Base class for representing a model.

    Does not support tempering. It can be overridden in part or in whole by
    classes that inherit it. See ``StanModel`` for an example."""

    def __init__(self, log_density):
        """
        Parameters
        ----------
        log_density : `function`
            Function for computing the (unnormalized) log density of the model.
            Must support automatic differentiation with ``autograd``.
        """
        self._log_density = log_density

    def __call__(self, model_param):
        """Compute (unnormalized) log density of the model.

        Parameters
        ----------
        model_param : `numpy.ndarray`, shape (dim,)
            Model parameter value

        Returns
        -------
        log_density : `float`
        """

        return self._log_density(model_param)

    def constrain(self, model_param):
        """Construct dictionary of constrained parameters.

        Parameters
        ----------
        model_param : `numpy.ndarray`, shape (dim,)
            Model parameter value

        Returns
        -------
        constrained_params : `dict`

        Raises
        ------
        NotImplementedError
            If constrained parameterization is not supported.
        """
        raise NotImplementedError()

    @property
    def supports_tempering(self):
        """Whether the model supports tempering."""
        return False

    def set_inverse_temperature(self, inverse_temp):
        """If tempering supported, set inverse temperature.

        Parameters
        ----------
        inverse_temp : `float`

        Raises
        ------
        NotImplementedError
            If tempering is not supported.
        """
        raise NotImplementedError()


#
# def _make_stan_log_density(fitobj):
#     @primitive
#     def log_density(x):
#
#         return vectorize_if_needed(fitobj.log_density, x)
#
#     def log_gradient(x):
#         _, gradient = fitobj.log_density_gradient(x)
#         return gradient
#
#     def log_density_vjp(ans, x):
#         return lambda g: ensure_2d(g) * vectorize_if_needed(fitobj.grad_log_prob, x)
#     defvjp(log_density, log_density_vjp)
#     return log_density


# class StanModel(Model):
#     """Class that encapsulates a PyStan model."""
#
#     def __init__(self, fit):
#         """
#         Parameters
#         ----------
#         fit : `StanFit4model` object
#         """
#         self._fit = fit
#
#         super().__init__(_make_stan_log_density(fit))
#
#     def constrain(self, model_param):
#
#         return self._fit.constrain_pars(model_param)

class SubModel(Model):
    seed = 42
    rng = random.PRNGKey(seed)

    def __init__(self, log_prior, log_likelihood, dataset, subsample_size, new_seed=42):
        if not new_seed == SubModel.seed:
            SubModel.seed = new_seed
            SubModel.rng = random.PRNGKey(SubModel.seed)

        def posterior_func(x):
            return SubModel.posterior(x, log_prior, log_likelihood, dataset, subsample_size)

        super().__init__(posterior_func)

    @staticmethod
    def posterior(x, prior, model, dataset, subsample_size):
        SubModel.rng, sub_rng = random.split(SubModel.rng)
        subsample_indices = random.choice(sub_rng, dataset.shape[0], shape=[subsample_size], replace=False)
        subsample_data = dataset[subsample_indices]
        # print(subsample_data.shape) #10,4

        likelihood = (jnp.shape(dataset)[0] / subsample_size) * jnp.sum(model(x, subsample_data), axis=-1)

        return likelihood + prior(x)


class SubsamplingModel(Model):
    def __init__(self, log_prior, log_likelihood, dataset, subsample_size, seed=42):
        self.seed = seed
        self.rng = random.PRNGKey(SubModel.seed)
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.dataset = dataset
        self.subsample_size = subsample_size

        super().__init__(self.posterior)

    def posterior(self, x):
        self.rng, sub_rng = random.split(self.rng)
        subsample_indices = random.choice(sub_rng, self.dataset.shape[0], shape=[self.subsample_size], replace=False)
        subsample_data = self.dataset[subsample_indices]
        # print(subsample_data.shape) #10,4
        likelihood = (jnp.shape(self.dataset)[0] / self.subsample_size) * jnp.sum(
            self.log_likelihood(x, subsample_data), axis=-1)

        return likelihood + self.log_prior(x)
