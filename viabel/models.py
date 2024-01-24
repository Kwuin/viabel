import jax
import numpy as np

from ._utils import ensure_2d, vectorize_if_needed
from jax import random
import jax.numpy as jnp

__all__ = [
    'Model',
    'StanModel'
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


def _make_stan_log_density(bs_model):
    @jax.custom_vjp
    def log_density(x):
        return vectorize_if_needed(bs_model.log_density, x)

    def log_density_fwd(x):
        x = np.asarray(x, dtype="float64")
        vectorized_fun = jax.vmap(bs_model.log_density_gradient)
        result = vectorized_fun(x)
        return log_density(x), np.array([a[1] for a in result])

    def log_density_bwd(res, g):
        grad = res
        g = np.asarray(g, dtype=object)
        return ensure_2d(g) * grad,

    log_density.defvjp(log_density_fwd, log_density_bwd)
    return log_density

class StanModel(Model):
    """Class that encapsulates a BridgeStan model."""

    def __init__(self, bs_model):
        """
        Parameters
        ----------
        fit : `StanFit4model` object
        """
        self._fit = bs_model
        super().__init__(_make_stan_log_density(bs_model))

    def constrain(self, model_param):
        return self._fit.param_constrain(model_param)
        return self._fit.constrain_pars(model_param)
    
    

class SubsamplingModel(Model):
    def __init__(self, log_prior, log_likelihood, dataset, subsample_size, seed=42):
        self.seed = seed
        self.rng = random.PRNGKey(self.seed)
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

