

from ._utils import ensure_2d, vectorize_if_needed

from autograd.extend import primitive, defvjp
import autograd.numpy as np



from autograd.numpy import numpy_boxes

__all__ = [
    'Model',
    'StanModel',
    'BridgeStanModel'
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


def _make_stan_log_density(fitobj):
    @primitive
    def log_density(x):
        return vectorize_if_needed(fitobj.log_prob, x)

    def log_density_vjp(ans, x):
        return lambda g: ensure_2d(g) * vectorize_if_needed(fitobj.grad_log_prob, x)
    defvjp(log_density, log_density_vjp)
    return log_density


def _make_bridgestan_log_density(model):
    @primitive
    def log_density(x):
        return vectorize_if_needed(model.log_density, x)

    def log_density_vjp(ans, x):
        return lambda g: vectorize_bs_if_needed(model.log_density_gradient, x)

        
    defvjp(log_density, log_density_vjp)
    return log_density

    
class StanModel(Model):
    """Class that encapsulates a PyStan model."""

    def __init__(self, fit):
        """
        Parameters
        ----------
        fit : `StanFit4model` object
        """
        self._fit = fit
        super().__init__(_make_stan_log_density(fit))

    def constrain(self, model_param):
        return self._fit.constrain_pars(model_param)


class BridgeStanModel(Model):

    def __init__(self, BridgeStanModelObject):
        super().__init__(_make_bridgestan_log_density(BridgeStanModelObject))
        self.BridgeStanModelObject = BridgeStanModelObject

    def constrain(self, model_param):
        return self.BridgeStanModelObject.param_constrain(model_param)

    def vectorized_gradient(self, param):
        param = np.atleast_2d(param)
        output = np.zeros_like(param)
        for i in range(param.shape[0]):
             a = self.BridgeStanModelObject.log_density_gradient(param[i, :])[1]
             output[i, :] = a
        return output
        
    
    def gradient(self,param):
        return self.BridgeStanModelObject.log_density_gradient(param)[1]
        
    
    def hessian(self, param):

        return self.BridgeStanModelObject.log_density_hessian(param)[2]

    def num_params(self):
        return self.BridgeStanModelObject.param_num()
    
    def value_and_grad(self, param):
        return self.BridgeStanModelObject.log_density_gradient(param)
        
    

