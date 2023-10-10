from models import SubModel
import jax.numpy as jnp
import jax


def log_prior1(x):
    return - jnp.sum(x ** 2)


def log_likelihood1(param, x):
    # param = param[jnp.newaxis, :]
    param = param[:, jnp.newaxis, :]
    # print(x.shape) data_num * D
    # print(param.shape) sample_num * D
    return - jnp.sum((x - param) ** 2, axis=2)


data = ((jnp.arange(50) + 1) / 20)[:, jnp.newaxis] * (jnp.arange(4) + 1)[jnp.newaxis, :]

# sm = SubModel()
sm = SubModel(log_prior1, log_likelihood1, data, 5, 1)

param = jnp.array([1.0, 2.0, 1.5, 1.0])
param = jnp.arange(5)[:, jnp.newaxis] * param
n = 30


def f(sm, param):
    return jnp.mean(sm(param))


g = jax.grad(f, argnums=1)
# print(g(sm, param))

# print(log_likelihood1(param, data))
#
# grad_f = jax.jacobian(sm)
# grad_f = jax.vmap(grad_f)
#
g_list = []
for i in range(n):
    g_list.append(g(sm, param))

g_list = jnp.array(g_list)
sm2 = SubModel(log_prior1, log_likelihood1, data, 50, 1)
# grad_f2 = jax.jacobian(sm2)
# grad_f2 = jax.vmap(grad_f2)

print(g(sm2, param))
# g_list = jnp.array(g_list)
print(jnp.mean(g_list, axis=0))
print(jnp.var(g_list, axis=0))
h = jax.hessian(f, argnums=1)
print(h(sm2, param))
