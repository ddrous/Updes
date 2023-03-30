#%%

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
import os
from functools import partial
import matplotlib.pyplot as plt

# import functools
from updec.utils import plot, dataloader, make_dir, random_name
from updec.cloud import SquareCloud
import time


EXPERIMENET_ID = random_name()
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)
KEY = jax.random.PRNGKey(46)     ## Use same random points for all iterations

Nx = 10
Ny = Nx
BATCH_SIZE = Nx*Ny // 10

#%%

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        # inputs = jnp.concatenate([x, t], axis=-1)
        y = nn.Dense(2)(x)
        for _ in range(0, 8):
            y = nn.Dense(20)(y)
            y = nn.tanh(y)          ## Don't use ReLU !!!!
        return nn.Dense(1)(y)

def init_flax_params(net:nn.Module):
    # init_data = jnp.ones((1,2))
    init_data = jnp.ones((2,))
    params = net.init(KEY, init_data)
    print(net.tabulate(KEY, init_data, depth=8, console_kwargs={"force_jupyter":False}))
    return params

pinn = MLP()
params = init_flax_params(pinn)




#%%


facet_types={"South":"d", "West":"d", "North":"d", "East":"d"}
cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=None)

cloud.visualize_cloud()
# create 4D tensor with batch dimension in addition to space, time, and channel

x_in = cloud.sorted_nodes[:cloud.Ni, :]
dataloader_in = dataloader(x_in, BATCH_SIZE, KEY)       ## For training

# u_pinn = pinn.apply(params, next(dataloader_in))
u_pinn = pinn.apply(params, cloud.sorted_nodes)         ## For plotting

print("Size of u_pinn output: "+format(u_pinn.shape))

print("Randomly initialized network state:")
cloud.visualize_field(u_pinn, cmap="jet", projection="2d", title="Untrained NN", figsize=(6,5));


#%%


d_north = jax.vmap(lambda x: jnp.sin(jnp.pi * x[0]), in_axes=(0,), out_axes=0)
d_zero = jax.vmap(lambda x: 0., in_axes=(0,), out_axes=0)

boundary_conditions = {"South":d_zero, "West":d_zero, "North":d_north, "East":d_zero}

x_bc = []
u_bc = []

for f_id in cloud.facet_types.keys():
    bd_node_ids = jnp.array(cloud.facet_nodes[f_id])
    x_id = cloud.sorted_nodes[bd_node_ids]
    u_id = boundary_conditions[f_id](x_id)

    x_bc.append(x_id)
    u_bc.append(u_id)

x_bc = jnp.concatenate(x_bc, axis=0)
u_bc = jnp.concatenate(u_bc, axis=0)[:, jnp.newaxis]

# dataloader_bc = dataloader(jnp.concatenate([x_bc, u_bc], axis=-1), BATCH_SIZE, KEY)     ## TODO For training Do we need this ?

fig, ax = plt.subplots(1, 4, figsize=(4*4,3))
for i, f_id in enumerate(cloud.facet_types.keys()):
    x = jnp.linspace(0,1,100)[:, jnp.newaxis]
    ax[i].plot(x, boundary_conditions[f_id](x), label=f_id)
    ax[i].set_ylim(0-1e-1,1+1e-1)
    ax[i].legend()


#%%


## Optimizer
scheduler = optax.linear_schedule(init_value=1e-1, end_value=1e-5, transition_steps=10000)
optimizer = optax.sgd(learning_rate=scheduler)

## Flax training state
state = train_state.TrainState.create(apply_fn=pinn.apply,
                                        params=params,
                                        tx=optimizer)



#%%

@jax.jit
def u(x, params):
    print("test u shape", pinn.apply(params, x).shape)
    return pinn.apply(params, x).reshape(-1)



@jax.jit
def gradu(params):
    print("test shape", jax.jacfwd(u)(x, params).shape)
    fgrad = jax.grad(lambda *args: test_func(jnp.array(args)), argnums=0)
    # return jnp.trace(jax.jacfwd(jax.grad(u))(x, params))
    ufunc = lambda x: pinn.apply(params, x)
    return jax.vmap(jax.jacfwd(ufunc))

# @jax.jit
# def lapu(x, params):
#     # print("test shape", jax.jacfwd(u)(x, params).shape)
#     # return jnp.trace(jax.jacfwd(jax.grad(u))(x, params))
#     # return jnp.trace(jax.jacfwd(jax.grad(u))(x, params))
#     ufunc = lambda y: pinn.apply(params, y).reshape(-1)
#     # lapfunc = jax.jacfwd(jax.jacfwd(ufunc))
#     gradfunc = jax.jacfwd(ufunc)
#     # gradfuncx = jax.jacfwd(ufunc)[...,0]
#     # gradfuncy = jax.jacfwd(ufunc)[...,1]

#     # lapfuncx = jax.jacfwd(gradfuncx)[...,0]
#     lapfunc = jax.jacfwd(gradfunc)

#     print("test shape", gradfunc(x).shape, lapfunc(x).shape)
#     # return jnp.trace(lapfunc(x))
#     # return lapfuncx[x] + lapfuncy[x]
#     return lapfunc(x)



@jax.jit
def lapu(x, params):
    # print("test shape", jax.jacfwd(u)(x, params).shape)
    # return jnp.trace(jax.jacfwd(jax.grad(u))(x, params))
    # return jnp.trace(jax.jacfwd(jax.grad(u))(x, params))
    ufunc = lambda y: pinn.apply(params, y).reshape(-1)
    lapfunc = jax.jacfwd(jax.jacfwd(ufunc))
    # lapfunc = jax.jacfwd(ufunc)
    print("test shape", lapfunc(x).shape)
    return jnp.trace(lapfunc(x))

def init_op_vmap(state):
    # init_data = jnp.ones((1,2))
    init_data = jnp.ones((3,2))
    return lapu(init_data, state.params)
print(init_op_vmap(state).shape)



def residual(params, x):
    """ Physics-based loss function with Burgers equation """
    # val = u(x, params)
    # lap = jax.vmap(lapu, in_axes=(0, None), out_axes=0)(x, params)
    return lapu(x, params)

# u_vec = jax.vmap(u, in_axes=(0, None), out_axes=0)
# residual_vec = jax.vmap(residual, in_axes=(None, 0), out_axes=0)

def loss_fn(params, x_in, x_bc, u_bc):
    # u_bc_pred = pinn.apply(params, x_bc)                  ## TODO ERROR !! This is never vmapped
    # u_bc_pred = u_vec(params, x_bc)                  ## TODO ERROR !! This is never vmapped
    u_bc_pred = u(x_bc, params)                  ## TODO ERROR !! This is never vmapped
    loss_bc = optax.l2_loss(u_bc_pred - u_bc)        ## Data loss

    res = residual(params, x_in)
    # res = residual_vec(params, x_in)
    loss_in = optax.l2_loss(res)                    ## Residual loss

    print("shapes:", u_bc_pred.shape, res.shape)

    eps = 1.00      ## TODO Use Bayesian optimisation here !
    return jnp.mean(loss_in) + eps*jnp.mean(loss_bc)


#%%

from jax.config import config
# config.update('jax_enable_x64', True)

# Create simple data
x = jnp.arange(10, dtype=jnp.float32)

# Create three functions, increasing in complexity
f = lambda x: jnp.array([jnp.dot(x, 6*jnp.ones_like(x))**2])

# def egrad(g):
#   def wrapped(x, *rest):
#     y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
#     x_bar, = g_vjp(np.ones_like(y))
#     return x_bar
#   return wrapped

def egrad(g):
    return jax.vmap(jax.jacfwd(g))

# Print results
print("Function f")
print(egrad(f)(x))
print("")



#%%

# @jax.jit
# @partial(jax.vmap, in_axis=(None, 0, 0, 0), out_axis=(0,0,0))
def train_step(state, x_in, x_bc, u_bc):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, x_in, x_bc, u_bc)
    state = state.apply_gradients(grads=grads)
    return state, loss



epochs = 1000
history_loss = []   # data for plotting
if os.path.isfile(DATAFOLDER+"pinn_checkpoint_"+str(state.step)):
    print("Found existing network, loading & skipping training")
    state = checkpoints.restore_checkpoint(ckpt_dir=DATAFOLDER, prefix="pinn_checkpoint_", target=state)
else:
    print("Training from scratch")
    for epoch in range(epochs+1):

        loss_epch = 0.
        for i, batch in enumerate(dataloader_in):

            state, loss = train_step(state, batch, x_bc, u_bc)
            loss_epch += loss
        history_loss.append(loss_epch)

        if epoch<3 or epoch%100==0:
            print("Epoch: %-5d                Loss: %.6f" % (epoch, loss_epch))

checkpoints.save_checkpoint(DATAFOLDER, prefix="pinn_checkpoint_", target=state, step=state.step, overwrite=True)
print("Training done, saved network")




#%%


ax = plot(history_loss, label='Training', x_label='epochs', title='MSE loss', figsize=(6,3))


#%%

u_pinn = pinn.apply(params, cloud.sorted_nodes)         ## For plotting
print("After training:")
cloud.visualize_field(u_pinn, cmap="jet", projection="2d", title="Untrained NN", figsize=(6,5));




#%%
