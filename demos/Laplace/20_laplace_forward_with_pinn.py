#%%

"""
Control of Laplace equation with PINNs (Preliminary step)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
import os
import matplotlib.pyplot as plt

# import functools
from updec.utils import plot, dataloader, make_dir, random_name
from updec.cloud import SquareCloud
import tracemalloc, time



#%%

# EXPERIMENET_ID = random_name()
EXPERIMENET_ID = "LaplaceForward"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)

## Save data for comparison
COMPFOLDER = "./data/" + "Comparison" +"/"
make_dir(COMPFOLDER)

KEY = jax.random.PRNGKey(41)     ## Use same random points for all iterations

Nx = 100
Ny = Nx

BATCH_SIZE = Nx*Ny // 10
INIT_LR = 1e-3
NB_LAYERS = 4
EPOCHS = 20000

W_in = 1.
W_bc = 1.


#%%

facet_types={"South":"d", "West":"d", "North":"d", "East":"d"}

train_cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=KEY)

test_cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=None)

fig, ax = plt.subplots(1, 2, figsize=(6*2,5))
train_cloud.visualize_cloud(s=0.1, title="Training cloud", ax=ax[0])
test_cloud.visualize_cloud(s=0.1, title="Testing cloud", ax=ax[1])


start = time.process_time()
tracemalloc.start()


x_in = train_cloud.sorted_nodes[:train_cloud.Ni, :]
dataloader_in = dataloader(x_in, BATCH_SIZE, KEY)       ## For training

x_test = test_cloud.sorted_nodes[:, :]
# dataloader_test = dataloader(x_test, BATCH_SIZE, KEY)

#%%

d_north = jax.vmap(lambda x: jnp.sin(jnp.pi * x[0]), in_axes=(0,), out_axes=0)
d_zero = jax.vmap(lambda x: 0., in_axes=(0,), out_axes=0)

boundary_conditions = {"South":d_zero, "West":d_zero, "North":d_north, "East":d_zero}

x_bc = []
u_bc = []

for f_id in train_cloud.facet_types.keys():
    bd_node_ids = jnp.array(train_cloud.facet_nodes[f_id])
    x_id = train_cloud.sorted_nodes[bd_node_ids]
    u_id = boundary_conditions[f_id](x_id)

    x_bc.append(x_id)
    u_bc.append(u_id)

x_bc = jnp.concatenate(x_bc, axis=0)
u_bc = jnp.concatenate(u_bc, axis=0)

fig, ax = plt.subplots(1, 4, figsize=(4*4,3))
for i, f_id in enumerate(train_cloud.facet_types.keys()):
    x = jnp.linspace(0,1,100)[:, jnp.newaxis]
    ax[i].plot(x, boundary_conditions[f_id](x), label=f_id)
    ax[i].set_ylim(0-1e-1,1+1e-1)
    ax[i].legend()

#%%

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        # inputs = jnp.concatenate([x, t], axis=-1)
        y = nn.Dense(2)(x)
        y = nn.Dense(50)(y)
        for _ in range(0, NB_LAYERS-1):
            y = nn.tanh(y)
            y = nn.Dense(50)(y)
        return nn.Dense(1)(y)[...,0]

def init_flax_params(net:nn.Module):
    # init_data = jnp.ones((1,2))
    init_data = jnp.ones((2,))
    params = net.init(KEY, init_data)
    print(net.tabulate(KEY, init_data, depth=8, console_kwargs={"force_jupyter":False}))
    return params

pinn = MLP()

print("PINN archtecture: ")
params = init_flax_params(pinn)


#%%

## Exact solution
def laplace_exact_sol(coord):
    # return jnp.sin(jnp.pi*coord[0])*jnp.cosh(jnp.pi*coord[1]) / jnp.cosh(jnp.pi)
    return jnp.sin(jnp.pi*coord[0])*jnp.sinh(jnp.pi*coord[1]) / jnp.sinh(jnp.pi)
exact_sol = jax.vmap(laplace_exact_sol)(x_test)

u_pinn = pinn.apply(params, x_test)

fig, ax = plt.subplots(1, 2, figsize=(6*2,5))

# print("Randomly initialized network state:")
test_cloud.visualize_field(u_pinn, cmap="jet", projection="2d", title="Untrained PINN solution", figsize=(6,5), ax=ax[0]);

test_cloud.visualize_field(exact_sol, cmap="jet", projection="2d", title="Exact solution", figsize=(6,5), ax=ax[1]);


#%%

total_steps = EPOCHS*(x_in.shape[0]//BATCH_SIZE)

## Optimizer
scheduler = optax.piecewise_constant_schedule(init_value=INIT_LR,
                                            boundaries_and_scales={int(total_steps*0.5):0.1, int(total_steps*0.75):0.1})

optimizer = optax.adam(learning_rate=scheduler)
# optimizer = optax.sgd(learning_rate=scheduler)

## Flax training state
state = train_state.TrainState.create(apply_fn=pinn.apply,
                                        params=params,
                                        tx=optimizer)


#%%

@jax.jit
def u(x, params):
    return pinn.apply(params, x)

@jax.jit
def gradu(x, params):
    ufunc = lambda x: pinn.apply(params, x)
    return jax.vmap(jax.jacfwd(ufunc))(x)

@jax.jit
def lapu(x, params):
    ufunc = lambda y: pinn.apply(params, y)
    lapfunc = jax.vmap(jax.jacfwd(jax.jacfwd(ufunc)))
    return jax.vmap(jnp.trace)(lapfunc(x))

def residual(x, params):
    """ Physics-based loss function with Burgers equation """
    return lapu(x, params)

def loss_fn_in(params, x_in):
    res = residual(x_in, params)
    loss_in = optax.l2_loss(res)                    ## Residual loss
    return jnp.mean(loss_in)

def loss_fn_bc(params, x_bc, u_bc):
    u_bc_pred = u(x_bc, params)
    loss_bc = optax.l2_loss(u_bc_pred - u_bc)        ## Data loss
    return jnp.mean(loss_bc)

def loss_fn(params, x_in, x_bc, u_bc):
    return W_in*loss_fn_in(params, x_in) + W_bc*loss_fn_bc(params,x_bc, u_bc)


#%%

@jax.jit
def train_step(state, x_in, x_bc, u_bc):
    loss_in = loss_fn_in(state.params, x_in)
    loss_bc = loss_fn_bc(state.params, x_bc, u_bc)

    grads = jax.grad(loss_fn)(state.params, x_in, x_bc, u_bc)
    state = state.apply_gradients(grads=grads)

    return state, loss_in, loss_bc

@jax.jit
def test_step(state, x_test, u_exact):
    u_pred = u(x_test, state.params)
    error_diff = optax.l2_loss(u_pred-u_exact)

    error_exact = optax.l2_loss(u_exact)

    return jnp.mean(error_diff) / jnp.mean(error_exact)


history_loss_in = []   # data for plotting
history_loss_bc = []   # data for plotting
history_loss_test = []
loader_keys = jax.random.split(key=KEY, num=EPOCHS)


#%%

if len(os.listdir(DATAFOLDER)) != 0:
    print("Found existing network, loading & training")
    state = checkpoints.restore_checkpoint(ckpt_dir=DATAFOLDER, prefix="pinn_checkpoint_", target=state)

else:
    print("Training from scratch")


for epoch in range(1,EPOCHS+1):

    dataloader_in = dataloader(x_in, BATCH_SIZE, loader_keys[epoch-1])

    loss_epch_in = 0.
    loss_epch_bc = 0.

    for i, batch in enumerate(dataloader_in):

        state, loss_in, loss_bc = train_step(state, batch, x_bc, u_bc)
        loss_epch_in += loss_in
        loss_epch_bc += loss_bc

    loss_test = test_step(state, x_test, exact_sol)

    history_loss_in.append(loss_epch_in)
    history_loss_bc.append(loss_epch_bc)
    history_loss_test.append(loss_test)

    if epoch<=3 or epoch%1000==0:
        print("Epoch: %-5d      ResidualLoss: %.6f     BoundaryLoss: %.6f   TestLoss: %.6f" % (epoch, loss_epch_in, loss_epch_bc, loss_test))

checkpoints.save_checkpoint(DATAFOLDER, prefix="pinn_checkpoint_", target=state, step=state.step, overwrite=True)
print("Training done, saved network")


mem_usage = tracemalloc.get_traced_memory()[1]
exec_time = time.process_time() - start

print("A few script details:")
print(" Peak memory usage: ", mem_usage, 'bytes')
print(' CPU execution time:', exec_time, 'seconds')

tracemalloc.stop()



#%%

fig, ax = plt.subplots(1, 2, figsize=(6*2,4))

plot(history_loss_in[:], label='Residual', x_label='epochs', title='MSE loss', y_scale="log", ax=ax[0])
plot(history_loss_bc[:], label='Boundary', x_label='epochs', title='MSE loss', y_scale="log", ax=ax[0]);

plot(history_loss_test[:], label='PINN Test Error', x_label='epochs', title='MSE loss', y_scale="log", ax=ax[1]);


#%%
fig, ax = plt.subplots(1, 2, figsize=(6*2,5))

u_pinn = pinn.apply(state.params, x_test)         ## For plotting
print("After training:")

test_cloud.visualize_field(u_pinn, cmap="jet", projection="2d", title="PINN forward solution", figsize=(6,5), ax=ax[0]);

# test_cloud.visualize_field(exact_sol, cmap="jet", projection="2d", title="Exact solution", figsize=(6,5), ax=ax[0]);

test_cloud.visualize_field(jnp.abs(exact_sol-u_pinn), cmap="magma", projection="2d", title="PINN absolute error", figsize=(6,5), ax=ax[1]);


# %%


jnp.savez(COMPFOLDER+"pinn_forward", in_loss_train=history_loss_in, bc_loss_train=history_loss_bc, total_loss_test=history_loss_test, exact_solution=exact_sol, optimal_solution_test=u_pinn, mem_time=jnp.array([mem_usage, exec_time]))


# %%
