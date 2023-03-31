#%%

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
import time


# EXPERIMENET_ID = random_name()
EXPERIMENET_ID = "01652"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)
KEY = jax.random.PRNGKey(42)     ## Use same random points for all iterations

Nx = 110
Ny = Nx
BATCH_SIZE = Nx*Ny // 10
EPOCHS = 50000



#%%

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        # inputs = jnp.concatenate([x, t], axis=-1)
        y = nn.Dense(2)(x)
        for _ in range(0, 4):
            y = nn.Dense(50)(y)
            y = nn.tanh(y)          ## Don't use ReLU !!!!
        return nn.Dense(1)(y)[...,0]

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
u_bc = jnp.concatenate(u_bc, axis=0)

# print(x_bc[:,1])
# print(u_bc[:,0])

# dataloader_bc = dataloader(jnp.concatenate([x_bc, u_bc], axis=-1), BATCH_SIZE, KEY)     ## TODO For training Do we need this ?

fig, ax = plt.subplots(1, 4, figsize=(4*4,3))
for i, f_id in enumerate(cloud.facet_types.keys()):
    x = jnp.linspace(0,1,100)[:, jnp.newaxis]
    ax[i].plot(x, boundary_conditions[f_id](x), label=f_id)
    ax[i].set_ylim(0-1e-1,1+1e-1)
    ax[i].legend()

# fig, ax = plt.subplots(2, 1, figsize=(4*4,3*2))
# ax[0].scatter(x_bc[:,0], u_bc[:,0], label="all boundaries along x")
# ax[1].scatter(x_bc[:,1], u_bc[:,0], label="all boundaries along y")
# plt.legend()


#%%


## Optimizer
scheduler = optax.linear_schedule(init_value=1e-2, end_value=1e-3, transition_steps=EPOCHS)
# scheduler = optax.exponential_decay(init_value=1e-1, 
#                                     end_value=1e-6, 
#                                     decay_rate=1e-1, transition_steps=EPOCHS, staircase=False)
# optimizer = optax.adabelief(learning_rate=scheduler)
optimizer = optax.sgd(learning_rate=scheduler)

## Flax training state
state = train_state.TrainState.create(apply_fn=pinn.apply,
                                        params=params,
                                        tx=optimizer)



#%%

@jax.jit
def u(x, params):
    # print("test u shape", pinn.apply(params, x).shape)
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

# def init_op_vmap(state):
#     # init_data = jnp.ones((1,2))
#     init_data = jnp.ones((3,2))
#     return lapu(init_data, state.params)
# print("init test shape", init_op_vmap(state).shape)


def residual(x, params):
    """ Physics-based loss function with Burgers equation """
    # val = u(x, params)
    # lap = jax.vmap(lapu, in_axes=(0, None), out_axes=0)(x, params)
    return lapu(x, params)

def loss_fn(params, x_in, x_bc, u_bc):
    u_bc_pred = u(x_bc, params)
    loss_bc = optax.l2_loss(u_bc_pred - u_bc)        ## Data loss

    res = residual(x_in, params)
    loss_in = optax.l2_loss(res)                    ## Residual loss

    w_in = 1.00
    w_bc = 1.00
    return w_in*jnp.mean(loss_in) + w_bc*jnp.mean(loss_bc)


#%%

@jax.jit
def train_step(state, x_in, x_bc, u_bc):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, x_in, x_bc, u_bc)
    state = state.apply_gradients(grads=grads)
    return state, loss


history_loss = []   # data for plotting
loader_keys = jax.random.split(key=KEY, num=EPOCHS)

if os.path.isfile(DATAFOLDER+"pinn_checkpoint_"+str(state.step)):
    print("Found existing network, loading & skipping training")
    state = checkpoints.restore_checkpoint(ckpt_dir=DATAFOLDER, prefix="pinn_checkpoint_", target=state)

else:
    print("Training from scratch")
    for epoch in range(1,EPOCHS+1):

        loss_epch = 0.
        dataloader_in = dataloader(x_in, BATCH_SIZE, loader_keys[epoch-1])

        for i, batch in enumerate(dataloader_in):

            state, loss = train_step(state, batch, x_bc, u_bc)
            loss_epch += loss

        history_loss.append(loss_epch)

        if epoch<=3 or epoch%100==0:
            print("Epoch: %-5d                Loss: %.6f" % (epoch, loss_epch))

checkpoints.save_checkpoint(DATAFOLDER, prefix="pinn_checkpoint_", target=state, step=state.step, overwrite=True)
print("Training done, saved network")


#%%


ax = plot(history_loss[1:], label='Training', x_label='epochs', title='MSE loss', y_scale="log", figsize=(6,4))

# ax.plot(jnp.log(jnp.array(history_loss)))
# ax.loglog(history_loss[1:])

#%%

u_pinn = pinn.apply(state.params, cloud.sorted_nodes)         ## For plotting
print("After training:")
cloud.visualize_field(u_pinn, cmap="jet", projection="2d", title="Trained PINN solution", figsize=(6,5));


#%%


## Exact solution
def laplace_exact_sol(coord):
    return jnp.sin(jnp.pi*coord[0])*jnp.cosh(jnp.pi*coord[1]) / jnp.cosh(jnp.pi)
laplace_exact_sol = jax.vmap(laplace_exact_sol, in_axes=(0,), out_axes=0)
exact_sol = laplace_exact_sol(cloud.sorted_nodes)

cloud.visualize_field(exact_sol, cmap="jet", projection="2d", title="Exact solution", figsize=(6,5));
