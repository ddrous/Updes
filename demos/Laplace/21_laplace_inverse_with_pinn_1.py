#%%
"""
Control of Laplace equation with PINNs (Step 1)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
import os
import matplotlib.pyplot as plt
from functools import partial

# import functools
from updec.utils import plot, dataloader, make_dir, random_name
from updec.cloud import SquareCloud
import tracemalloc, time



#%%

# EXPERIMENET_ID = random_name()
EXPERIMENET_ID = "LaplaceStep1"
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
EPOCHS = 20000

W_in = 1.
W_bc = 1.


#%%

facet_types={"North":"d", "South":"d", "West":"d", "East":"d"}

train_cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=None, support_size=None)     ## Train set is same as test set, regular !

# test_cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types, noise_key=None, support_size=None)
test_cloud = train_cloud

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

d_north = jax.vmap(lambda x: 0.)        ## Random initialisation of d_north
d_south = jax.vmap(lambda x: jnp.sin(2*jnp.pi * x[0]))
d_east = jax.vmap(lambda x: jnp.sinh(2*jnp.pi*x[1]) / (2*jnp.pi * jnp.cosh(2*jnp.pi)))
d_west = d_east

boundary_conditions = {"South":d_south, "West":d_west, "East":d_east, "North":d_north}

x_bc = []
u_bc = []
north_start = 0

for f_id in train_cloud.facet_types.keys():
    bd_node_ids = jnp.array(train_cloud.facet_nodes[f_id])
    x_id = train_cloud.sorted_nodes[bd_node_ids]
    u_id = boundary_conditions[f_id](x_id)

    x_bc.append(x_id)
    u_bc.append(u_id)

    if f_id == "North":
        north_ids = jnp.arange(north_start, north_start+u_id.shape[0], step=1)      ## Important for later training !
    else:
        north_start += u_id.shape[0]

x_bc = jnp.concatenate(x_bc, axis=0)
u_bc = jnp.concatenate(u_bc, axis=0)

## For the cost function
x_north = x_bc[north_ids, 0, jnp.newaxis]       ## New axis to maintain shape [BATCH, 1]
q_cost = jax.vmap(lambda x: jnp.cos(2*jnp.pi * x))(x_north)
print("North coordinates:", x_north.shape)

fig, ax = plt.subplots(1, 4, figsize=(4*4,3))
for i, f_id in enumerate(train_cloud.facet_types.keys()):
    x = jnp.linspace(0,1,100)[:, jnp.newaxis]
    ax[i].plot(x, boundary_conditions[f_id](x), label=f_id)
    ax[i].set_ylim(-1-1e-1,1+1e-1)
    ax[i].legend(loc="upper right")

#%%

class MLP(nn.Module):
    input_size: int
    nb_layers:int
    nb_neurons_per_layer:int

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.input_size)(x)
        y = nn.Dense(self.nb_neurons_per_layer)(y)
        for _ in range(0, self.nb_layers-1):
            y = nn.tanh(y)
            y = nn.Dense(self.nb_neurons_per_layer)(y)
        return nn.Dense(1)(y)[...,0]

def init_flax_params(net:nn.Module, input_size):
    init_data = jnp.ones((1,input_size))
    params = net.init(KEY, init_data)
    print(net.tabulate(KEY, init_data, depth=8, console_kwargs={"force_jupyter":False}))
    return params

u_pinn = MLP(input_size=2, nb_layers=4, nb_neurons_per_layer=50)
c_pinn = MLP(input_size=1, nb_layers=3, nb_neurons_per_layer=30)

print("Solution PINN archtecture: ")
u_params = init_flax_params(u_pinn, 2)

print("Control PINN archtecture: ")
c_params = init_flax_params(c_pinn, 1)


#%%

## Exact solution
def laplace_exact_sol(coord):
    PI = jnp.pi
    x, y = coord

    a = 0.5 * jnp.sin(2*PI*x) * (jnp.exp(2*PI*(y-1)) + jnp.exp(2*PI*(1-y))) / jnp.cosh(2*PI)
    b = jnp.cos(2*PI*x) * (jnp.exp(2*PI*y) + jnp.exp(-2*PI*y)) / (4*PI*jnp.cosh(2*PI))

    return a+b

exact_sol = jax.vmap(laplace_exact_sol)(x_test)

pinn_sol = u_pinn.apply(u_params, x_test)

fig, ax = plt.subplots(1, 2, figsize=(6*2,5))

# print("Randomly initialized network state:")
test_cloud.visualize_field(pinn_sol, cmap="jet", projection="2d", title="Untrained PINN solution", figsize=(6,5), ax=ax[0]);

test_cloud.visualize_field(exact_sol, cmap="jet", projection="2d", title="Exact solution", figsize=(6,5), ax=ax[1]);


#%%


def laplace_exact_control(x):
    PI = jnp.pi
    return (jnp.sin(2*PI*x)/jnp.cosh(2*PI)) + (jnp.cos(2*PI*x)*jnp.tanh(2*PI)/(2*PI))

pinn_control = c_pinn.apply(c_params, x_north)
exact_control = jax.vmap(laplace_exact_control)(x_north)

ax = plot(x_north, pinn_control, label="Untrained PINN control", x_label=r"$x$", figsize=(6,3));
ax = plot(x_north, exact_control, label="Exact control", x_label=r"$x$", ax=ax);


#%%

total_steps = EPOCHS*(x_in.shape[0]//BATCH_SIZE)

## Optimizer
u_scheduler = optax.piecewise_constant_schedule(init_value=INIT_LR,
                                            boundaries_and_scales={int(total_steps*0.5):0.1, int(total_steps*0.75):0.1})
c_scheduler = optax.piecewise_constant_schedule(init_value=INIT_LR,
                                            boundaries_and_scales={int(total_steps*0.5):0.1, int(total_steps*0.75):0.1})

u_optimizer = optax.adam(learning_rate=u_scheduler)
c_optimizer = optax.adam(learning_rate=c_scheduler)
# optimizer = optax.sgd(learning_rate=scheduler)

# ## Flax training state
# u_state = train_state.TrainState.create(apply_fn=u_pinn.apply,
#                                         params=u_params,
#                                         tx=u_optimizer)
# c_state = train_state.TrainState.create(apply_fn=c_pinn.apply,
#                                         params=c_params,
#                                         tx=c_optimizer)


#%%

@jax.jit
def u(x, params):
    return u_pinn.apply(params, x)

@jax.jit
def c(x, params):
    return c_pinn.apply(params, x)

@jax.jit
def gradu(x, params):
    ufunc = lambda y: u_pinn.apply(params, y)
    return jax.vmap(jax.jacfwd(ufunc))(x)

@jax.jit
def lapu(x, params):
    ufunc = lambda y: u_pinn.apply(params, y)
    lapfunc = jax.vmap(jax.jacfwd(jax.jacfwd(ufunc)))
    return jax.vmap(jnp.trace)(lapfunc(x))

# print(gradu(jnp.ones((10,2)), u_params))


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

def loss_fn_ct(params, north_ids, x_bc, q_cost):
    xy_north = x_bc[north_ids, :]
    grad_u_north_y = gradu(xy_north, params)[..., 1]
    loss_cost = (grad_u_north_y - q_cost[...,0])**2
    return jnp.trapz(loss_cost, x=xy_north[:, 0])

def set_north_bc(c_params, u_bc, north_ids, x_bc):
    x_north = x_bc[north_ids, 0, jnp.newaxis]
    u_north = c(x_north, c_params)
    return u_bc.at[north_ids].set(u_north)

def loss_fn(u_params, c_params, x_in, x_bc, u_bc, north_ids, q_cost, W_ct):

    new_u_bc = set_north_bc(c_params, u_bc, north_ids, x_bc)

    return W_in*loss_fn_in(u_params, x_in) \
            + W_bc*loss_fn_bc(u_params, x_bc, new_u_bc) \
            + W_ct*loss_fn_ct(u_params, north_ids, x_bc, q_cost)


#%%

@partial(jax.jit, static_argnames="W_ct")
def train_step(u_state, c_state, x_in, x_bc, u_bc, north_ids, q_cost, W_ct):
    loss_in = loss_fn_in(u_state.params, x_in)
    loss_bc = loss_fn_bc(u_state.params, x_bc, u_bc)
    loss_ct = loss_fn_ct(u_state.params, north_ids, x_bc, q_cost)

    u_grads = jax.grad(loss_fn, argnums=0)(u_state.params, c_state.params, x_in, x_bc, u_bc, north_ids, q_cost, W_ct)
    u_state = u_state.apply_gradients(grads=u_grads)

    c_grads = jax.grad(loss_fn, argnums=1)(u_state.params, c_state.params, x_in, x_bc, u_bc, north_ids, q_cost, W_ct)
    c_state = c_state.apply_gradients(grads=c_grads)

    return u_state, c_state, loss_in, loss_bc, loss_ct

# @jax.jit
# def test_step(state, x_test, u_exact):
#     u_pred = u(x_test, state.params)
#     error_diff = optax.l2_loss(u_pred-u_exact)

#     error_exact = optax.l2_loss(u_exact)

#     return jnp.mean(error_diff) / jnp.mean(error_exact)


# history_loss_in = []
# history_loss_bc = []
# history_loss_ct = []
# # history_loss_test = []


#%%

cost_weights = []
min_costs_per_weight = []
loader_keys = jax.random.split(key=KEY, num=EPOCHS)

### Step 1 Line search strategy
for W_id, exp in enumerate(range(-3, 8)):

    W_ct = 10**(exp)

    ## Flax training state
    u_state = train_state.TrainState.create(apply_fn=u_pinn.apply,
                                            params=u_params,
                                            tx=u_optimizer)
    c_state = train_state.TrainState.create(apply_fn=c_pinn.apply,
                                            params=c_params,
                                            tx=c_optimizer)

    history_loss_in = []
    history_loss_bc = []
    history_loss_ct = []

    if len(os.listdir(DATAFOLDER)) != 0:
        print("Found existing networks, loading & training")
        u_state = checkpoints.restore_checkpoint(ckpt_dir=DATAFOLDER, prefix="u_pinn_checkpoint_"+str(W_ct)+"_", target=u_state)
        c_state = checkpoints.restore_checkpoint(ckpt_dir=DATAFOLDER, prefix="c_pinn_checkpoint_"+str(W_ct)+"_", target=c_state)

    else:
        print("Training from scratch")


    for epoch in range(1,EPOCHS+1):

        dataloader_in = dataloader(x_in, BATCH_SIZE, loader_keys[epoch-1])

        loss_epch_in = 0.
        loss_epch_bc = 0.
        loss_epch_ct = 0.

        for i, batch in enumerate(dataloader_in):

            u_state, c_state, loss_in, loss_bc, loss_ct = train_step(u_state, c_state, batch, x_bc, u_bc, north_ids, q_cost, W_ct)
            loss_epch_in += loss_in
            loss_epch_bc += loss_bc
            loss_epch_ct += loss_ct

        # loss_test = test_step(state, x_test, exact_sol)

        history_loss_in.append(loss_epch_in)
        history_loss_bc.append(loss_epch_bc)
        history_loss_ct.append(loss_epch_ct)
        # history_loss_test.append(loss_test)

        if epoch<=3 or epoch%1000==0:
            print("Epoch: %-5d      ResidualLoss: %.6f     BoundaryLoss: %.6f   CostLoss: %.6f" % (epoch, loss_epch_in, loss_epch_bc, loss_epch_ct))

    checkpoints.save_checkpoint(DATAFOLDER, prefix="u_pinn_checkpoint_"+str(W_ct)+"_", target=u_state, step=u_state.step, overwrite=True)
    checkpoints.save_checkpoint(DATAFOLDER, prefix="c_pinn_checkpoint_"+str(W_ct)+"_", target=c_state, step=c_state.step, overwrite=True)
    print("Training done, saved networks")

    cost_weights.append(W_ct)
    min_costs_per_weight.append((jnp.array(history_loss_ct)).min())


## %%

    fig, ax = plt.subplots(1, 2, figsize=(6*2,4))

    plot(history_loss_in[:], label='Residual', x_label='epochs', y_scale="log", ax=ax[0])
    plot(history_loss_bc[:], label='Boundary', x_label='epochs', title="Cost Weight: "+str(W_ct), y_scale="log", ax=ax[0]);
    # plot(history_loss_ct[:], label='Cost', x_label='epochs', y_scale="log", ax=ax[0]);

    # plot(history_loss_test[:], label='PINN Test Error', x_label='epochs', y_scale="log", ax=ax[1]);
    plot(history_loss_ct[:], label='Cost', x_label='epochs', title="Cost Weight: "+str(W_ct), y_scale="log", ax=ax[1]);
    # plt.show()


##%%
    fig, ax = plt.subplots(1, 2, figsize=(6*2,5))

    pinn_sol = u_pinn.apply(u_state.params, x_test)         ## For plotting
    print("After training:")

    plt.suptitle("Loss Wieght: "+str(W_ct))
    test_cloud.visualize_field(pinn_sol, cmap="jet", projection="2d", title="PINN forward solution", figsize=(6,5), ax=ax[0]);

    # test_cloud.visualize_field(exact_sol, cmap="jet", projection="2d", title="Exact solution", figsize=(6,5), ax=ax[0]);

    test_cloud.visualize_field(jnp.abs(exact_sol-pinn_sol), cmap="magma", projection="2d", title="PINN absolute error", figsize=(6,5), ax=ax[1]);
    # plt.show()

## %%

    xy_north = x_bc[north_ids, :]
    pinn_control = c_pinn.apply(c_state.params, x_north)
    pinn_sol_control = u_pinn.apply(u_state.params, xy_north)

    ax = plot(x_north, exact_control, label="Analytical control", x_label=r"$x$", figsize=(6,3));
    ax = plot(x_north, pinn_control, label="PINN control", title="Cost weight: "+str(W_ct), x_label=r"$x$", ax=ax);
    ax = plot(x_north, pinn_sol_control, "--", label="PINN solution at North", ax=ax);


## %%

    mem_usage = tracemalloc.get_traced_memory()[1]
    exec_time = time.process_time() - start

    print("A few script details:")
    print(" Peak memory usage: ", mem_usage, 'bytes')
    print(' CPU execution time:', exec_time, 'seconds')

    jnp.savez(COMPFOLDER+"pinn_inv_1_"+str(W_id), objective_cost=history_loss_ct, in_loss_train=history_loss_in, bc_loss_train=history_loss_bc, exact_control=exact_control, optimal_bcn_c=pinn_control, optimal_bcn_u=pinn_sol_control, exact_solution=exact_sol, optimal_solution=pinn_sol, mem_time_cum=jnp.array([mem_usage, exec_time]))

    plt.show()

tracemalloc.stop()


# %%

plot(cost_weights, min_costs_per_weight, ".-", title='Minimal costs vs. weights', x_label='cost weights', y_label='cost', x_scale="log", y_scale="log", figsize=(6,3));

# %%
