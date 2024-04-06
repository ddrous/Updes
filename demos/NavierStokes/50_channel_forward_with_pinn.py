#%%

"""
Control of Navier-Stokes equation with PINNs (Preliminary step)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
import os
import matplotlib.pyplot as plt

# import functools
from updes.utils import plot, dataloader, make_dir
from updes.cloud import GmshCloud
from updes.visualise import pyvista_animation

import tracemalloc, time



#%%

# EXPERIMENET_ID = random_name()
EXPERIMENET_ID = "ChannelForward"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)

## Save data for comparison
COMPFOLDER = "./data/" + "Comparison" +"/"
make_dir(COMPFOLDER)


KEY = jax.random.PRNGKey(41)     ## Use same random points for all iterations


W_mo = 1.
W_co = 1.
# W_bc = 100.       ## See the paper !!
W_bc = 1.
 
Re = 100.
Pa = 0.


## Normalising the x coordinates
NORM_FACTOR = jnp.array([[1.5, 1.0]])


INIT_LR = 1e-3
EPOCHS = 50000


#%%

facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n", "Blowing":"d", "Suction":"d"}
facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Blowing":"n", "Suction":"n"}

cloud_vel = GmshCloud(filename="./meshes/channel_blowing_suction.py", facet_types=facet_types_vel, mesh_save_location=DATAFOLDER, support_size=1)
cloud_p = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi, support_size=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,3*2), sharex=True)
cloud_vel.visualize_cloud(ax=ax1, s=1, title="Cloud for velocity", xlabel=False);
cloud_p.visualize_cloud(ax=ax2, s=1, title=r"Cloud for pressure");


x_in_vel = cloud_vel.sorted_nodes[:cloud_vel.Ni, :] / NORM_FACTOR
x_in_p = cloud_p.sorted_nodes[:cloud_p.Ni, :] / NORM_FACTOR


#%%

start = time.process_time()
tracemalloc.start()

parabolic = jax.vmap(lambda x: 4*x[1]*(1.-x[1]))
blowing = jax.vmap(lambda x: 0.3)
suction = jax.vmap(lambda x: 0.3)
zero = jax.vmap(lambda x: 0.)
atmospheric = jax.vmap(lambda x: Pa)

bc_u = {"Wall":zero, "Inflow":parabolic, "Outflow":zero, "Blowing":zero, "Suction":zero}
bc_v = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Blowing":blowing, "Suction":suction}
bc_p = {"Wall":zero, "Inflow":zero, "Outflow":atmospheric, "Blowing":zero, "Suction":zero}

## Dirichlet boundaries
x_bc_vel = []
x_bc_p = []
u_bc = []
v_bc = []
p_bc = []

## Neumann boundaries (for veleocity) only
x_bc_n_vel = []
x_bc_n_p = []

for f_id in cloud_vel.facet_types.keys():

    if cloud_vel.facet_types[f_id]=="d":
        bd_node_ids = jnp.array(cloud_vel.facet_nodes[f_id])
        x_id_vel = cloud_vel.sorted_nodes[bd_node_ids]
        u_id = bc_u[f_id](x_id_vel)
        v_id = bc_v[f_id](x_id_vel)

        x_bc_vel.append(x_id_vel)
        u_bc.append(u_id)
        v_bc.append(v_id)

    if cloud_p.facet_types[f_id]=="d":
        bd_node_ids = jnp.array(cloud_p.facet_nodes[f_id])
        x_id_p = cloud_p.sorted_nodes[bd_node_ids]
        p_id = bc_p[f_id](x_id_p)
        x_bc_p.append(x_id_p)
        p_bc.append(p_id)

    if cloud_vel.facet_types[f_id]=="n":
        bd_node_ids = jnp.array(cloud_vel.facet_nodes[f_id])
        x_id_vel = cloud_vel.sorted_nodes[bd_node_ids]
        x_bc_n_vel.append(x_id_vel)

    if cloud_p.facet_types[f_id]=="n":
        bd_node_ids = jnp.array(cloud_p.facet_nodes[f_id])
        x_id_vel = cloud_p.sorted_nodes[bd_node_ids]
        x_bc_n_p.append(x_id_vel)


x_bc_vel = jnp.concatenate(x_bc_vel, axis=0) / NORM_FACTOR
u_bc = jnp.concatenate(u_bc, axis=0)
v_bc = jnp.concatenate(v_bc, axis=0)

x_bc_n_vel = jnp.concatenate(x_bc_n_vel, axis=0) / NORM_FACTOR

x_bc_p = jnp.concatenate(x_bc_p, axis=0)    / NORM_FACTOR
p_bc = jnp.concatenate(p_bc, axis=0)

x_bc_n_p = jnp.concatenate(x_bc_n_p, axis=0) / NORM_FACTOR


fig, ax = plt.subplots(1, 5, figsize=(4*5,3))
fig.suptitle("Boundary conditions for velocity u (ROUGHLY SPEAKING)")
for i, f_id in enumerate(cloud_vel.facet_types.keys()):
    x = jnp.linspace(0,1.5,100)[:, jnp.newaxis]
    ax[i].plot(x, bc_u[f_id](x), label=f_id)
    ax[i].set_ylim(0-1e-1,1+1e-1)
    ax[i].legend()
fig, ax = plt.subplots(1, 5, figsize=(4*5,3))
fig.suptitle("Boundary conditions for velocity v (ROUGHLY SPEAKING)")
for i, f_id in enumerate(cloud_vel.facet_types.keys()):
    x = jnp.linspace(0,1,100)[:, jnp.newaxis]
    ax[i].plot(x, bc_v[f_id](x), label=f_id)
    ax[i].set_ylim(0-1e-1,1+1e-1)
    ax[i].legend()


#%%

class MLP(nn.Module):
    input_size: int=2
    nb_layers:int=5
    nb_neurons_per_layer:int=50

    @nn.compact
    def __call__(self, x):                              ## TODO no normalisation
        y = nn.Dense(self.input_size)(x)
        y = nn.Dense(self.nb_neurons_per_layer)(y)
        for _ in range(0, self.nb_layers-1):
            y = nn.tanh(y)
            y = nn.Dense(self.nb_neurons_per_layer)(y)
        return nn.Dense(3)(y)

def init_flax_params(net:nn.Module):
    init_data = jnp.ones((2,))
    params = net.init(KEY, init_data)
    print(net.tabulate(KEY, init_data, depth=8, console_kwargs={"force_jupyter":False}))
    return params

pinn = MLP()

print("PINN archtecture: ")
params = init_flax_params(pinn)


#%%

# total_steps = EPOCHS*(x_in_vel.shape[0]//BATCH_SIZE)
total_steps = EPOCHS     ## No batch size

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

def u(x, params):
    return pinn.apply(params, x)[...,0]
def v(x, params):
    return pinn.apply(params, x)[...,1]
def p(x, params):
    return pinn.apply(params, x)[...,2]


def gradu(x, params):
    ufunc = lambda x: pinn.apply(params, x)[...,0]
    return jax.vmap(jax.jacfwd(ufunc))(x)
def gradv(x, params):
    vfunc = lambda x: pinn.apply(params, x)[...,1]
    return jax.vmap(jax.jacfwd(vfunc))(x)
def gradp(x, params):
    pfunc = lambda x: pinn.apply(params, x)[...,2]
    return jax.vmap(jax.jacfwd(pfunc))(x)

def lapu(x, params):
    ufunc = lambda y: pinn.apply(params, y)[...,0]
    lapfunc = jax.vmap(jax.jacfwd(jax.jacfwd(ufunc)))
    return jax.vmap(jnp.trace)(lapfunc(x))
def lapv(x, params):
    vfunc = lambda y: pinn.apply(params, y)[...,1]
    lapfunc = jax.vmap(jax.jacfwd(jax.jacfwd(vfunc)))
    return jax.vmap(jnp.trace)(lapfunc(x))

def residual(x_vel, x_p, params):
    """ Physics-based loss function with Burgers equation """
    vals = jnp.stack([u(x_vel,params), v(x_vel,params)], axis=-1)
    grads = jnp.stack([gradu(x_vel, params), gradv(x_vel, params)], axis=-1)
    laps = jnp.stack([lapu(x_vel, params), lapv(x_vel, params)], axis=-1)
    dot_vec = jax.vmap(jnp.dot, in_axes=(0,0), out_axes=0)

    momentum = dot_vec(vals, grads) + gradp(x_p,params) - (laps/Re)
    continuity = grads[:, 0, 0] + grads[:, 1, 1]

    return jnp.linalg.norm(momentum, axis=-1), continuity

def loss_fn_in(params, x_vel, x_p):
    mom, cont = residual(x_vel, x_p, params)
    return jnp.mean(optax.l2_loss(mom)), jnp.mean(optax.l2_loss(cont))

def loss_fn_bc(params, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc):
    u_bc_pred = u(x_bc_vel, params)
    v_bc_pred = v(x_bc_vel, params)
    p_bc_pred = p(x_bc_p, params)

    loss_bc_u = jnp.mean(optax.l2_loss(u_bc_pred - u_bc))
    loss_bc_v = jnp.mean(optax.l2_loss(v_bc_pred - v_bc))
    loss_bc_p = jnp.mean(optax.l2_loss(p_bc_pred - p_bc))

    return loss_bc_u + loss_bc_v + loss_bc_p

def loss_fn_bc_n(params, x_bc_n_vel):
    u_bc_pred = gradu(x_bc_n_vel, params)[:, 0]
    v_bc_pred = gradv(x_bc_n_vel, params)[:, 0]

    loss_bc_u = jnp.mean(optax.l2_loss(u_bc_pred))
    loss_bc_v = jnp.mean(optax.l2_loss(v_bc_pred))

    return loss_bc_u + loss_bc_v


def loss_fn(params, x_in_vel, x_in_p, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc, x_bc_n_vel):
    mon, cont = loss_fn_in(params, x_in_vel, x_in_p)
    bc = loss_fn_bc(params, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc)

    bc_n = loss_fn_bc_n(params, x_bc_n_vel)

    return W_mo*mon + W_co*cont + W_bc*bc   +     W_bc*bc_n


#%%

@jax.jit
def train_step(state, x_in_vel, x_in_p, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc, x_bc_n_vel):
    loss_mon, loss_cont = loss_fn_in(state.params, x_in_vel, x_in_p)
    loss_bc = loss_fn_bc(state.params, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc)
    loss_bc_n = loss_fn_bc_n(state.params, x_bc_n_vel)

    grads = jax.grad(loss_fn)(state.params, x_in_vel, x_in_p, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc, x_bc_n_vel)
    state = state.apply_gradients(grads=grads)

    return state, loss_mon, loss_cont, loss_bc+loss_bc_n


history_loss_mon = []
history_loss_cont = []
history_loss_bc = []


#%%

if len(os.listdir(DATAFOLDER)) > 2:
    print("Found existing network, loading & training")
    state = checkpoints.restore_checkpoint(ckpt_dir=DATAFOLDER, prefix="pinn_checkpoint_", target=state)

else:
    print("Training from scratch")


for epoch in range(1,EPOCHS+1):

    state, loss_mon, loss_cont, loss_bc = train_step(state, x_in_vel, x_in_p, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc, x_bc_n_vel)

    history_loss_mon.append(loss_mon)
    history_loss_cont.append(loss_cont)
    history_loss_bc.append(loss_bc)

    if epoch<=3 or epoch%5000==0:
        print("Epoch: %-5d      MomentumLoss: %.6f     ContinuityLoss: %.6f   BoundaryLoss: %.6f" % (epoch, loss_mon, loss_cont, loss_bc))

checkpoints.save_checkpoint(DATAFOLDER, prefix="pinn_checkpoint_", target=state, step=state.step, overwrite=True)
print("Training done, saved network")


mem_usage = tracemalloc.get_traced_memory()[1]
exec_time = time.process_time() - start

print("A few performance details:")
print(" Peak memory usage: ", mem_usage, 'bytes')
print(' CPU execution time:', exec_time, 'seconds')

tracemalloc.stop()


#%%

fig, ax = plt.subplots(1, 1, figsize=(6*1,4))

plot(history_loss_mon[:], label='Momentum', x_label='epochs', y_scale="log", ax=ax)
plot(history_loss_cont[:], label='Continuity', x_label='epochs', y_scale="log", ax=ax);
plot(history_loss_bc[:], label='Boundary', x_label='epochs', title='Training loss', y_scale="log", ax=ax);


#%%
fig, ax = plt.subplots(1, 2, figsize=(6*2,3.6))

vals_pinn = pinn.apply(state.params, cloud_vel.sorted_nodes / NORM_FACTOR)
u_pinn, v_pinn = vals_pinn[:, 0], vals_pinn[:, 1]
p_pinn = p(cloud_p.sorted_nodes / NORM_FACTOR, state.params)

vel_pinn = jnp.linalg.norm(vals_pinn[:, :2], axis=-1)
print("After training:")

cloud_vel.visualize_field(vel_pinn, cmap="jet", projection="2d", title="PINN forward velocity", ax=ax[0]);

cloud_p.visualize_field(p_pinn, cmap="jet", projection="2d", title="PINN forward pressure", ax=ax[1]);


# %%

renum_map_vel = jnp.array(list(cloud_vel.renumbering_map.keys()))
renum_map_p = jnp.array(list(cloud_p.renumbering_map.keys()))

jnp.savez(DATAFOLDER+'u.npz', renum_map_vel, jnp.stack([u_pinn]*5, axis=0))
jnp.savez(DATAFOLDER+'v.npz', renum_map_vel, jnp.stack([v_pinn]*5, axis=0))
jnp.savez(DATAFOLDER+'vel.npz', renum_map_vel, jnp.stack([vel_pinn]*5, axis=0))
jnp.savez(DATAFOLDER+'p.npz', renum_map_p, jnp.stack([p_pinn]*5, axis=0))

pyvista_animation(DATAFOLDER, "u", duration=5, vmin=jnp.min(u_pinn), vmax=jnp.max(u_pinn))
pyvista_animation(DATAFOLDER, "v", duration=5, vmin=jnp.min(v_pinn), vmax=jnp.max(v_pinn))
pyvista_animation(DATAFOLDER, "vel", duration=5, vmin=jnp.min(vel_pinn), vmax=jnp.max(vel_pinn))
pyvista_animation(DATAFOLDER, "p", duration=5, vmin=jnp.min(p_pinn), vmax=jnp.max(p_pinn))

# %%

jnp.savez(COMPFOLDER+"pinn_forward", mom_loss=history_loss_mon, cont_loss=history_loss_cont, bc_loss=history_loss_bc, mem_time=jnp.array([mem_usage, exec_time]))
