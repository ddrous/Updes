#%%
"""
Control of Navier-Stokes equation with PINNs (Step 3) [Finetuning for Weight Cost == 1]
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
from updes.utils import plot, dataloader, make_dir
from updes.cloud import GmshCloud
from updes.visualise import pyvista_animation

import tracemalloc, time


#%%

# EXPERIMENET_ID = random_name()
EXPERIMENET_ID = "ChannelInverseStep3"
DATAFOLDER = "./data/" + EXPERIMENET_ID +"/"
make_dir(DATAFOLDER)

## Save data for comparison
COMPFOLDER = "./data/" + "Comparison" +"/"
make_dir(COMPFOLDER)

KEY = jax.random.PRNGKey(41)     ## Use same random points for all iterations

W_mo = 1.
W_co = 1.
W_bc = 1.

Re = 100
Pa = 0.

NORM_FACTOR = jnp.array([[1.5, 1.0]])

INIT_LR = 1e-3
EPOCHS = 100000

#%%

facet_types_vel = {"Wall":"d", "Inflow":"d", "Outflow":"n", "Blowing":"d", "Suction":"d"}
facet_types_phi = {"Wall":"n", "Inflow":"n", "Outflow":"d", "Blowing":"n", "Suction":"n"}

cloud_vel = GmshCloud(filename="./meshes/channel_blowing_suction.py", facet_types=facet_types_vel, mesh_save_location=DATAFOLDER, support_size=1)
cloud_p = GmshCloud(filename=DATAFOLDER+"mesh.msh", facet_types=facet_types_phi, support_size=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,3*2), sharex=True)
cloud_vel.visualize_cloud(ax=ax1, s=1, title="Cloud for velocity", xlabel=False);
cloud_p.visualize_cloud(ax=ax2, s=1, title=r"Cloud for pressure");

## Normalising the x coordinates

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

bc_u = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Blowing":zero, "Suction":zero}      ## Random initialisation of Inflow vel
bc_v = {"Wall":zero, "Inflow":zero, "Outflow":zero, "Blowing":blowing, "Suction":suction}
bc_p = {"Wall":zero, "Inflow":zero, "Outflow":atmospheric, "Blowing":zero, "Suction":zero}

x_bc_vel = []
x_bc_p = []
u_bc = []
v_bc = []
p_bc = []

x_bc_n_vel = []

inlet_start = 0     ## Helps identify inlet ids amond boundary conditions

for f_id in cloud_vel.facet_types.keys():

    if cloud_vel.facet_types[f_id]=="d":
        bd_node_ids = jnp.array(cloud_vel.facet_nodes[f_id])
        x_id_vel = cloud_vel.sorted_nodes[bd_node_ids]
        u_id = bc_u[f_id](x_id_vel)
        v_id = bc_v[f_id](x_id_vel)

        x_bc_vel.append(x_id_vel)
        u_bc.append(u_id)
        v_bc.append(v_id)

        if f_id == "Inflow":
            inlet_ids = jnp.arange(inlet_start, inlet_start+u_id.shape[0], step=1)
        else:
            inlet_start += u_id.shape[0]

    if cloud_vel.facet_types[f_id]=="n":
        bd_node_ids = jnp.array(cloud_vel.facet_nodes[f_id])
        x_id_vel = cloud_vel.sorted_nodes[bd_node_ids]
        x_bc_n_vel.append(x_id_vel)

    if cloud_p.facet_types[f_id]=="d":
        bd_node_ids = jnp.array(cloud_p.facet_nodes[f_id])
        x_id_p = cloud_p.sorted_nodes[bd_node_ids]
        p_id = bc_p[f_id](x_id_p)
        x_bc_p.append(x_id_p)
        p_bc.append(p_id)

x_bc_vel = jnp.concatenate(x_bc_vel, axis=0) / NORM_FACTOR
u_bc = jnp.concatenate(u_bc, axis=0)
v_bc = jnp.concatenate(v_bc, axis=0)

x_bc_n_vel = jnp.concatenate(x_bc_n_vel, axis=0) / NORM_FACTOR

x_bc_p = jnp.concatenate(x_bc_p, axis=0)    / NORM_FACTOR
p_bc = jnp.concatenate(p_bc, axis=0)

xy_inlet = x_bc_vel[inlet_ids, :] 
y_inlet = x_bc_vel[inlet_ids, 1, jnp.newaxis] 

outlet_ids = jnp.array(cloud_vel.facet_nodes["Outflow"])
xy_outlet = cloud_vel.sorted_nodes[outlet_ids, :] / NORM_FACTOR
y_outlet = xy_outlet[:, 1, jnp.newaxis]

u_parab = parabolic(xy_outlet)


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
    output_size: int=1
    nb_layers:int=5
    nb_neurons_per_layer:int=50

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.input_size)(x)
        y = nn.Dense(self.nb_neurons_per_layer)(y)
        for _ in range(0, self.nb_layers-1):
            y = nn.tanh(y)
            y = nn.Dense(self.nb_neurons_per_layer)(y)
        if self.output_size == 1:
            return nn.Dense(1)(y)[...,0]
        else:
            return nn.Dense(self.output_size)(y)

def init_flax_params(net:nn.Module, input_size, print_network=True):
    init_data = jnp.ones((1,input_size))
    params = net.init(KEY, init_data)
    if print_network == True:
        print(net.tabulate(KEY, init_data, depth=8, console_kwargs={"force_jupyter":False}))
    return params

u_pinn = MLP(input_size=2, nb_layers=5, nb_neurons_per_layer=50, output_size=3)
c_pinn = MLP(input_size=1, nb_layers=3, nb_neurons_per_layer=30, output_size=1)

print("Solution PINN archtecture: ")
u_params = init_flax_params(u_pinn, 2)

print("Control PINN archtecture: ")
c_params = init_flax_params(c_pinn, 1)


#%%

total_steps = EPOCHS

## Optimizer
u_scheduler = optax.piecewise_constant_schedule(init_value=INIT_LR,
                                            boundaries_and_scales={int(total_steps*0.5):0.1, int(total_steps*0.75):0.1})
c_scheduler = optax.piecewise_constant_schedule(init_value=INIT_LR,
                                            boundaries_and_scales={int(total_steps*0.5):0.1, int(total_steps*0.75):0.1})

u_optimizer = optax.adam(learning_rate=u_scheduler)
c_optimizer = optax.adam(learning_rate=c_scheduler)


controls_folder = DATAFOLDER[:-2]+"1/"
controls_folder


W_ct_list = []
control_states = []

for dir in os.listdir(controls_folder):
    if dir[0] == "c":

        W_ct = float(dir.rsplit("_", 2)[-2])

        if W_ct == 1:
            prefix = dir.rsplit("_", 1)[0]
            c_pinn = MLP(input_size=1, nb_layers=3, nb_neurons_per_layer=30, output_size=1)
            c_params = init_flax_params(c_pinn, 1, False)
            c_state = train_state.TrainState.create(apply_fn=c_pinn.apply,
                                            params=c_params,
                                            tx=c_optimizer)
            c_state = checkpoints.restore_checkpoint(ckpt_dir=controls_folder, 
                                                        prefix=prefix+"_", 
                                                        target=c_state)

            W_ct_list.append(W_ct)
            control_states.append(c_state)



#%%

def c(x, params):
    return c_pinn.apply(params, x)

def u(x, params):
    return u_pinn.apply(params, x)[...,0]
def v(x, params):
    return u_pinn.apply(params, x)[...,1]
def p(x, params):
    return u_pinn.apply(params, x)[...,2]


def gradu(x, params):
    ufunc = lambda x: u_pinn.apply(params, x)[...,0]
    return jax.vmap(jax.jacfwd(ufunc))(x)
def gradv(x, params):
    vfunc = lambda x: u_pinn.apply(params, x)[...,1]
    return jax.vmap(jax.jacfwd(vfunc))(x)
def gradp(x, params):
    pfunc = lambda x: u_pinn.apply(params, x)[...,2]
    return jax.vmap(jax.jacfwd(pfunc))(x)

def lapu(x, params):
    ufunc = lambda y: u_pinn.apply(params, y)[...,0]
    lapfunc = jax.vmap(jax.jacfwd(jax.jacfwd(ufunc)))
    return jax.vmap(jnp.trace)(lapfunc(x))
def lapv(x, params):
    vfunc = lambda y: u_pinn.apply(params, y)[...,1]
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


def loss_fn_ct(params):                 ## TODO make this function pure
    u_out = u(xy_outlet, params)
    v_out = v(xy_outlet, params)

    integrand = (u_out-u_parab)**2 + v_out**2
    return 0.5 * jnp.trapz(integrand, x=y_outlet[...,0])

def set_inlet_bc(c_params, u_bc):       ## TODO make this pure
    u_north = c(y_inlet, c_params)
    return u_bc.at[inlet_ids].set(u_north)

def loss_fn_bc_n(params, x_bc_n_vel):
    u_bc_pred = gradu(x_bc_n_vel, params)[:, 0]
    v_bc_pred = gradv(x_bc_n_vel, params)[:, 0]

    loss_bc_u = jnp.mean(optax.l2_loss(u_bc_pred))
    loss_bc_v = jnp.mean(optax.l2_loss(v_bc_pred))

    return loss_bc_u + loss_bc_v

def loss_fn(u_params, c_params, x_in_vel, x_in_p, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc, x_bc_n_vel):

    new_u_bc = set_inlet_bc(c_params, u_bc)

    mon, cont = loss_fn_in(u_params, x_in_vel, x_in_p)
    bc = loss_fn_bc(u_params, x_bc_vel, new_u_bc, v_bc, x_bc_p, p_bc)

    bc_n = loss_fn_bc_n(u_params, x_bc_n_vel)

    return W_mo*mon + W_co*cont + W_bc*(bc+bc_n)
    # return W_mo*mon + W_co*cont + W_bc*(bc)


#%%

@jax.jit
def train_step(u_state, c_state, x_in_vel, x_in_p, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc, x_bc_n_vel):
    loss_mon, loss_cont = loss_fn_in(u_state.params, x_in_vel, x_in_p)
    loss_bc = loss_fn_bc(u_state.params, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc)
    loss_bc_n = loss_fn_bc_n(u_state.params, x_bc_n_vel)

    loss_ct = loss_fn_ct(u_state.params)

    u_grads = jax.grad(loss_fn, argnums=0)(u_state.params, c_state.params, x_in_vel, x_in_p, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc, x_bc_n_vel)
    u_state = u_state.apply_gradients(grads=u_grads)

    return u_state, c_state, loss_mon, loss_cont, loss_bc+loss_bc_n, loss_ct
    # return u_state, c_state, loss_mon, loss_cont, loss_bc, loss_ct


#%%

### Read all the saved 


#%%

costs_vs_weight = []
loader_keys = jax.random.split(key=KEY, num=EPOCHS)

### Step 2 Line search strategy

for W_id, W_ct in enumerate(W_ct_list):

    ## Flax training state
    u_state = train_state.TrainState.create(apply_fn=u_pinn.apply,
                                            params=u_params,
                                            tx=u_optimizer)
    c_state = control_states[W_id]

    history_loss_mon = []
    history_loss_cont = []
    history_loss_bc = []
    history_loss_ct = []

    if len(os.listdir(DATAFOLDER)) > 2:
        print("Found existing networks, loading & training")
        u_state = checkpoints.restore_checkpoint(ckpt_dir=DATAFOLDER, prefix="u_pinn_checkpoint_"+str(W_ct)+"_", target=u_state)

    else:
        print("Training from scratch")

    for epoch in range(1,EPOCHS+1):

        u_state, c_state, loss_mon, loss_cont, loss_bc, loss_ct = train_step(u_state, c_state, x_in_vel, x_in_p, x_bc_vel, u_bc, v_bc, x_bc_p, p_bc, x_bc_n_vel)

        history_loss_mon.append(loss_mon)
        history_loss_cont.append(loss_cont)
        history_loss_bc.append(loss_bc)
        history_loss_ct.append(loss_ct)

        if epoch<=3 or epoch%5000==0:
            print("Epoch: %-5d      MomentumLoss: %.6f     ContinuityLoss: %.6f   BoundaryLoss: %.6f    CostLoss: %.6f" % (epoch, loss_mon, loss_cont, loss_bc, loss_ct))

    checkpoints.save_checkpoint(DATAFOLDER, prefix="u_pinn_checkpoint_"+str(W_ct)+"_", target=u_state, step=u_state.step, overwrite=True)
    print("Training done, saved networks")

    costs_vs_weight.append(loss_ct)


## %%

    fig, ax = plt.subplots(1, 1, figsize=(6*1,4))

    plot(history_loss_mon[:], label='Momentum', x_label='epochs', y_scale="log", ax=ax)
    plot(history_loss_cont[:], label='Continuity', x_label='epochs', y_scale="log", ax=ax);
    plot(history_loss_bc[:], label='Boundary', x_label='epochs', y_scale="log", ax=ax);
    plot(history_loss_ct[:], label='Cost', x_label='epochs', title='Training loss - Cost weight='+str(W_ct), y_scale="log", ax=ax);


##%%
    fig, ax = plt.subplots(1, 2, figsize=(6*2,3.6))
    fig.suptitle("Cost weight: "+str(W_ct))

    vals_pinn = u_pinn.apply(u_state.params, cloud_vel.sorted_nodes / NORM_FACTOR)
    # u_pinn, v_pinn = vals_pinn[:, 0], vals_pinn[:, 1]
    p_pinn = p(cloud_p.sorted_nodes / NORM_FACTOR, u_state.params)

    vel_pinn = jnp.linalg.norm(vals_pinn[:, :2], axis=-1)
    print("After training:")

    cloud_vel.visualize_field(vel_pinn, cmap="jet", projection="2d", title="PINN forward velocity", ax=ax[0]);
    cloud_p.visualize_field(p_pinn, cmap="jet", projection="2d", title="PINN forward pressure", ax=ax[1]);


## %%

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6*2,5))
    pinn_control = c(y_inlet, c_state.params)
    pinn_sol_control = u(xy_inlet, u_state.params)

    plot(pinn_control, y_inlet, label="PINN control", title="Cost weight: "+str(W_ct), y_label=r"$y$", xlim=(-2.0, 2.5), ax=ax1);
    plot(pinn_sol_control, y_inlet, "--", label="PINN solution at Inlet", ax=ax1);

    u_outlet = u(xy_outlet, u_state.params)
    v_outlet = v(xy_outlet, u_state.params)
    parab_error = jnp.mean((u_outlet-y_outlet)**2)
    plot(u_parab, y_outlet, "-", label=r"$u$ target", y_label=r"$y$", xlim=(-0.1, 1.1), figsize=(5,3), ax=ax2)
    plot(u_outlet, y_outlet, "--", label=r"$u$ PINN", ax=ax2, title=f"Outlet velocity / MSE = {parab_error:.4f}");
    plot(jnp.zeros_like(y_outlet), y_outlet, "-", label=r"$v$ target", y_label=r"$y$", ax=ax2)
    plot(v_outlet, y_outlet, "--", label=r"$v$ PINN", ax=ax2);



    mem_usage = tracemalloc.get_traced_memory()[1]
    exec_time = time.process_time() - start

    print("A few performance details:")
    print(" Peak memory usage: ", mem_usage, 'bytes')
    print(' CPU execution time:', exec_time, 'seconds')

    jnp.savez(COMPFOLDER+"pinn_inv_3_"+str(W_id), objective_cost=history_loss_ct, mom_loss=history_loss_mon, cont_loss=history_loss_cont, bc_loss=history_loss_bc, pinn_control=pinn_control, pinn_sol_control=pinn_sol_control, vel_solution=vel_pinn, p_solution=p_pinn, u_target=u_parab, u_outlet=u_outlet, v_target=jnp.zeros_like(y_outlet), v_outlet=v_outlet, mem_time_cum=jnp.array([mem_usage, exec_time]))

    plt.show()

tracemalloc.stop()


# %%

W_ct_list = jnp.array(W_ct_list)
costs_vs_weight = jnp.array(costs_vs_weight)
ordering = jnp.argsort(W_ct_list)

plot(W_ct_list[ordering], costs_vs_weight[ordering], ".-", title='Step2: Inverse PINN', x_label='cost weights' , y_label='cost', x_scale="log", y_scale="log", figsize=(6,3));

## SOlution: PICK Weight = 0.1 !

# %%

jnp.savez(COMPFOLDER+"pinn_inv_3_final", weight_list=W_ct_list[ordering], cost_list=costs_vs_weight[ordering])
