#%%

import jax.numpy as jnp
from updec import *


facet_types = {"Wall":"n", "Inflow":"d", "Outflow":"d", "Cylinder":"n"}
cloud = GmshCloud(filename="./demos/meshes/channel_cylinder.py", facet_types=facet_types, mesh_save_location="./demos/meshes/")    ## TODO Pass the savelocation here


print(cloud.outward_normals)

cloud.visualize_cloud(figsize=(10,2))
plt.show()

# %%
