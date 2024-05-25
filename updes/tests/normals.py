#%%

import jax.numpy as jnp
from updes import *


facet_types = {"Wall":"n", "Inflow":"n", "Outflow":"n", "Cylinder":"n"}
cloud = GmshCloud(filename="./data/", facet_types=facet_types, mesh_save_location="./data/")    ## TODO Pass the savelocation here

# cloud.visualize_cloud(figsize=(10,2), s=4)
# # cloud.visualize_normals(figsize=(10,10))
# cloud.visualize_normals(figsize=(10,10), zoom_region=(-0.25,0.25,-0.25,0.25))
# # plt.show()

print(cloud.facet_names)
print(cloud.facet_types)
print(cloud.facet_tag_nodes[3])

test_node = cloud.renumbering_map[55]
print(cloud.nodes[test_node])
print(cloud.outward_normals[test_node])

