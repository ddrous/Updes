import jax
import jax.numpy as jnp
from sklearn.neighbors import BallTree

from updec.utils import distance


class Cloud(object):
    def __init__(self, Nx=7, Ny=5, facet_types={0:"d", 1:"d", 2:"d", 3:"n"}, support_size=35, noise_seed=None):
        self.Nx = Nx
        self.Ny = Ny
        self.N = self.Nx*self.Ny
        self.facet_types = facet_types

        self.define_global_indices()

        self.define_node_boundary_types()

        self.define_node_coordinates(noise_seed)

        self.define_local_supports(support_size)

        self.define_outward_normals()

        self.renumber_nodes()

        # self.visualise_cloud()        ## TODO Finsih this properly


    def define_global_indices(self):
        ## defines the 2d to 1d indices and vice-versa

        self.global_indices = jnp.zeros((self.Nx, self.Ny), dtype=int)
        self.global_indices_rev = {}

        count = 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                self.global_indices = self.global_indices.at[i,j].set(count)
                self.global_indices_rev[count] = (i,j)
                count += 1


    def define_node_coordinates(self, noise_seed):
        """ Can be used to redefine coordinates for performance study """
        x = jnp.linspace(0, 1., self.Nx)
        y = jnp.linspace(0, 1., self.Ny)
        xx, yy = jnp.meshgrid(x, y)

        if noise_seed is not None:
            key = jax.random.split(jax.random.PRNGKey(seed=noise_seed), self.N)
            delta_noise = min((x[1]-x[0], y[1]-y[0])) / 2.   ## To make sure nodes don't go into each other

        self.nodes = {}

        for i in range(self.Nx):
            for j in range(self.Ny):
                global_id = int(self.global_indices[i,j])

                if self.node_boundary_types[global_id] not in ["d", "n"]:
                    noise = jax.random.uniform(key[global_id], (2,), minval=-delta_noise, maxval=delta_noise)         ## Just add some noisy noise !!
                else:
                    noise = jnp.zeros((2,))

                self.nodes[global_id] = jnp.array([xx[j,i], yy[j,i]]) + noise


    def define_node_boundary_types(self):
        """ Makes the boundaries for the square domain """

        self.facet_nodes = {k:[] for k in self.facet_types.keys()}     ## List of nodes belonging to each facet
        self.node_boundary_types = {}                              ## Coding structure: internal="i", dirichlet="d", neumann="n", external="e" (not supported yet)

        for i in range(self.N):
            [k, l] = list(self.global_indices_rev[i])
            if l == 0:            ## Surface number 1
                self.facet_nodes[0].append(i)
                self.node_boundary_types[i] = self.facet_types[0]
            elif k == 0:              ## Surface number 2
                self.facet_nodes[1].append(i)
                self.node_boundary_types[i] = self.facet_types[1]
            elif l == self.Ny-1:    ## Surface number 3
                self.facet_nodes[2].append(i)
                self.node_boundary_types[i] = self.facet_types[2]
            elif k == self.Nx-1:    ## Surface number 4
                self.facet_nodes[3].append(i)
                self.node_boundary_types[i] = self.facet_types[3]
            else:
                self.node_boundary_types[i] = "i"       ## Internal node (not a boundary). But very very important!

        self.Nd = 0
        self.Nn = 0
        for f_id, f_type in self.facet_types.items():
            if f_type == "d":
                self.Nd += len(self.facet_nodes[f_id])
            if f_type == "n":
                self.Nn += len(self.facet_nodes[f_id])

        self.Ni = self.N - self.Nd - self.Nn


    def define_local_supports(self, support_size):
        ## finds the 'support_size' nearest neighbords of each node
        self.local_supports = {}
        assert support_size <= self.N-1, "Support size must be strictly less than the number of nodes"

        #### BALL TREE METHOD
        renumb_map = {i:k for i,k in enumerate(self.nodes.keys())}
        coords = jnp.stack(list(self.nodes.values()), axis=-1).T
        # ball_tree = KDTree(coords, leaf_size=40, metric='euclidean')
        ball_tree = BallTree(coords, leaf_size=40, metric='euclidean')
        for i in range(self.N):
            _, neighboorhs = ball_tree.query(coords[i:i+1], k=support_size+1)
            neighboorhs = neighboorhs[0][1:]                    ## Result is a 2d list, with the first el itself
            self.local_supports[renumb_map[i]] = [renumb_map[j] for j in neighboorhs]

        #### BRUTE FORCE METHOD
        # for i in range(self.N):
        #     distances = jnp.zeros((self.N), dtype=jnp.float32)
        #     for j in range(self.N):
        #             distances = distances.at[j].set(distance(self.nodes[i], self.nodes[j]))

        #     closest_neighbours = jnp.argsort(distances)
        #     self.local_supports[i] = closest_neighbours[1:n+1].tolist()      ## node i is closest to itself




    def define_outward_normals(self):
        ## Makes the outward normal vectors to boundaries
        neumann_nodes = [k for k,v in self.node_boundary_types.items() if v=="n"]   ## Neumann nodes
        self.outward_normals = {}

        for i in neumann_nodes:
            k, l = self.global_indices_rev[i]
            if k==0:
                n = jnp.array([-1., 0.])
            elif k==self.Nx-1:
                n = jnp.array([1., 0.])
            elif l==0:
                n = jnp.array([0., -1.])
            elif l==self.Ny-1:
                n = jnp.array([0., 1.])

            self.outward_normals[int(self.global_indices[k,l])] = n


    def renumber_nodes(self):
        """ Places the internal nodes at the top of the list, then the dirichlet, then neumann: good for matrix afterwards """

        i_nodes = []
        d_nodes = []
        n_nodes = []
        for i in range(self.N):         
            if self.node_boundary_types[i] == "i":
                i_nodes.append(i)
            elif self.node_boundary_types[i] == "d":
                d_nodes.append(i)
            elif self.node_boundary_types[i] == "n":
                n_nodes.append(i)

        new_numb = {v:k for k, v in enumerate(i_nodes+d_nodes+n_nodes)}       ## Reads as: node k is now node v

        self.global_indices_rev = {new_numb[k]: v for k, v in self.global_indices_rev.items()}
        for i, (k, l) in self.global_indices_rev.items():
            self.global_indices = self.global_indices.at[k, l].set(i)

        self.node_boundary_types = {new_numb[k]:v for k,v in self.node_boundary_types.items()}
        self.nodes = {new_numb[k]:v for k,v in self.nodes.items()}

        if hasattr(self, 'local_supports'):
            self.local_supports = jax.tree_util.tree_map(lambda i:new_numb[i], self.local_supports)
            self.local_supports = {new_numb[k]:v for k,v in self.local_supports.items()}

        self.facet_nodes = jax.tree_util.tree_map(lambda i:new_numb[i], self.facet_nodes)

        self.outward_normals = {new_numb[k]:v for k,v in self.outward_normals.items()}

        self.renumbering_map = new_numb


    def visualize_cloud(self, ax=None, figsize=(6,5), **kwargs):
        import matplotlib.pyplot as plt
        ## TODO Color and print important stuff appropriately

        if ax is None:
            fig = plt.figure(figsize=figsize)

        sorted_nodes = sorted(self.nodes.items(), key=lambda x:x[0])
        coords = jnp.stack(list(dict(sorted_nodes).values()), axis=-1).T

        ax = fig.add_subplot(1, 1, 1)

        # colours = []
        # for i in range(self.N):
        #     if self.node_boundary_types[i] == "i":
        #         colours.append("k")
        #     elif self.node_boundary_types[i] == "d":
        #         colours.append("r")
        #     elif self.node_boundary_types[i] == "n":
        #         colours.append("g")

        # ax.scatter(x=coords[:, 0], y=coords[:, 1], c=colours, **kwargs)

        # groups = [0, 1, 2]
        # cdict = dict(zip(["i", "d", "n"], ["k", "r", "g"]))
        Ni, Nd = self.Ni, self.Nd
        ax.scatter(x=coords[:Ni, 0], y=coords[:Ni, 1], c="k", label="internal", **kwargs)
        ax.scatter(x=coords[Ni:Ni+Nd, 0], y=coords[Ni:Ni+Nd, 1], c="r", label="dirichlet", **kwargs)
        ax.scatter(x=coords[Ni+Nd:, 0], y=coords[Ni+Nd:, 1], c="g", label="neumann", **kwargs)

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.legend(loc='upper right')

        return ax


    def visualize_field(self, field, projection, levels=50, ax=None, figsize=(6,5), **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage.filters import gaussian_filter

        sorted_nodes = sorted(self.nodes.items(), key=lambda x:x[0])
        coords = jnp.stack(list(dict(sorted_nodes).values()), axis=-1).T
        x, y = coords[:, 0], coords[:, 1]

        if ax is None:
            fig = plt.figure(figsize=figsize)

        if projection == "2d":
            ax = fig.add_subplot(1, 1, 1)
            # img = ax.scatter(x=coords[:, 0], y=coords[:, 1], c=field, **kwargs)
            # fig.colorbar(img)

            # img = ax.hexbin(x=x, y=y, C=field, **kwargs)
            # ax.set_xlim([x.min(), x.max()])
            # ax.set_ylim([y.min(), y.max()])
            # fig.colorbar(img)

            img = ax.tricontourf(x, y, field, levels=levels, **kwargs)
            fig.colorbar(img)

        elif projection == "3d":
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            img = ax.plot_trisurf(x, y, field, **kwargs)
            # fig.colorbar(img, shrink=0.25, aspect=20)

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

        return ax, img
