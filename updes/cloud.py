import warnings
import jax
import jax.numpy as jnp
from sklearn.neighbors import BallTree, KDTree
from updes.utils import distance

import os
from functools import cache

class Cloud(object):
    """ A class to store the meshfree cloud of points and related properties """

    def __init__(self, facet_types, support_size="max"):
        """Initializes the cloud of points

        Args:
            facet_types (dict): Name of the various facets (faces in 2D) of the domains (keys), and their boundary condition types (values). Types can be : 
            - "d" for dirichlet, 
            - "n" for neumann, 
            - "r" for robin, 
            - "p*" for periodic (where * is a letter of the alphabet to match a corresponding boundary).
            support_size (int, str): The size of the support (the closest centroids) to evaluate RBFs at each point of the cloud. Defaults to "max" to use all other centroids in the cloud.
        """

        self.N = 0 
        self.Ni = 0
        self.Nd = 0
        self.Nr = 0
        self.Nn = 0
        self.Np = []
        self.nodes = {}
        self.outward_normals = {}
        self.node_types = {}
        self.facet_nodes = {}
        self.facet_types = facet_types
        self.support_size = support_size
        self.dim = 2    ## TODO: implement 3-dimensional RBFs

        ## Facet precedence usefull for determining which facet a corner node belongs to
        self.facet_precedence = {k:i for i,(k,v) in enumerate(facet_types.items())}        

        ## For each periodic facet we encounter, we append a number to its end. This is useful for renumbering nodes and matching them
        new_facet_types = {}
        for i, (k,v) in enumerate(facet_types.items()):
            if v[0]=="p":
                new_facet_types[k] = v+str(i)
            else:
                new_facet_types[k] = v
        self.facet_types = new_facet_types

    def print_global_indices(self):
        
        """
        Print the global ids of the cloud of points.

        :return: A 2d array of the global indices
        """
        print(jnp.flip(self.global_indices.T, axis=0))

    def average_spacing(self):
        """ Computes the average distance between nodes in the cloud

        Returns:
            float: the mean distance between nodes
        """
        spacings = []
        for i in range(self.N):
            for j in range(i, self.N):
                spacings.append(distance(self.nodes[i], self.nodes[j]))
        return jnp.mean(jnp.array(spacings))

    def sort_dict_by_keys(self, dictionnary):
        """ Sorts a dictionnay whose values are jax arrays; and returns an array easily indexable """ 
        
        ## add the warning, only use this after nodes have been renumbered
        if not hasattr(self, "renumbering_map"):
            warnings.warn("Nodes have not been renumbered to place internals first, and boundary nodes afterwards. This function might not work properly")

        sorted_dict = sorted(dictionnary.items(), key=lambda x:x[0])
        sorted_dict = {k:jnp.array(v) for k,v in sorted_dict}
        return jnp.stack(list(dict(sorted_dict).values()), axis=-1).T

    def define_local_supports(self):
        """Finds the 'support_size' nearest neighbords of each node. This function is aware of the renumbering of nodes and uses the renumbering map to renumber the local supports accordingly.

        Raises:
            AssertionError: When the support size not between 1 (included) and N-1 (included)

        Returns:
            None: creates aditional attribute 'local_supports' in the class
        """
        self.local_supports = {}

        renumb_map = {i:k for i,k in enumerate(self.nodes.keys())}      ## TODO use the sorted nodes for this
        coords = jnp.stack(list(self.nodes.values()), axis=-1).T

        if self.support_size == "max" or self.support_size == None:
            self.support_size = coords.shape[0]
        assert self.support_size > 0, "Support size must be strictly greater than 0"
        assert self.support_size <= self.N, "Support size must be strictly less than or equal the number of nodes"

        ## Use BallTree for fast nearest neighbors search
        # ball_tree = KDTree(coords, leaf_size=40, metric='euclidean')
        ball_tree = BallTree(coords, leaf_size=40, metric='euclidean')
        for i in range(self.N):
            _, neighbours = ball_tree.query(self.nodes[i][jnp.newaxis], k=self.support_size)
            neighbours = neighbours[0][1:]  ## Result is a 2d list, without the first el (the node itself)
            self.local_supports[renumb_map[i]] = [renumb_map[j] for j in neighbours]


    def renumber_nodes(self):
        """Places the internal nodes at the top of the list, then the dirichlet, then neumann, then robin, then periodic nodes.

        Raises:
            ValueError: When an unknown node type is found

        Returns:
            None: creates additional attributes like: 'facet_nodes', 'renumbering_map' in the class. Also renumnbers the local supports, the outward normals, and the global indices
        """
        i_nodes = []
        d_nodes = []
        n_nodes = []
        r_nodes = []
        p_nodes = {}
        for i in range(self.N):
            if self.node_types[i][0] == "i":
                i_nodes.append(i)
            elif self.node_types[i][0] == "d":
                d_nodes.append(i)
            elif self.node_types[i][0] == "n":
                n_nodes.append(i)
            elif self.node_types[i][0] == "r":
                r_nodes.append(i)
            elif self.node_types[i][0] == "p":
                p_id = self.node_types[i]
                p_nodes[p_id] = p_nodes.get(p_id, [])+[i]
            else:
                raise ValueError("Unknown node type")

        ## Sort the keys of nodes with periodic bc before adding the values back to back 
        all_p_nodes = []
        for k in sorted(p_nodes.keys()):
            all_p_nodes += p_nodes[k]

        ## Renumber the nodes such that it reads as: node v is now node k
        new_numb = {v:k for k, v in enumerate(i_nodes+d_nodes+n_nodes+r_nodes+all_p_nodes)}

        if hasattr(self, "global_indices_rev"):
            self.global_indices_rev = {new_numb[k]: v for k, v in self.global_indices_rev.items()}
        if hasattr(self, "global_indices"):
            for i, (k, l) in self.global_indices_rev.items():
                self.global_indices = self.global_indices.at[k, l].set(i)

        self.node_types = {new_numb[k]:v for k,v in self.node_types.items()}
        self.nodes = {new_numb[k]:v for k,v in self.nodes.items()}

        if hasattr(self, 'local_supports'):
            self.local_supports = jax.tree_util.tree_map(lambda i:new_numb[i], self.local_supports)
            self.local_supports = {new_numb[k]:v for k,v in self.local_supports.items()}

        self.facet_nodes = jax.tree_util.tree_map(lambda i:new_numb[i], self.facet_nodes)
        if hasattr(self, 'facet_tag_nodes'):
            self.facet_tag_nodes = jax.tree_util.tree_map(lambda i:new_numb[i], self.facet_tag_nodes)

        if hasattr(self, 'outward_normals'):
            self.outward_normals = {new_numb[k]:v for k,v in self.outward_normals.items()}

        self.renumbering_map = new_numb



    def visualize_cloud(self, ax=None, title="Cloud", xlabel=r'$x$', ylabel=r'$y$', legend_size=8, figsize=(5.5,5), **kwargs):
        import matplotlib.pyplot as plt
        """Visualizes the cloud of points
        """

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)

        coords = self.sorted_nodes

        Ni, Nd, Nn = self.Ni, self.Nd, self.Nn
        if Ni > 0:
            ax.scatter(x=coords[:Ni, 0], y=coords[:Ni, 1], c="w", label="internal", **kwargs)
        if Nd > 0:
            ax.scatter(x=coords[Ni:Ni+Nd, 0], y=coords[Ni:Ni+Nd, 1], c="r", label="dirichlet", **kwargs)
        if Nn > 0:
            ax.scatter(x=coords[Ni+Nd:Ni+Nd+Nn, 0], y=coords[Ni+Nd:Ni+Nd+Nn, 1], c="g", label="neumann", **kwargs)
        if Ni+Nd+Nn < self.N:
            ax.scatter(x=coords[Ni+Nd+Nn:, 0], y=coords[Ni+Nd+Nn:, 1], c="b", label="robin", **kwargs)
        ## TODO: Add periodic nodes

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.0, 0.5), loc='center left', prop={'size': legend_size})
        plt.tight_layout()

        return ax


    def visualize_normals(self, ax=None, title="Normal vectors", xlabel=r'$x$', ylabel=r'$y$', figsize=(5.5,5), zoom_region=None, **kwargs):
        import matplotlib.pyplot as plt
        """Displays the outward normal vectors on Neumann and Robin boundaries"""

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)

        concerned_nodes = [i for i in range(self.N) if self.node_types[i] in ["n", "r"]]
        if len(concerned_nodes)==0:  ## Nothing to plot
            return ax

        coords = jnp.stack([self.nodes[node_id] for node_id in concerned_nodes], axis=0)
        normals = jnp.stack([self.outward_normals[node_id] for node_id in concerned_nodes], axis=0)/100     ## Devide by 100 for better visualization

        q = ax.quiver(coords[:,0], coords[:,1], normals[:,0], normals[:,1], color="w",label="normals", **kwargs)
        ax.quiverkey(q, X=0.5, Y=1.1, U=1, label='Normals', labelpos='E')
        ax.scatter(x=coords[:, 0], y=coords[:, 1], c="m", **kwargs)

        if xlabel: 
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_title(title)
        if zoom_region:     ## To zoom on a specific region
            ax.set_xlim((zoom_region[0], zoom_region[1]))
            ax.set_ylim((zoom_region[2], zoom_region[3]))
        plt.tight_layout()

        return ax


    def visualize_field(self, field, projection="2d", title="Field", xlabel=r'$x$', ylabel=r'$y$', levels=50, colorbar=True, ax=None, figsize=(6,5), extend="neither", **kwargs):
        import matplotlib.pyplot as plt
        """Visualizes the field on the cloud of points"""

        x, y = self.sorted_nodes[:, 0], self.sorted_nodes[:, 1]
        if len(field.shape) > 1:    ## If field has superfluous dimensions, e.g. (N, 1)
            field = field[:, 0, ...]

        if ax is None:
            fig = plt.figure(figsize=figsize)
            if projection == "2d":
                ax = fig.add_subplot(1, 1, 1)
            elif projection == "3d":
                ax = fig.add_subplot(1, 1, 1, projection='3d')

        if projection == "2d":
            img = ax.tricontourf(x, y, field, levels=levels, extend=extend, **kwargs)
            if colorbar == True:
                plt.sca(ax)
                plt.colorbar(img, extend=extend)

        elif projection == "3d":
            img = ax.plot_trisurf(x, y, field, **kwargs)
            # fig.colorbar(img, shrink=0.25, aspect=20)

        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        plt.tight_layout()

        return ax, img


    def animate_fields(self, fields, filename=None, titles="Field", xlabel=r'$x$', ylabel=r'$y$', levels=50, figsize=(6,5), cmaps="jet", cbarsplit=50, duration=5, **kwargs):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        """Animation of each field in the list of fields on the cloud of points"""

        ## Convert to arrays if not array already
        signals = []
        for field in fields:
            if isinstance(field, list):
                signals.append(jnp.stack(field, axis=0))
            else:
                signals.append(field)

        nb_signals = len(signals)

        x, y = self.sorted_nodes[:, 0], self.sorted_nodes[:, 1]

        fig, ax = plt.subplots(nb_signals, 1, figsize=figsize, sharex=True)
        if nb_signals == 1:
            ax = [ax]

        if not isinstance(cmaps, list):
            cmaps = [cmaps]*nb_signals

        ## Setup animation and colorbars
        imgs = []
        boundaries = []
        minmaxs = []
        for i in range(nb_signals):
            minmax = jnp.min(signals[i]), jnp.max(signals[i])

            ## To avoid crashing
            if minmax[1] <= minmax[0]:
                minmax = minmax[0], minmax[0]+1e-3

            minmaxs.append(minmax)
            boundaries = jnp.linspace(minmax[0], minmax[1], cbarsplit)

            imgs.append(ax[i].tricontourf(x, y, signals[i][0], levels=levels, vmin=minmax[0], vmax=minmax[1], cmap=cmaps[i], **kwargs))

            m = plt.cm.ScalarMappable(cmap=cmaps[i])
            m.set_array(signals[i])
            m.set_clim(minmax[0], minmax[1])
            # m.set_norm(plt.Normalize(vmin=minmax[0], vmax=minmax[1]))
            plt.colorbar(m, boundaries=boundaries, shrink=1.0, aspect=10, ax=ax[i])

            try:
                title = titles[i]
            except IndexError:
                title = "field # "+str(i+1)
            ax[i].set_title(title)

            if i == nb_signals-1:
                ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(ylabel)

        ## Animation function
        def animate(frame):
            imgs = [ax[i].tricontourf(x, y, signals[i][frame], levels=levels, vmin=minmaxs[i][0], vmax=minmaxs[i][1], cmap=cmaps[i], extend='min', **kwargs) for i in range(nb_signals)]
            # plt.suptitle("iter = "+str(i), size="large", y=0.95)      ## TODO doesn't work well with tight layout
            return imgs

        step_count = signals[0].shape[0]
        anim = FuncAnimation(fig, animate, frames=step_count, repeat=False, interval=100)
        plt.tight_layout()

        ### Save the video
        if filename:
            fps = step_count / duration
            writer = 'ffmpeg' if filename.endswith('.mp4') else 'pillow'
            anim.save(filename, writer=writer, fps=fps)
            print("Animation sucessfully saved at:", filename)

        return ax



















class SquareCloud(Cloud):
    """ A class to store a cloud of (scattered or regular) points on a unit square """


    def __init__(self, Nx=7, Ny=5, noise_key=None, **kwargs):
        """Initializes a unit square cloud of points

        Args:
            Nx (int, optional): Number of points alont the x direction. Defaults to 7.
            Ny (int, optional): Number of points alont the y direction. Defaults to 5.
            noise_key (int, optional): An optional noise key to scramble the points' locations. Defaults to None.
        """
        super().__init__(**kwargs)

        self.Nx = Nx
        self.Ny = Ny
        self.N = self.Nx*self.Ny

        self.define_global_indices()
        self.define_node_types()
        self.define_node_coordinates(noise_key)
        self.define_local_supports()
        self.define_outward_normals()
        self.renumber_nodes()

        self.sorted_nodes = self.sort_dict_by_keys(self.nodes)

        self.sorted_local_supports = self.sort_dict_by_keys(self.local_supports)

        if len(self.outward_normals) > 0:
            self.sorted_outward_normals = self.sort_dict_by_keys(self.outward_normals)


    def define_global_indices(self):
        """Defines the 2d coordinates to 1d indices map and vice-versa
        """
        self.global_indices = jnp.zeros((self.Nx, self.Ny), dtype=int)
        self.global_indices_rev = {}

        count = 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                self.global_indices = self.global_indices.at[i,j].set(count)
                self.global_indices_rev[count] = (i,j)
                count += 1


    def define_node_coordinates(self, noise_key):
        """ Calculates the coordinates of the nodes in the square domain. Optionally adds noise to the coordinates """
        x = jnp.linspace(0, 1., self.Nx)
        y = jnp.linspace(0, 1., self.Ny)
        xx, yy = jnp.meshgrid(x, y)

        if noise_key is not None:
            key = jax.random.split(noise_key, self.N)
            delta_noise = min((x[1]-x[0], y[1]-y[0])) / 2.   ## To make sure nodes don't go into each other

        self.nodes = {}

        for i in range(self.Nx):
            for j in range(self.Ny):
                global_id = int(self.global_indices[i,j])

                if (self.node_types[global_id] not in ["d", "n", "r"]) and (noise_key is not None):
                    noise = jax.random.uniform(key[global_id], (2,), minval=-delta_noise, maxval=delta_noise)
                else:
                    noise = jnp.zeros((2,))

                self.nodes[global_id] = jnp.array([xx[j,i], yy[j,i]]) + noise


    def define_node_types(self):
        """ Defines the boundary dype for each node in the square domain """

        self.facet_nodes = {k:[] for k in self.facet_types.keys()}     ## List of nodes belonging to each facet
        self.node_types = {}    ## internal="i", dirichlet="d", neumann="n", robin="r", pediodic="p*"

        for i in range(self.N):
            [k, l] = list(self.global_indices_rev[i])
            if l == self.Ny-1:
                self.facet_nodes["North"].append(i)
                self.node_types[i] = self.facet_types["North"]
            elif l == 0:
                self.facet_nodes["South"].append(i)
                self.node_types[i] = self.facet_types["South"]
            elif k == self.Nx-1:
                self.facet_nodes["East"].append(i)
                self.node_types[i] = self.facet_types["East"]
            elif k == 0:
                self.facet_nodes["West"].append(i)
                self.node_types[i] = self.facet_types["West"]
            else:
                self.node_types[i] = "i"       ## Internal node (not a boundary). But very very important to be numbered!

        self.Nd = 0
        self.Nn = 0
        self.Nr = 0
        self.Np = {v[:-1]:0 for v in self.facet_types.values() if v[0]=="p"}    ## Number of nodes per periodic boundary id

        for f_id, f_type in self.facet_types.items():
            if f_type == "d":
                self.Nd += len(self.facet_nodes[f_id])
            if f_type == "n":
                self.Nn += len(self.facet_nodes[f_id])
            if f_type == "r":
                self.Nr += len(self.facet_nodes[f_id])
            if f_type[0] == "p":
                self.Np[f_type[:-1]] += len(self.facet_nodes[f_id])

        ## Get periodic count as a list sorted by keys
        self.Np = [self.Np[k] for k in sorted(self.Np.keys())]

        self.Ni = self.N - self.Nd - self.Nn - self.Nr - sum(self.Np)


    def define_outward_normals(self):
        """ Makes the outward normal vectors to nodes on the boundaries
        """
        bd_nodes = [k for k,v in self.node_types.items() if v[0] in ["n", "r", "p"]]   ## Neumann, Robin, or Periodic nodes
        self.outward_normals = {}

        for i in bd_nodes:
            k, l = self.global_indices_rev[i]
            if l==self.Ny-1:
                n = jnp.array([0., 1.])
            elif l==0:
                n = jnp.array([0., -1.])
            elif k==self.Nx-1:
                n = jnp.array([1., 0.])
            elif k==0:
                n = jnp.array([-1., 0.])

            self.outward_normals[int(self.global_indices[k,l])] = n




















class GmshCloud(Cloud):
    """ A class to store the cloud of points issued from GMSH and related properties """

    def __init__(self, filename, mesh_save_location=None, **kwargs):
        """ Initializes the cloud of points from a Gmsh mesh file

        Args:
            filename (str): a Gmsh mesh file in either .msh or .py format. 
            - If a .msh file is provided, the mesh is loaded directly assuming the Gmsh format it was generated with is 4.0.8
            - If a .py file is provided, the mesh is generated using the `gmsh` Python package with format 4.0.8, then saved.
            mesh_save_location (str, optional): The path to save the .msh file if a .py file was provided. Defaults to None.
        """
        super().__init__(**kwargs)

        self.get_meshfile(filename, mesh_save_location)

        self.extract_nodes_and_boundary_type()
        self.define_outward_normals()
        self.define_local_supports()
        self.renumber_nodes()

        self.sorted_nodes = self.sort_dict_by_keys(self.nodes)

        self.sorted_local_supports = self.sort_dict_by_keys(self.local_supports)

        if len(self.outward_normals) > 0:
            self.sorted_outward_normals = self.sort_dict_by_keys(self.outward_normals)


    def get_meshfile(self, filename, mesh_save_location):
        """ Gets the definitive mesh file to work with

        Args:
            filename (str): see __init__()
            mesh_save_location (str): see __init__(). If None, the mesh is saved in the same directory as the filename
        """
        if mesh_save_location is None:
            mesh_save_location = os.path.dirname(filename)+"/"

        _, extension = filename.rsplit('.', maxsplit=1)
        if extension == "msh":   ## Gmsh Geo file
            self.filename = filename
        elif extension == "py":  ## Gmsh Python API
            os.system("python "+filename + " " + mesh_save_location +" --nopopup")
            self.filename = mesh_save_location+"mesh.msh"


    def extract_nodes_and_boundary_type(self):
        """ Extract nodes and boundary types from a .msh file according to Gmsh format 4.0.8 """

        f = open(self.filename, "r")

        #--- Facet names ---#
        line = f.readline()
        while line.find("$PhysicalNames") < 0: line = f.readline()
        splitline = f.readline().split()

        facet_physical_names = {}
        nb_facets = int(splitline[0]) - 1
        for facet in range(nb_facets):
            splitline = f.readline().split()
            facet_physical_names[int(splitline[1])] = (splitline[2])[1:-1]    ## Removes quotes

        #--- Physical names to entities ---#
        self.facet_names = {}
        line = f.readline()
        while line.find("$Entities") < 0: line = f.readline()
        splitline = f.readline().split()
        n_vertices, n_facets = int(splitline[0]), int(splitline[1])
        for _ in range(n_vertices):
            line = f.readline()     ## Skip the vertices
        for _ in range(n_facets):
            splitline = f.readline().split()     ## Skip the vertices
            self.facet_names[int(splitline[0])] = facet_physical_names[int(splitline[-4])]

        #--- Reading mesh nodes ---#
        line = f.readline()
        while line.find("$Nodes") < 0: line = f.readline()
        splitline = f.readline().split()

        self.N = int(splitline[1])
        self.nodes = {}
        self.facet_nodes = {v:[] for v in self.facet_names.values()}
        self.facet_tag_nodes = {k:[] for k in self.facet_names.keys()}        ## Useful for normals
        self.node_types = {}
        corner_membership = {}

        line = f.readline()
        while line.find("$EndNodes") < 0:
            splitline = line.split()
            entity_id = int(splitline[0])
            dim = int(splitline[1])
            nb = int(splitline[-1])
            facet_nodes = []

            for i in range(nb):
                splitline = f.readline().split()
                node_id = int(splitline[0]) - 1
                x = float(splitline[1])
                y = float(splitline[2])
                z = float(splitline[3])

                self.nodes[node_id] = jnp.array([x, y])

                if dim==0: ## A corner point
                    corner_membership[node_id] = []

                elif dim==1:  ## A curve
                    self.node_types[node_id] = self.facet_types[self.facet_names[entity_id]]
                    facet_nodes.append(node_id)

                elif dim==2:  ## A surface
                    self.node_types[node_id] = "i"

            if dim==1:
                self.facet_nodes[self.facet_names[entity_id]] += facet_nodes
                self.facet_tag_nodes[entity_id] += facet_nodes

            line = f.readline()

        # --- Read mesh elements for corner nodes ---#
        while line.find("$Elements") < 0: line = f.readline()
        f.readline()

        line = f.readline()
        while line.find("$EndElements") < 0:
            splitline = line.split()
            entity_id = int(splitline[0])
            dim = int(splitline[1])
            nb = int(splitline[-1])

            if dim == 1:                ## Only considering elements of dim=DIM-1
                for i in range(nb):
                    splitline = [int(n_id)-1 for n_id in f.readline().split()[1:]]

                    for c_node_id in corner_membership.keys():
                        if c_node_id in splitline:
                            for neighboor in splitline:
                                if neighboor != c_node_id:
                                    corner_membership[c_node_id].append(entity_id)
                                    break

            else:
                for i in range(nb): f.readline()

            line = f.readline()

        f.close()

        ## Sort the entity ids by precedence
        for c_id, f_ids in corner_membership.items():
            sorted_f_ids = sorted(f_ids, key=lambda f_id:self.facet_precedence[self.facet_names[f_id]])
            choosen_facet_id = sorted_f_ids[0]   ## The corner node belongs to this facet exclusively
            choosen_facet_name = self.facet_names[choosen_facet_id]

            self.node_types[c_id] = self.facet_types[choosen_facet_name]
            self.facet_nodes[choosen_facet_name].append(c_id)
            self.facet_tag_nodes[choosen_facet_id].append(c_id)


        self.Ni = len({k:v for k,v in self.node_types.items() if v[0]=="i"})
        self.Nd = len({k:v for k,v in self.node_types.items() if v[0]=="d"})
        self.Nr = len({k:v for k,v in self.node_types.items() if v[0]=="r"})
        self.Nn = len({k:v for k,v in self.node_types.items() if v[0]=="n"})



    def define_outward_normals(self):
        """ Calculates (approximate) outward normal vectors to nodes on the boundaries. 
        Raises:
            AssertionError: When the mesh is not fine enough to compute normals
        """
        ## In 3D, we could use the Gmesh API: https://stackoverflow.com/a/59279502/8140182

        ## To get the closes internal point
        in_coords = jnp.stack([self.nodes[node_id] for node_id in range(self.N) if self.node_types[node_id] == "i"], axis=0)
        in_ball_tree = BallTree(in_coords, leaf_size=40, metric='euclidean')
        # in_ball_tree = KDTree(in_coords, leaf_size=40, metric='euclidean')

        for f_tag, f_nodes in self.facet_tag_nodes.items():

            if self.facet_types[self.facet_names[f_tag]][0] in ["n", "r", "p"]:      ### Only Neuman, Robin, and Periodic need normals !
                assert len(f_nodes) >= 2, " Mesh not fine enough for normal computation "

                ## Sort the nodes in this facet
                f_coords = jnp.stack([self.nodes[node_id] for node_id in f_nodes], axis=0)
                f_ball_tree = BallTree(f_coords, leaf_size=40, metric='euclidean')

                for node_id in f_nodes:
                    current = self.nodes[node_id]
                    _, neighbours = f_ball_tree.query(current[jnp.newaxis], k=2)
                    closest_f = f_coords[neighbours[0][1]]      ## The closest point on the same facet

                    _, neighbours = in_ball_tree.query(current[jnp.newaxis], k=2)
                    closest_in = in_coords[neighbours[0][1]]    ## The closest point in the domain

                    invector = closest_in - current         ## An inward pointing vector
                    tangent = closest_f - current           ## A tangent vector

                    normal = jnp.array([-tangent[1], tangent[0]])
                    if jnp.dot(normal, invector) > 0:       ## The normal is pointing inward
                        self.outward_normals[node_id] = -normal / jnp.linalg.norm(normal)
                    else:                                   ## The normal is pointing outward
                        self.outward_normals[node_id] = normal / jnp.linalg.norm(normal)
