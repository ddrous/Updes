import jax
import jax.numpy as jnp
from updec.utils import distance, print_line_by_line


class Cloud(object):
    def __init__(self, Nx=7, Ny=5):
        self.Nx = Nx
        self.Ny = Ny
        self.N = self.Nx*self.Ny

        self.make_global_indices()

        self.make_node_coordinates()

        self.make_boundaries()

        self.make_local_supports()

        self.make_outward_normals()

        self.renumber_nodes()

        # self.visualise_cloud()        ## TODO


    def make_global_indices(self):
        ## defines the 2d to 1d indices and vice-versa

        self.global_indices = jnp.zeros((self.Nx, self.Ny), dtype=int)
        self.global_indices_rev = {}

        count = 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                self.global_indices = self.global_indices.at[i,j].set(count)
                self.global_indices_rev[count] = (i,j)
                count += 1


    def make_node_coordinates(self):
        x = jnp.linspace(0, 1., self.Nx)
        y = jnp.linspace(0, 1., self.Ny)
        xx, yy = jnp.meshgrid(x, y)

        self.nodes = {}

        for i in range(self.Nx):
            for j in range(self.Ny):
                global_id = int(self.global_indices[i,j])
                self.nodes[global_id] = jnp.array([xx[j,i], yy[j,i]])


    def make_boundaries(self):
        self.surfaces = {}              ## For each node id, say wihch surface it belongs to: 1 to 4

        self.boundaries = {}
        self.M = 0
        self.MD = 0
        self.MN = 0
        ## internal: 0
        ## dirichlet: 1
        ## neumann: 2
        ## external: -1     (not supported yet)
        for i in range(self.N):
            [k, l] = list(self.global_indices_rev[i])
            if k == 0:
                self.boundaries[i] = 1
                self.surfaces[i] = 2
                self.MD +=1
            elif l == 0:
                self.boundaries[i] = 1
                self.surfaces[i] = 1
                self.MD +=1
            elif l == self.Ny-1:
                self.boundaries[i] = 1
                self.surfaces[i] = 3
                self.MD +=1
            elif k == self.Nx-1:
                self.boundaries[i] = 2
                self.surfaces[i] = 4
                self.MN +=1
            else:
                self.boundaries[i] = 0
                self.surfaces[i] = 0        ## Number 0 is not a surface
                self.M +=1

        self.surface_types = {1:"dirichlet", 2:"dirichlet", 3:"dirichlet", 4:"neumann"}         ## For each surface id, say whether Dirichlet or Neumann - For the user


    def make_local_supports(self, n=7):
        ## finds the n nearest neighbords of each node
        self.local_supports = {}

        for i in range(self.N):
            distances = jnp.zeros((self.N), dtype=jnp.float32)
            for j in range(self.N):
                    distances = distances.at[j].set(distance(self.nodes[i], self.nodes[j]))

            closest_neighbours = jnp.argsort(distances)
            self.local_supports[i] = closest_neighbours[1:n+1].tolist()      ## node i is closest to itself


    def make_outward_normals(self):
        ## Makes the outward normal vectors to boundaries
        neumann_boundaries = {k:v for k,v in self.boundaries.items() if v==2}   ## Neumann node
        self.outward_normals = {}

        for i in neumann_boundaries.keys():
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

        ## If 0, 1, 2 convention already adopted
        # new_numbering = list(zip(*sorted(self.boundaries.items(), key=lambda x: x[1])))[0]
        # new_numbering = list(dict(sorted(self.boundaries.items(), key=lambda x: x[1])).keys())

        new_numbering = []
        for i in range(self.N):         ## Find all internals
            if self.boundaries[i] == 0:
                new_numbering.append(i)

        for i in range(self.N):         ## Find all dirichlet
            if self.boundaries[i] == 1:
                new_numbering.append(i)

        for i in range(self.N):         ## Find all neumann
            if self.boundaries[i] == 2:
                new_numbering.append(i)

        new_numb = {v:k for k, v in enumerate(new_numbering)}       ## Reads as: node k is now node v (usefull)

        self.global_indices_rev = {new_numb[k]: v for k, v in self.global_indices_rev.items()}
        for i, (k, l) in self.global_indices_rev.items():
            self.global_indices = self.global_indices.at[k, l].set(i)

        self.boundaries = {new_numb[k]:v for k,v in self.boundaries.items()}
        self.surfaces = {new_numb[k]:v for k,v in self.surfaces.items()}
        self.nodes = {new_numb[k]:v for k,v in self.nodes.items()}

        self.local_supports = jax.tree_util.tree_map(lambda i:new_numb[i], self.local_supports)
        self.local_supports = {new_numb[k]:v for k,v in self.local_supports.items()}

        self.outward_normals = {new_numb[k]:v for k,v in self.outward_normals.items()}

        self.renumbering_map = new_numb


    def print_cloud_properties(self):           ## Replace this with an appropriate __repr__ (a table would be nice)
        print("\n=== Meshfree cloud for RBF method ===\n")
        print()
        print("Cloud bounding box: Nx =", self.Nx, " -  Ny =", self.Ny)
        print()
        print("Boundary types (0=internal, 1=dirichlet, 2=neumann):\n", self.boundaries)
        print("Number of: \n\t-Internal points: M =", self.M, "\n\t-Dirichlet points: MD =", self.MD, "\n\t-Neumann points: MN =", self.MN)
        print("Surfaces:\n", self.surfaces)
        print()
        print("Global indices:\n", self.global_indices)
        print()
        # print("Global indices reversed:\n", self.global_indices_rev)
        # print()
        print("Outward normals on Neumann boundaries:")
        print_line_by_line(self.outward_normals)
        print()
        print("Node coordinates:", )
        print_line_by_line(self.nodes)
        print()
        print("Local supports (n closest neighbours):")
        print_line_by_line(self.local_supports)
