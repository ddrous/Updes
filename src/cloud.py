import jax
import jax.numpy as jnp
from utils import distance


class Cloud(object):
    def __init__(self, Nx=10, Ny=10):
        self.Nx = Nx
        self.Ny = Ny
        self.N = self.Nx*self.Ny

        self.make_global_indices()

        self.make_node_coordinates()

        self.make_boundaries()

        self.make_local_supports()

        self.renumber_nodes()

    def make_global_indices(self):
        ## defines the 2d to 1d indices and vice-versa

        self.global_indices = jnp.zeros((self.Nx, self.Ny), dtype=jnp.int32)
        self.global_indices_rev = jnp.zeros((self.N, 2), dtype=jnp.int32)

        count = 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                self.global_indices = self.global_indices.at[i,j].set(count)
                self.global_indices_rev = self.global_indices_rev.at[count].set(jnp.array([i,j]))
                count += 1

    def make_node_coordinates(self):
        x = jnp.linspace(0, 1., self.Nx)
        y = jnp.linspace(0, 1., self.Ny)
        xx, yy = jnp.meshgrid(x, y)

        self.nodes = jnp.zeros((self.N, 2), dtype=jnp.float32)

        for i in range(self.Nx):
            for j in range(self.Ny):
                global_id = self.global_indices[i,j]
                self.nodes = self.nodes.at[global_id].set(jnp.array([xx[j,i], yy[j,i]]))

    def make_boundaries(self):
        bds = jnp.zeros((self.Nx, self.Ny), dtype=jnp.int32)
        ## internal: 0
        ## dirichlet: 1
        ## neumann: 2
        ## external: -1     (not supported yet)
        bds = jnp.zeros((self.N), dtype=jnp.int32)
        for i in range(self.N):
            [k, l] = list(self.global_indices_rev[i])
            if k == 0 or l == 0 or l == self.Ny-1:
                bds = bds.at[i].set(1)
            elif k == self.Nx-1:
                bds = bds.at[i].set(2)

        self.M = jnp.size(bds[bds==0])
        self.MD = jnp.size(bds[bds==1])
        self.MN = jnp.size(bds[bds==2])
        self.boundaries = bds

    def make_local_supports(self, n=7):
        ## finds the n nearest neighbords of each node
        self.local_supports = jnp.zeros((self.N, n), dtype=jnp.int32)

        for i in range(self.N):
            distances = jnp.zeros((self.N), dtype=jnp.float32)
            for j in range(self.N):
                    distances = distances.at[j].set(distance(self.nodes[i], self.nodes[j]))

            closest_neighbours = jnp.argsort(distances)
            self.local_supports = self.local_supports.at[i].set(closest_neighbours[1:n+1])      ## node i is closest to itself

    def renumber_nodes(self):
        """ Places the internal nodes at the top of the list, then the dirichlet, then neumann: good for matrix afterwards """

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

        new_numb_fw = jnp.array(new_numbering)

        self.global_indices_rev = self.global_indices_rev[new_numb_fw]
        for k, ij in enumerate(self.global_indices_rev):
            self.global_indices = self.global_indices.at[ij[0], ij[1]].set(k)

        self.boundaries = self.boundaries[new_numb_fw]
        self.nodes = self.nodes[new_numb_fw]

        self.local_supports = self.local_supports[new_numb_fw]
        new_numb_rev = {v:k for k, v in enumerate(new_numbering)}
        for i in range(self.N):
            neighbours = self.local_supports[i].tolist()
            for k, j in enumerate(neighbours):
                self.local_supports = self.local_supports.at[i,k].set(new_numb_rev[j])

        self.numbering_map = {k:v for k, v in enumerate(new_numbering)} ## Reads as: node k used to be node v

if __name__ == '__main__':
    cloud = Cloud(Nx=7, Ny=5)
    print("\n=== Meshfree cloud for RBF-FD method ===\n")
    print()
    print("Cloud bounding box: Nx =", cloud.Nx, " -  Ny =", cloud.Ny)
    print()
    print("Global indices:\n", cloud.global_indices)
    # print("Global indices reversed:\n", cloud.global_indices_rev)
    print()
    print("Boundary types - 0=internal, 1=dirichlet, 2=neumann:\n", cloud.boundaries)
    print("Number of: \n\t-Internal points: M =", cloud.M, "\n\t-Dirichlet points: MD =", cloud.MD, "\n\t-Neumann points: MN =", cloud.MN)
    print()
    print("Node coordinates:\n", cloud.nodes)
    print()
    print("Local supports (n closest neighbours):\n", cloud.local_supports)
