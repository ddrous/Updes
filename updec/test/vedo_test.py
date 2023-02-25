import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import seaborn as sns

# from updec import *

# RBF = polyharmonic
# MAX_DEGREE = 4
# Nx = 30
# Ny = Nx
# SUPPORT_SIZE = Nx*Ny-1

# facet_types={"south":"n", "west":"d", "north":"d", "east":"d"}
# cloud = SquareCloud(Nx=Nx, Ny=Ny, facet_types=facet_types)

# """Solve the wave equation using finite differences and the Euler method"""
# import numpy as np
# from scipy.ndimage import gaussian_filter
# from vedo import Plotter, Grid, Text2D











# """Solve the wave equation using finite differences and the Euler method"""
# import numpy as np
# from scipy.ndimage import gaussian_filter
# from vedo import Plotter, Grid, Text2D


# N = 400      # grid resolution
# A, B = 5, 4  # box sides
# end = 5      # end time
# nframes = 150

# X, Y = np.mgrid[-A:A:N*1j, -B:B:N*1j]
# dx = X[1,0] - X[0,0]
# dt = 0.1 * dx
# time = np.arange(0, end, dt)
# m = int(len(time)/nframes)

# # initial condition (a ring-like wave)
# Z0 = np.ones_like(X)
# Z0[X**2+Y**2 < 1] = 0
# Z0[X**2+Y**2 > 2] = 0
# Z0 = gaussian_filter(Z0, sigma=4)
# Z1 = np.array(Z0)

# grid = Grid(s=(X[:,0], Y[0])).linewidth(0).lighting('glossy')
# txt = Text2D(font='Brachium', pos='bottom-left', bg='yellow5')

# cam = dict(
#     pos=(5.715, -10.54, 12.72),
#     focal_point=(0.1380, -0.7437, -0.5408),
#     viewup=(-0.2242, 0.7363, 0.6384),
#     distance=17.40,
# )

# plt = Plotter(axes=1, size=(1000,700), interactive=False)
# plt.show(grid, txt, __doc__, camera=cam)

# for i in range(nframes):
#     # iterate m times before showing the frame
#     for _ in range(m):
#         ZC = Z1.copy()
#         Z1[1:N-1, 1:N-1] = (
#             2*Z1[1:N-1, 1:N-1]
#             - Z0[1:N-1, 1:N-1]
#             + (dt/dx)**2
#             * (  Z1[2:N,   1:N-1]
#                + Z1[0:N-2, 1:N-1]
#                + Z1[1:N-1, 0:N-2]
#                + Z1[1:N-1, 2:N  ]
#                - 4*Z1[1:N-1, 1:N-1] )
#         )
#         Z0[:] = ZC[:]

#     wave = Z1.ravel()
#     txt.text(f"frame: {i}/{nframes}, height_max = {wave.max()}")
#     grid.cmap("Blues", wave, vmin=-2, vmax=2)
#     newpts = grid.points()
#     newpts[:,2] = wave
#     grid.points(newpts)  # update the z component
#     plt.render()

# plt.interactive()
# plt.close()


# exit()






from vedo import Plotter, Grid, Text2D, Mesh
import numpy as np



xcoords = np.arange(0, 2, 0.2)
ycoords = np.arange(0, 1, 0.2)
# xcoords = sqrt(xcoords)
grid = Grid(s=(xcoords, ycoords), c="red5", alpha=0.5).lw(0)    ## Linwidth to 0 for interpolation
txt = Text2D(font='Brachium', pos='bottom-left', bg='yellow5')

grid = grid.compute_normals().add_scalarbar(c='black')

# grid.show(axes=8)
# mesh.show(axes=8)


# nv = mesh.ncells                           # nr. of cells
scals = grid.points()[:, 0]                          # coloring by the index of cell

# grid.cmap("hot", scals)
plt = Plotter(N=1, axes=1, interactive=False)
plt.show(grid, txt, __doc__)


for i in range(150):
    scals += 5
    grid.cmap("hot", scals, vmin=0, vmax=500)
    txt.text(f"frame: {i}/{150}")

    grid.add_scalarbar(horizontal=True)
    plt.show(grid, txt, __doc__)        ## Show again for colorbar
    
    # plt.render()


plt.interactive()
plt.close()





# plt = Plotter(N=1, axes=1, interactive=False)
# # mesh = Mesh(grid)
# mesh= grid
# nv = mesh.ncells                           # nr. of cells
# scals = mesh.cell_centers()[:, 0] + 37

# mesh.cmap("afmhot", scals, on='cells')
# mesh.add_scalarbar(horizontal=True)

# plt.show(mesh, "mesh.cmap(on='cells')")

# plt.interactive().close()






# """Boolean operations with Meshes"""
# from vedo import *

# settings.use_depth_peeling = True

# # declare the instance of the class
# plt = Plotter(shape=(2, 2), interactive=False, axes=3)

# # build to sphere meshes
# s1 = Sphere(pos=[-0.7, 0, 0], c="red5", alpha=0.5)
# s2 = Sphere(pos=[0.7, 0, 0], c="green5", alpha=0.5)
# plt.at(0).show(s1, s2, __doc__)

# # make 3 different possible operations:
# b1 = s1.boolean("intersect", s2).c('magenta')
# plt.at(1).show(b1, "intersect", resetcam=False)

# b2 = s1.boolean("plus", s2).c("blue").wireframe(True)
# plt.at(2).show(b2, "plus", resetcam=False)

# b3 = s1.boolean("minus", s2).compute_normals().add_scalarbar(c='white')
# plt.at(3).show(b3, "minus", resetcam=False)

# plt.interactive().close()