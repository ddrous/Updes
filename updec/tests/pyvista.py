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










# """Create a simple Play/Pause app with a timer event
# You can interact with the scene during the loop!
# ..press q to quit"""
# import time
# import numpy as np
# from vedo import Plotter
# from vedo.pyplot import plot


# def bfunc():
#     global timer_id
#     plotter.timer_callback("destroy", timer_id)
#     if "Play" in button.status():
#         # instruct to call handle_timer() every 10 msec:
#         timer_id = plotter.timer_callback("create", dt=10)
#     button.switch()

# def handle_timer(event):
#     t = time.time() - t0
#     x = np.linspace(t, t + 4*np.pi, 50)
#     y = np.sin(x) * np.sin(x/12)
#     fig = plot(
#         x, y, '-o', ylim=(-1.2, 1.2), aspect=3/1,
#         xtitle="time window [s]", ytitle="intensity [a.u.]",
#     )
#     fig.shift(-x[0]) # put the whole plot object back at (0,0)
#     # pop (remove) the old plot and add the new one
#     plotter.pop().add(fig)


# timer_id = None
# t0 = time.time()
# plotter= Plotter(size=(1200,600))
# button = plotter.add_button(bfunc, states=[" Play ","Pause"], size=40)
# evntId = plotter.add_callback("timer", handle_timer)

# x = np.linspace(0, 4*np.pi, 50)
# y = np.sin(x) * np.sin(x/12)
# fig = plot(x, y, ylim=(-1.2, 1.2), xtitle="time", aspect=3/1, lc='grey5')

# plotter.show(__doc__, fig, zoom=2)

# exit()






# """
# Reconstruct a polygonal surface
# from a point cloud:
#  1. An object is loaded and
#     noise is added to its vertices.
#  2. The point cloud is smoothened
#     with MLS (Moving Least Squares)
#  3. Impose a minimum distance among points
#  4. A triangular mesh is extracted from
#     this set of sparse Points.
# """
# from vedo import dataurl, printc, Plotter, Points, Mesh, Text2D


# plt = Plotter(shape=(1,5))
# # plt.at(0).show(Text2D(__doc__, s=0.75, font='Theemim', bg='green5'))

# # 1. load a mesh
# mesh = Mesh(dataurl+"apple.ply").subdivide()
# plt.at(1).show(mesh)

# # Add noise
# pts0 = Points(mesh, r=3).add_gaussian_noise(1)
# plt.at(2).show(pts0)

# # 2. Smooth the point cloud with MLS
# pts1 = pts0.clone().smooth_mls_2d(f=0.8)
# printc("Nr of points before cleaning nr. points:", pts1.npoints)

# # 3. Impose a min distance among mesh points
# pts1.subsample(0.05)
# printc("             after  cleaning nr. points:", pts1.npoints)
# plt.at(3).show(pts1)

# # 4. Reconstruct a polygonal surface from the point cloud
# reco = pts1.reconstruct_surface(dims=100, radius=0.2).c("gold")
# plt.at(4).show(reco, axes=7, zoom=1.2)

# plt.interactive().close()


# exit()


















# from vedo import Plotter, Grid, Text2D, Points, settings, Mesh, Spline, Video
# import numpy as np

# settings.default_font = "Theemim"


# xcoords = np.random.uniform(0,2,10)
# # xcoords = [0]*3 + [0.25]*3  + [0.5]*3  + [0.75]*3  + [1]*3
# ycoords = np.random.uniform(0,1,10)
# # ycoords = [0, 0.25, 0.5, 0.75, 1]*3
# zcoords = np.zeros_like(xcoords)

# # shape = Spline((xcoords, ycoords), closed=True).color('red4').linewidth(5)

# grid_ = Points((xcoords, ycoords, zcoords))
# # grid = Mesh(grid, c="red5", alpha=0.5).lw(0).wireframe(True)
# # msh = shape.generate_mesh(grid=grid, quads=True)

# ### TODO OPTIONS
# # reco = grid.reconstruct_surface(dims=100, radius=0.2, bounds=(0,1,0,1,0,0)).c("gold")

# grid = grid_.tomesh(quads=False).lw(0)

# # print(grid)

# # print(grid.cell_centers())
# # print(grid.points())

# # plt = Plotter(shape=(1,2))
# # plt.at(0).show(grid, axes=7, zoom=1.2)
# # plt.at(1).show(reco, axes=7, zoom=1.2)
# # plt.interactive().close()
# # exit()

# txt = Text2D(font='Brachium', pos='bottom-left', bg='yellow5')
# # grid = grid.compute_normals().add_scalarbar(c='black')

# # grid.show(axes=8)
# # mesh.show(axes=8)


# # nv = mesh.ncells                           # nr. of cells
# scals = grid.points()[:, 0]                          # coloring by the index of cell

# # grid.cmap("hot", scals)
# plt = Plotter(N=1, axes=1, interactive=False)
# plt.show(grid, txt, __doc__)

# # Open a video file and force it to last 3 seconds in total
# # video = Video("demos/temp/vedovid.mp4", duration=10, backend='ffmpeg')

# nbframes=750
# for i in range(nbframes):
#     scals += 2
#     grid.cmap("jet", scals, vmin=0, vmax=2*nbframes)
#     txt.text(f"frame: {i}/{nbframes}")

#     grid.add_scalarbar(horizontal=True)
#     plt.show(grid, txt, __doc__)        ## Show again for colorbar
    
#     # plt.render()
#     # video.add_frame()

# # video.close()

# plt.interactive()
# plt.close()








# plt = Plotter(N=1, axes=1, interactive=False)
# # mesh = Mesh(grid)
# mesh= grid
# nv = mesh.ncells                           # nr. of cells
# scals = mesh.cell_centers()[:, 0] + 37

# mesh.cmap("afmhot", scals, on='cells')
# mesh.add_scalarbar(horizontal=True)

# plt.show(mesh, "mesh.cmap(on='cells')")

# plt.interactive().close()





# from vedo import Plotter, Grid, Text2D, Points, settings, Mesh, Spline, Video, load
# import numpy as np

# settings.default_font = "Theemim"

# mesh = load("./updec/test/channel.msh")

# plt = Plotter(N=1, axes=1, interactive=False)
# plt.show(mesh, "Loaded msh file")

# plt.interactive().close()



########################## pyvisat

# import pyvista
# from pyvista import examples
# filename = examples.download_human(load=False)
# print(filename.split("/")[-1])

# reader = pyvista.get_reader(filename)
# print(reader)  

# mesh = reader.read()
# print(mesh)

# mesh.plot(color='tan')




# import pyvista
# from pyvista import examples
# filename = "./updec/test/channel.vtk"
# print(filename.split("/")[-1])

# reader = pyvista.get_reader(filename)
# print(reader)  

# mesh = reader.read()
# print(mesh)

# mesh.plot(color='tan')





import numpy as np

import updec.tests.pyvista as pv

filename = "sphere-shrinking.mp4"

mesh = pv.Sphere()
print(mesh.n_points)
mesh.cell_data["data"] = np.random.random(mesh.n_cells)

plotter = pv.Plotter()
# Open a movie file
plotter.open_movie(filename)

# Add initial mesh
plotter.add_mesh(mesh, scalars="data", clim=[0, 1])
# Add outline for shrinking reference
plotter.add_mesh(mesh.outline_corners())

plotter.show(auto_close=False)  # only necessary for an off-screen movie

# Run through each frame
plotter.write_frame()  # write initial data

# Update scalars on each frame
for i in range(100):
    random_points = np.random.random(mesh.points.shape)
    mesh.points = random_points * 0.01 + mesh.points * 0.99
    mesh.points -= mesh.points.mean(0)
    mesh.cell_data["data"] = np.random.random(mesh.n_cells)
    plotter.add_text(f"Iteration: {i}", name='time-label')
    plotter.write_frame()  # Write this frame

# Be sure to close the plotter when finished
plotter.close()