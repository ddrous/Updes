import jax.numpy as jnp

######
""" Turn this into a submodule for visualisation """
######


def pyvista_animation(folderpath, fieldnames, duration=5):  ## TODO use duration
    """ Make a PyVista animation from save it to video """


    import pyvista as pv
    pv.global_theme.cmap = 'jet'


    fieldpath = folderpath  + fieldnames + ".npz"    ## TODO Make fieldnames a list / assumption that saved in npz

    loaded_arrays = jnp.load(fieldpath)
    arraynames = loaded_arrays.files
    renumb_map = loaded_arrays[arraynames[0]]
    field = loaded_arrays[arraynames[1]]

    videoname = fieldpath.rsplit(".", maxsplit=1)[0] + ".mp4"
    meshname = folderpath + "mesh.vtk"

    reader = pv.get_reader(meshname)
    mesh = reader.read()

    mesh.point_data["data"] = field[0]  ## Just create the data field
    mesh.point_data["data"][renumb_map] = field[0]

    plt = pv.Plotter()
    # Open a movie file
    plt.open_movie(videoname)

    # Add initial mesh
    plt.add_mesh(mesh, scalars="data", clim=[0, 1])     ##TODO colorbar

    plt.view_xy()
    plt.show(auto_close=False)  # only necessary for an off-screen movie

    # Run through each frame
    plt.write_frame()  # write initial data


    # Update scalars on each frame
    for i in range(field.shape[0]):
        ### Make sure field[i] is properly orderd first
        mesh.point_data["data"][renumb_map] = field[i]
        plt.add_text(f"Iteration: {i}", name='time-label')
        plt.write_frame()  # Write this frame

    # Be sure to close the plotter when finished
    plt.close()
