import jax.numpy as jnp

######
""" Turn this into a submodule for visualisation """
######


def pyvista_animation(folderpath, fieldname, duration=10, vmin=None, vmax=None):
    ## TODO use duration
    ## TODO Make another function for many different fields
    """ Make a PyVista animation from save it to video """


    import pyvista as pv
    pv.start_xvfb()    ## To avoid seg fault in X-server
    pv.global_theme.cmap = 'jet'


    fieldpath = folderpath  + fieldname + ".npz"    ## TODO Make fieldnames a list / assumption that saved in npz

    loaded_arrays = jnp.load(fieldpath)
    arraynames = loaded_arrays.files
    renumb_map = loaded_arrays[arraynames[0]]
    field = loaded_arrays[arraynames[1]]

    videoname = fieldpath.rsplit(".", maxsplit=1)[0] + ".mp4"
    meshname = folderpath + "mesh.vtk"

    reader = pv.get_reader(meshname)
    mesh = reader.read()

    mesh.point_data[fieldname] = field[0]  ## Just create the data field
    mesh.point_data[fieldname][renumb_map] = field[0]

    plt = pv.Plotter(notebook=True)
    # Open a movie file
    nbframes = field.shape[0]
    plt.open_movie(videoname, framerate=nbframes/duration)

    # Add initial mesh
    if vmin==None:
        vmin = jnp.min(field)
    if vmax==None:
        vmax = jnp.max(field)
    plt.add_mesh(mesh, scalars=fieldname, clim=[vmin, vmax])

    plt.view_xy()
    plt.show(auto_close=False, jupyter_backend='ipyvtklink')  # only necessary for an off-screen movie

    # Run through each frame
    plt.write_frame()  # write initial data


    # Update scalars on each frame
    for i in range(nbframes):
        ### Make sure field[i] is properly orderd first
        mesh.point_data[fieldname][renumb_map] = field[i]
        plt.add_text(f"Frame: {i+1} / {nbframes}", name='time-label')
        plt.write_frame()  # Write this frame

    # Be sure to close the plotter when finished
    plt.close()
