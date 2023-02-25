import sys
import jax.numpy as jnp
from vedo import Plotter, Grid, Text2D


def vedo_animation(filename, duration=5):

    loaded_arrays = jnp.load(filename)
    filenames = loaded_arrays.files
    coords = loaded_arrays[filenames[0]]

    field = []
    for name in filenames[1:]:
        field.append(loaded_arrays[name])
    field = jnp.stack(field, axis=0)


    grid = Grid(s=(coords[:,0], coords[:,1]), c="red5", alpha=0.5).lw(0)    ## Linewidth to 0 for interpolation
    txt = Text2D(font='Brachium', pos='bottom-left', bg='yellow5')

    plt = Plotter(N=1, axes=1, interactive=False)
    plt.show(grid, txt, __doc__)

    nbframes = len(filenames) - 1
    minval, maxval = jnp.min(field), jnp.max(field)
    print(grid.points().shape, field[0].shape)              ## TODO: Study the grid function more

    for i in range(nbframes):
        grid.cmap("hot", field[i], vmin=minval, vmax=maxval)
        txt.text(f"frame: {i+1}/{nbframes}")

        grid.add_scalarbar(horizontal=True)
        plt.show(grid, txt, __doc__)        ## Show again for colorbar

        plt.render()

    plt.interactive().close()



if __name__ == '__main__':
    vedo_animation(sys.argv[1])     ## ./demos/temp/79049/u.npz
