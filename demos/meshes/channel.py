
import gmsh
import sys
import os

gmsh.initialize()

gmsh.model.add("channel")


lc = 0.2
L = 1.0
DIM = 2

gmsh.model.geo.addPoint(-3*L, -1/2, 0, lc, 1)
gmsh.model.geo.addPoint(8*L, -1/2, 0, lc/2, 2)
gmsh.model.geo.addPoint(8*L, 1/2, 0, lc/2, 3)
gmsh.model.geo.addPoint(-3*L, 1/2, 0, lc, 4)


gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)


gmsh.model.geo.addCurveLoop([4, 1, 2, 3], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

## Geometry done. Synchornize here
gmsh.model.geo.synchronize()


gmsh.model.addPhysicalGroup(1, [4], name="Inflow")
gmsh.model.addPhysicalGroup(1, [2], name="Outflow")
gmsh.model.addPhysicalGroup(1, [1,3], name="Wall")
gmsh.model.addPhysicalGroup(2, [1], name="Fluid")


## Meshing paremeters for quadrangles
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.Algorithm", 11)
gmsh.option.setNumber("Mesh.MshFileVersion", 4.0)

## Visualisation parameters
gmsh.option.setNumber("Mesh.Points", 1)
gmsh.option.setNumber("Mesh.SurfaceEdges", 0)


## Mesh generation process
gmsh.model.mesh.generate(DIM)


#Save the mesh to disk
corename = os.path.realpath(__file__)[:-2]
gmsh.write(corename+"msh")

if '--nopopup' not in sys.argv:
    gmsh.fltk.run()


gmsh.finalize()