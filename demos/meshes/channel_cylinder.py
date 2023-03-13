
import gmsh
import sys


lc = 0.3
nm_factor = 4           ## Refinement factor to account for Neumann
cy_factor = 10         ## Refinement factor to account for cylinder
L = 1.0
DIM = 2


gmsh.initialize()

gmsh.model.add("channel")

## Mak e the square
gmsh.model.geo.addPoint(-3*L, -1/2, 0, lc, 1)
gmsh.model.geo.addPoint(8*L, -1/2, 0, lc/nm_factor, 2)
gmsh.model.geo.addPoint(8*L, 1/2, 0, lc/nm_factor, 3)
gmsh.model.geo.addPoint(-3*L, 1/2, 0, lc, 4)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

gmsh.model.geo.addCurveLoop([4, 1, 2, 3], 1)

## Ad the hole
gmsh.model.geo.addPoint(-L/10, 0, 0, lc/cy_factor, 5)
gmsh.model.geo.addPoint(0, 0, 0, lc/cy_factor, 6)
gmsh.model.geo.addPoint(L/10, 0, 0, lc/cy_factor, 7)
gmsh.model.geo.addCircleArc(5, 6, 7, 5)
gmsh.model.geo.addCircleArc(7, 6, 5, 6)
gmsh.model.geo.addCurveLoop([5, 6], 2)

gmsh.model.geo.addPlaneSurface([1, 2], 1)

## Geometry done. Synchornize here
gmsh.model.geo.synchronize()


gmsh.model.addPhysicalGroup(1, [4], name="Inflow")
gmsh.model.addPhysicalGroup(1, [2], name="Outflow")
gmsh.model.addPhysicalGroup(1, [1,3], name="Wall")
gmsh.model.addPhysicalGroup(1, [5,6], name="Cylinder")
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
if len(sys.argv) >= 2: ## User defined a location for save
    gmsh.write(sys.argv[1]+"mesh.msh")
    gmsh.write(sys.argv[1]+"mesh.vtk")      ## For visualisation with PyVista


if '--nopopup' not in sys.argv:
    gmsh.fltk.run()


gmsh.finalize()
