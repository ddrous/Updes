
import gmsh
import sys


lc = 0.15
ref = 2           ## Refinement factor to account for Neumann
Lx = 1.5
Ly = 1.0
DIM = 2


gmsh.initialize()

gmsh.model.add("channel")

## Make the square
gmsh.model.geo.addPoint(0, 0, 0, lc/ref, 1)
gmsh.model.geo.addPoint(Lx/10, 0, 0, lc, 2)
gmsh.model.geo.addPoint(0.5, 0, 0, lc/ref, 3)
gmsh.model.geo.addPoint(1, 0, 0, lc/ref, 4)
gmsh.model.geo.addPoint(9*Lx/10, 0, 0, lc, 5)   ## New
gmsh.model.geo.addPoint(Lx, 0, 0, lc/ref, 6)

gmsh.model.geo.addPoint(Lx, Ly, 0, lc/ref, 7)
gmsh.model.geo.addPoint(9*Lx/10, Ly, 0, lc, 8)
gmsh.model.geo.addPoint(1, Ly, 0, lc/ref, 9)
gmsh.model.geo.addPoint(0.5, Ly, 0, lc/ref, 10)
gmsh.model.geo.addPoint(Lx/10, Ly, 0, lc, 11)
gmsh.model.geo.addPoint(0, Ly, 0, lc/ref, 12)


gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 5, 4)
gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 7, 6)
gmsh.model.geo.addLine(7, 8, 7)
gmsh.model.geo.addLine(8, 9, 8)
gmsh.model.geo.addLine(9, 10, 9)
gmsh.model.geo.addLine(10, 11, 10)
gmsh.model.geo.addLine(11, 12, 11)
gmsh.model.geo.addLine(12, 1, 12)
gmsh.model.geo.addCurveLoop([12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 1)


gmsh.model.geo.addPlaneSurface([1], 1)

## Geometry done. Synchornize here
gmsh.model.geo.synchronize()


gmsh.model.addPhysicalGroup(1, [12], name="Inflow")
gmsh.model.addPhysicalGroup(1, [6], name="Outflow")
gmsh.model.addPhysicalGroup(1, [3], name="Blowing")
gmsh.model.addPhysicalGroup(1, [9], name="Suction")
gmsh.model.addPhysicalGroup(1, [1,2,4,5,7,8,10,11], name="Wall")
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
