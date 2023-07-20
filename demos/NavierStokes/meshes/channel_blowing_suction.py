
import gmsh
import sys


# lc = 0.1        ## TODO Set this to 4 !
# ref_io = 7           ## Refinement factor to account for Infow/Outflow
# ref_bs = 4           ## Refinement factor to account for Blowing/Suction
lc = 0.36
ref_io = 8
ref_bs = 6

box_half_length = 0.001
Lx = 1.5
Ly = 1.0
DIM = 2


gmsh.initialize()

gmsh.model.add("channel")


## Make the square
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(Lx/10, 0, 0, lc, 2)
gmsh.model.geo.addPoint(0.5, 0, 0, lc, 3)
gmsh.model.geo.addPoint(1, 0, 0, lc, 4)
gmsh.model.geo.addPoint(9*Lx/10, 0, 0, lc, 5)   ## New
gmsh.model.geo.addPoint(Lx, 0, 0, lc, 6)

gmsh.model.geo.addPoint(Lx, Ly, 0, lc, 7)
gmsh.model.geo.addPoint(9*Lx/10, Ly, 0, lc, 8)
gmsh.model.geo.addPoint(1, Ly, 0, lc, 9)
gmsh.model.geo.addPoint(0.5, Ly, 0, lc, 10)
gmsh.model.geo.addPoint(Lx/10, Ly, 0, lc, 11)
gmsh.model.geo.addPoint(0, Ly, 0, lc, 12)


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


#### Add Boxes for local refinements
gmsh.model.mesh.field.add("Box", 6)
gmsh.model.mesh.field.setNumber(6, "VIn", lc / ref_bs)
gmsh.model.mesh.field.setNumber(6, "VOut", lc)
gmsh.model.mesh.field.setNumber(6, "XMin", 0.5-box_half_length)
gmsh.model.mesh.field.setNumber(6, "XMax", 0.5+box_half_length)
gmsh.model.mesh.field.setNumber(6, "YMin", 0.0)
gmsh.model.mesh.field.setNumber(6, "YMax", 0.0+2*box_half_length)
gmsh.model.mesh.field.setNumber(6, "Thickness", 0.3)

gmsh.model.mesh.field.add("Box", 7)
gmsh.model.mesh.field.setNumber(7, "VIn", lc / ref_bs)
gmsh.model.mesh.field.setNumber(7, "VOut", lc)
gmsh.model.mesh.field.setNumber(7, "XMin", 1.0-box_half_length)
gmsh.model.mesh.field.setNumber(7, "XMax", 1.0+box_half_length)
gmsh.model.mesh.field.setNumber(7, "YMin", 0.0)
gmsh.model.mesh.field.setNumber(7, "YMax", 0.0+2*box_half_length)
gmsh.model.mesh.field.setNumber(7, "Thickness", 0.3)

gmsh.model.mesh.field.add("Box", 8)
gmsh.model.mesh.field.setNumber(8, "VIn", lc / ref_bs)
gmsh.model.mesh.field.setNumber(8, "VOut", lc)
gmsh.model.mesh.field.setNumber(8, "XMin", 1.0-box_half_length)
gmsh.model.mesh.field.setNumber(8, "XMax", 1.0+box_half_length)
gmsh.model.mesh.field.setNumber(8, "YMin", Ly-2*box_half_length)
gmsh.model.mesh.field.setNumber(8, "YMax", Ly)
gmsh.model.mesh.field.setNumber(8, "Thickness", 0.3)


gmsh.model.mesh.field.add("Box", 9)
gmsh.model.mesh.field.setNumber(9, "VIn", lc / ref_bs)
gmsh.model.mesh.field.setNumber(9, "VOut", lc)
gmsh.model.mesh.field.setNumber(9, "XMin", 0.5-box_half_length)
gmsh.model.mesh.field.setNumber(9, "XMax", 0.5+box_half_length)
gmsh.model.mesh.field.setNumber(9, "YMin", Ly-2*box_half_length)
gmsh.model.mesh.field.setNumber(9, "YMax", Ly)
gmsh.model.mesh.field.setNumber(9, "Thickness", 0.3)


gmsh.model.mesh.field.add("Box", 10)
gmsh.model.mesh.field.setNumber(10, "VIn", lc / ref_io)
gmsh.model.mesh.field.setNumber(10, "VOut", lc)
gmsh.model.mesh.field.setNumber(10, "XMin", 0.0)
gmsh.model.mesh.field.setNumber(10, "XMax", 0.1*Lx/25)
gmsh.model.mesh.field.setNumber(10, "YMin", 0)
gmsh.model.mesh.field.setNumber(10, "YMax", Ly)
gmsh.model.mesh.field.setNumber(10, "Thickness", 0.3)


gmsh.model.mesh.field.add("Box", 11)
gmsh.model.mesh.field.setNumber(11, "VIn", lc / ref_io)
gmsh.model.mesh.field.setNumber(11, "VOut", lc)
gmsh.model.mesh.field.setNumber(11, "XMin", Lx-0.1*Lx/25)
gmsh.model.mesh.field.setNumber(11, "XMax", Lx)
gmsh.model.mesh.field.setNumber(11, "YMin", 0)
gmsh.model.mesh.field.setNumber(11, "YMax", Ly)
gmsh.model.mesh.field.setNumber(11, "Thickness", 0.3)

# gmsh.model.mesh.field.setAsBackgroundMesh(9)


# Let's use the minimum of all the Box fields as the mesh size field: #TODO Just a combination will do
gmsh.model.mesh.field.add("Min", 12)
gmsh.model.mesh.field.setNumbers(12, "FieldsList", [6, 7, 8, 9, 10, 11])
gmsh.model.mesh.field.setAsBackgroundMesh(12)


gmsh.model.addPhysicalGroup(1, [12], name="Inflow")
gmsh.model.addPhysicalGroup(1, [6], name="Outflow")
gmsh.model.addPhysicalGroup(1, [3], name="Blowing")
gmsh.model.addPhysicalGroup(1, [9], name="Suction")
gmsh.model.addPhysicalGroup(1, [1,2,4,5,7,8,10,11], name="Wall")
gmsh.model.addPhysicalGroup(2, [1], name="Fluid")


## Meshing paremeters for quadrangles
# gmsh.option.setNumber("Mesh.RecombineAll", 1)
# gmsh.option.setNumber("Mesh.Algorithm", 11)
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
