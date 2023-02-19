// Gmsh project created on Sat Jan 28 2023

h=0.25;
Lx=1.5;
Ly=1;

Point(1) = {0, 0, 0, h};
Point(2) = {Lx/3, 0, 0, h/3};
Point(3) = {2*Lx/3, 0, 0, h/3};
Point(4) = {Lx, 0, 0, h};

Point(5) = {Lx, Ly, 0, h};
Point(6) = {2*Lx/3, Ly, 0, h/3};
Point(7) = {Lx/3, Ly, 0, h/3};
Point(8) = {0, Ly, 0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};

Line(4) = {4, 5};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};

Line(8) = {8, 1};

Line Loop(4) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(5) = {4};

Physical Line("wall") = {1,3,5,7};
Physical Line("blowing") = {2};
Physical Line("outlet") = {4};
Physical Line("suction") = {6};
Physical Line("inlet") = {8};

Physical Surface("omega") = {5};
