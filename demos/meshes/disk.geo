// Gmsh project created on Sat Apr 04 14:33:34 2020
h = 0.2;
//+
Point(1) = {0, 0, 0, h};
//+
Point(2) = {-1, 0, 0, h};
//+
Point(3) = {1, 0, 0, h};

//+
Circle(1) = {3, 1, 2};
//+
Circle(2) = {2, 1, 3};
//+
Curve Loop(1) = {1, 2};
//+
Plane Surface(1) = {1};
//+
Physical Curve("Dirichlet") = {1, 2};
//+
Physical Surface("Omega") = {1};
