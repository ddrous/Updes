Mesh.MshFileVersion = 4.0;

h = 0.3;
L = 1.0;

Point(1) = {-3*L, -1/2, 0, h};
Point(2) = {8*L, -1/2, 0, h/3};
Point(3) = {8*L, 1/2, 0, h/3};
Point(4) = {-3*L, 1/2, 0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(11) = {4, 1, 2, 3};
Plane Surface(11) = {11};
Physical Line("Inflow", 2) = {4};
Physical Line("Outflow", 3) = {2};
Physical Line("Wall", 4) = {1,3};

Coherence;
Physical Surface("Domain") = {11};