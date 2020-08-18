% =============================================================================
  % SIMULATION-BASED ENGINEERING LAB (SBEL) - http://sbel.wisc.edu
  %
  % Copyright (c) 2019 SBEL
  % All rights reserved.
  %
  % Use of this source code is governed by a BSD-style license that can be found
  % at https://opensource.org/licenses/BSD-3-Clause
  %
  % =============================================================================
  % Contributors: Luning Fang
  % =============================================================================

% 3D rotation of a vector around a given direction with a given angle
function u_prime = rotationAboutAxis(u, n, Xi)
% rotate vector u around vector n by angle of Xi

if abs(n'*n - 1) > 1e-5
    fprintf('direction vector has to be a unit vector\n');
end
% assemble quaternion based on direction and angle
e0 = cos(Xi/2);
e  = n*sin(Xi/2);
p  = [e0; e(1); e(2); e(3)];

if abs(p'*p-1) > 1e-5
    fprintf('quaternion not orthognal\n');
end

% get rotation matrix from p
A = getAfromP(p);
u_prime = A*u;
