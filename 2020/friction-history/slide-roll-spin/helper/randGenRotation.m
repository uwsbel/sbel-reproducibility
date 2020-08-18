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

% generate arbitrary 3-dimenstional rotation matrix 
function A = randGenRotation(dim)
if dim == 3
    
    % generate random rotation angle
    phi = -pi+2*pi*rand(1);
    theta = -pi+2*pi*rand(1);
    psi = -pi+2*pi*rand(1);
    
    A1 = [cos(phi) -sin(phi) 0; sin(phi) cos(phi) 0; 0 0 1];
    A2 = [1 0 0; 0 cos(theta) -sin(theta); 0 sin(theta) cos(theta)];
    A3 = [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1];
    
    A = A1*A2*A3;
    
end