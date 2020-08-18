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

% get the rotation matrix R that rotates unit vector a onto unit vector b
% b = R*a
% gives error when a or b is not unit vector
function R = getRotationMatrixFromAtoB(a, b)

if abs(norm(a) - 1) > 1e-12
    error('a should be unit vector');
end

if abs(norm(b) - 1) > 1e-12
    error('b should be unit vector');
end

% a same direction as b
if norm(a-b) < 1e-13
    R = eye(3);
    return;
end

% a opposite direction as b
if norm(a+b) < 1e-13
    R = -eye(3);
    return;
end


% none above
v = cross(a,b);
s = norm(v);
c = a'*b;

V_skewSym = [0   -v(3) v(2); ...
             v(3) 0   -v(1); ...
            -v(2) v(1) 0];
        
R = eye(3) + V_skewSym + V_skewSym*V_skewSym*(1-c)/s^2;

end