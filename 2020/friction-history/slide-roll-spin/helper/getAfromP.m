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

function A = getAfromP(p)

if length(p) ~= 4
    %fprintf('length of euler parameter should be 4\n');
end

if abs(p*p'-1) > 1e-5
    %fprintf('quaternion not orthognal, pp=%g\n', p*p');
end

e0 = p(1);
e = p(2:4);
e1 = e(1);
e2 = e(2);
e3 = e(3);
%A = (2*e0^2-1)*eye(3) + 2*(e*e' + e0*tensor(e));

A = 2*[e0^2+e1^2-0.5 e1*e2-e0*e3    e1*e3+e0*e2; ...
       e1*e2+e0*e3   e0^2+e2^2-0.5  e2*e3-e0*e1; ...
       e1*e3-e0*e2   e2*e3+e0*e1    e0^2+e3^2-0.5];

checkEye = norm(A'*A - eye(3));
if abs(checkEye)>1e-3
   % fprintf('orientation matrix not orthonormal, res = %g\n', checkEye);
end