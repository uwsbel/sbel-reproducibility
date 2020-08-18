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

function p = getPfromA(A)

e0 = sqrt((trace(A)+1)/4);

if abs(e0) > 1e-10
    
    e1 = (A(3,2) - A(2,3))/(4*e0);
    e2 = (A(1,3) - A(3,1))/(4*e0);
    e3 = (A(2,1) - A(1,2))/(4*e0);
    
else  % e0 = 0, p = [0; e]
    
    a11 = A(1,1);
    a22 = A(2,2);
    a33 = A(3,3);
    
    
    
    e1 = sqrt((a11+1)/2);
    e2 = sqrt((a22+1)/2);
    e3 = sqrt((a33+1)/2);
    
    
end

p = [e0;e1;e2;e3];