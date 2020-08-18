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

epsilon = 0.00000000000000000000000001;
OA = [0; 0; 1];
OB = [0; 0; 1 - epsilon];
OB(2) = sqrt(1 - OB(3)^2);


OA_length = norm(OA);
OB_length = norm(OB);

ratio = OA'*OB/(OA_length*OB_length);
asin(ratio)
acos(ratio)