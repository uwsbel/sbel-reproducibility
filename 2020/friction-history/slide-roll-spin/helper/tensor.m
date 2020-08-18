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

function E = tensor(e)

if length(e) ~= 3
    fprintf('vector e size should be 3\n');
end

E = [0 -e(3) e(2); e(3) 0 -e(1); -e(2) e(1) 0];
