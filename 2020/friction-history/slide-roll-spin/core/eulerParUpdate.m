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
  
function p_new = eulerParUpdate(omic, p_old, dt)

p = p_old + 0.5*getE(p_old)'*omic*dt;


p_prev = p;

p_abs = abs(p);

p(p_abs==max(p_abs)) = sqrt((1 - sum(p(p_abs~=max(p_abs)).^2))/sum(p_abs==max(p_abs)))*sign(p(p_abs==max(p_abs)));

p_new = p;