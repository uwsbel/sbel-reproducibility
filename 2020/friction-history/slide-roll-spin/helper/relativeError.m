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

function [rErrorMin, rErrorAvg, rErrorMax] = relativeError(approx, original)

num = length(approx);
relativeErrorArray = [];


for ii = 1:num
    
    if original(ii) ~= 0
        
        relativeErrorArray(ii) = (approx(ii) - original(ii))/original(ii);
    else
            relativeErrorArray(ii) = 0;

        
    end
    
end

rErrorMin = min(abs(relativeErrorArray));
rErrorAvg = sum(abs(relativeErrorArray))/length(relativeErrorArray);
rErrorMax = max(abs(relativeErrorArray));
