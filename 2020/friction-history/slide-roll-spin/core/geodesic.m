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
  
function s_proj = geodesic(contactPointOld, contactPointNew, contactNormal, sphereCenter, useGeodesic)

O = sphereCenter;
A = contactPointOld;
B = contactPointNew;
AB = B - A;
OB = B - O;
OA = A - O;


if norm(AB) < 1e-8
    
    angle_AOB = asin(norm(AB)/norm(OA));
else
    angle_AOB = acos(OA'*OB/sqrt(sum(OA.^2)*sum(OB.^2)));

end

%angle_AOB = acos(sqrt((OA'*OB)^2/sum(OA.^2)/sum(OB.^2)))

if useGeodesic == true
    AB_arc_dim = norm(OA)*angle_AOB;
else
    AB_arc_dim = norm(A-B);
end

n = contactNormal;
proj_AB_n = AB - AB'*n/norm(n)^2*n;

if norm(proj_AB_n) == 0
    s_proj = [0;0;0];
else
    dir_s_proj = proj_AB_n/norm(proj_AB_n);

    s_proj = dir_s_proj * AB_arc_dim;
end

