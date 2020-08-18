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

function arc = generateArc(arcLength, arcRadius, arcCenter)

PI = 3.1415;
circleLength = 2*PI*arcRadius;
arcAngle = abs(arcLength)/circleLength*2*PI;

arcBeginAngle = 0.75*PI - arcAngle/2;
arcEndAngle   = 0.75*PI + arcAngle/2;

arcPoints = 20;
arcSpace = arcAngle/(arcPoints+1);
arcAngleArray = arcBeginAngle:arcSpace:arcEndAngle;
arcX = zeros(length(arcAngleArray),1);
arcY = zeros(length(arcAngleArray),1);
arcZ = zeros(length(arcAngleArray),1);

for kk = 1:length(arcAngleArray)
    theta = arcAngleArray(kk);
    
    arcY(kk) = arcCenter(2) + arcRadius * cos(theta);
    arcZ(kk) = arcCenter(3) + arcRadius * sin(theta);
    
    
    
end

arc.Xdata = arcX;
arc.Ydata = arcY;
arc.Zdata = arcZ;
