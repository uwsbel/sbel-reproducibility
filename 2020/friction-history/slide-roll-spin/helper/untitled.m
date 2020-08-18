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


mass = 5;
radius = 0.2;
initialPos = [0.1;0.2;0.3];
initialVelo = [1;2;3];
initialOmic = [2;2;2];

phi = 0.5;
n1 =  0.1;
n2 =  0.2;
n3 =  sqrt(1-n1^2-n2^2);
e0 = cos(phi/2);
e  = sin(phi/2) * [n1;n2;n3];
initialQtn = [e0;e];


dt = 0.001;
acc = [0.01;0.02;0.03];
omicdot = [0.04;0.05;0.06];

mySphere = sphereClass(radius, mass);
mySphere.position = initialPos;
mySphere.velo = initialVelo;
mySphere.omic = initialOmic;
mySphere.eulerParameter = initialQtn;
mySphere.orientation = getAfromP(initialQtn);


% mySphere.updateKinematics(acc, omic_dot, dt);
% fprintf("after update\n");
% fprintf("quaternion\n")
% mySphere.eulerParameter
% fprintf("u = [%f, %f, %f]\n", mySphere.orientation(1,1), mySphere.orientation(2,1), mySphere.orientation(3,1));
% fprintf("w = [%f, %f, %f]\n", mySphere.orientation(1,2), mySphere.orientation(2,2), mySphere.orientation(3,2));
% fprintf("n = [%f, %f, %f]\n", mySphere.orientation(1,3), mySphere.orientation(2,3), mySphere.orientation(3,3));

% force  = [10;15;20];
% torque = [5;10;15];
% mySphere.updateKinematicFromForce(force, torque, dt);
% fprintf("after update\n");
% fprintf("quaternion\n")
% mySphere.eulerParameter
% fprintf("u = [%f, %f, %f]\n", mySphere.orientation(1,1), mySphere.orientation(2,1), mySphere.orientation(3,1));
% fprintf("w = [%f, %f, %f]\n", mySphere.orientation(1,2), mySphere.orientation(2,2), mySphere.orientation(3,2));
% fprintf("n = [%f, %f, %f]\n", mySphere.orientation(1,3), mySphere.orientation(2,3), mySphere.orientation(3,3));

testCF = contactFrame([1 4 7; 2 5 8; 3 6 9]);
CF_Global = testCF.expressContactFrameInGlobalRF(mySphere.orientation);
CF_Global.printOut;