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

% script for geodesic, after each run find the maximum magnitude and print
% out
clc
close all
clear all

scale = 1;
fprintf('scale=%d\n', scale);

Tend = 1.2 * scale;
geodesicSphereRolling;
postProcessing;