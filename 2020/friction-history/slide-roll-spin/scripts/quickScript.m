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

%load('/Users/lulu/Documents/Research/code/friction3DSimEngine/data/sphereUpIncline_variousEtaAndAngle_longerTime.mat');

TOL = 1e-2;

numEtaArray = length(eta_array);
numSlopeAngleArray = length(slopeAngleDeg_array);

plot_val = state_val;


for ii = 1:numEtaArray
    for jj = 1:numSlopeAngleArray
        if abs(state_val(ii,jj) - 0.2) < TOL  %% pure rolling
            plot_val(ii,jj) = 1;
        end
    end
end


for ii = 1:numEtaArray
    for jj = 1:numSlopeAngleArray
        if state_val(ii,jj) == 100  %% pure sliding
            plot_val(ii,jj) = 3;
        end
    end
end

for ii = 1:numEtaArray
    for jj = 1:numSlopeAngleArray
        if plot_val(ii,jj) ~= 0 && plot_val(ii,jj) ~= 1 && plot_val(ii,jj) ~= 3  %% pure sliding
            plot_val(ii,jj) = 2;
        end
    end
end

FS = 45;

cfh = figure(1)
cfh.Units = 'normalized';
cfh.Position = [0 0 1 1];


g = surf(slopeAngleDeg_array, eta_array, plot_val);
g.EdgeAlpha = 0;
view(0, 90);

xlim([1,30]);
ylim([0.2,0.6]);
cf = gca;
cf.FontSize = FS;
cf.LineWidth = 5;


xlabel('slope angle (degree)');
ylabel('$$\eta_r$$', 'Interpreter', 'latex');
title('steady state of a sphere rolling up an incline');


% box1 = text(2.5, 0.4, 0, 'stationary');
% box1.FontSize = FS+10;
% box1.LineStyle = 'none';
% 
% box2 = text(15, 0.4, 1, sprintf('\t \t \t \t   pure rolling \n (rolling without slipping)'));
% box2.LineStyle = 'none';
% box2.FontSize = FS+10;
% 
% box3 = text(28, 0.4, 3, sprintf('rolling with \n  \t slipping'));
% box3.LineStyle = 'none';
% box3.FontSize = FS+10;
% 
% box4 = text(20, 0.6, 4, sprintf('pure sliding'));
% box4.LineStyle = 'none';
% box4.FontSize = FS+10;
