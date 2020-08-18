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

addpath('/Users/lulu/Documents/Research/code/friction3DSimEngine/core')
addpath('/Users/lulu/Documents/Research/code/friction3DSimEngine/demos/')


% create a plane using normal direction and offset
groundNormal = [0;0;1];

numArray = [0.96306, 0.96307, 0.96308];

figHdl = figure;
figHdl.Position = [1 42 1280 663];
%subplot(1,2,1)
hold on

LW = 1;  % line width
FS = 20; % font size

omic_end_array = zeros(size(numArray));
omic_holder_cmp = [];

for index = 1:length(numArray)
    num = numArray(index);
    u1 = [ num; sqrt(1-num^2); 0];
    w1 = [ -sqrt(1-num^2);  num; 0];
    
    initialOrientation = contactFrame(u1, w1, groundNormal);
    demo_ellipsoid_smallAngle;
    fprintf('omic end');
    omic_holer(end);
    
    myPlot = makePlot(t, omic_holer, 'time (sec)', '\omega (rad/s)', '', LW, FS);
    omic_end_array(index) = omic_holer(end);
    

%    clearvars -except groundNormal numArray omic_end_array LW FS omic_holder_cmp
    
    
end

legend('0.96306', '0.96397', '0.96308');
legend(sprintf('u_x = %g, \\omega_{ss} = %g rad/s', numArray(1),  omic_end_array(1)), ...
       sprintf('u_x = %g, \\omega_{ss} = %g rad/s', numArray(2),  omic_end_array(2)), ...
       sprintf('u_x = %g, \\omega_{ss} = %g rad/s', numArray(3),  omic_end_array(3)));
%%
% subplot(1,2,2)
% makePlot(t, (omic_holder_cmp(:,1)-omic_holder_cmp(:,2))*10^3, 'time (sec)', 'diff \omega \times 10^{-3} (rad/s)', '', LW, FS);
