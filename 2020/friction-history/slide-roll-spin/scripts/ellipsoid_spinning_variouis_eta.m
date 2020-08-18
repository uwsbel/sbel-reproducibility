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

if ismac
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/core');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/demos');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/post_processing');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/helper');
end

if isunix
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/core');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/demos');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/post_processing');
end



eta_spin_array = [0.12 0.1 0.08 0.06]*0.1;
omic_end_array = zeros(size(eta_spin_array));
figHdl = figure;
figHdl.Position = [1 42 1280 663];
subplot(1,2,1);
hold on
subplot(1,2,2)

hold on

LW = 4;  % line width
FS = 36; % font size



for index = 1:length(eta_spin_array)
    eta_spin = eta_spin_array(index);
    fprintf('eta=%f\n', eta_spin);
    demo_ellipsoid_easy_spin;
%    demo_elipsoid_spinning    
    omic_end_array(index) = omic_holer(end);
    subplot(1,2,1)
    myPlot = makePlot(t, omic_holer, 'time (sec)', '\omega (rad/s)', '', LW, FS);
    subplot(1,2,2)
    myContact.CF_prev_global.drawContactFrameWithLabels([0;0;0], myPlot.Color, ...
        1, 'u', 'w', '');
    hold on
    view(0,90);
    xlabel('x')
    ylabel('y')
    zlabel('z')
    xlim([-1,1]);
    ylim([-1,1]);
    axis equal
    clearvars -except eta_spin_array omic_end_array LW FS
    
end

subplot(1,2,1)
lgd = legend(sprintf('\\eta_\\psi=%g, \\omega_{ss} = %.1e rad/s', eta_spin_array(1), omic_end_array(1)), ...
    sprintf('\\eta_\\psi=%g, \\omega_{ss} = %.1e rad/s', eta_spin_array(2), omic_end_array(2)), ...
    sprintf('\\eta_\\psi=%g, \\omega_{ss} = %.1e rad/s', eta_spin_array(3), omic_end_array(3)), ...
    sprintf('\\eta_\\psi=%g, \\omega_{ss} = %.1e rad/s', eta_spin_array(4), omic_end_array(4)));
subplot(1,2,2)
title('contact frame at the end of simulation', 'FontSize', FS)
currentPlot = gca;
currentPlot.FontSize = FS;
currentPlot.LineWidth = LW;
axis equal

%%
subplot(1,2,2)
xlim([-0.2,1])
ylim([-0.2,1])
hold on
initialFrame = contactFrame;
num = 1;
initialFrame.u = [ num,           sqrt(1-num^2), 0];
initialFrame.w = [-sqrt(1-num^2), num,           0];
initialFrame.drawContactFrameWithLabels([0;0;0], 'black', 1, 'initial contact', '', '')

