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
clear all
close all


if strcmp(computer, 'MACI64') == true
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/core');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/demos');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/post_processing');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/helper');
end

% if isunix
%     addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/core');
%     addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/demos');
%     addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/post_processing');
% end

%Ke_array = [5e5, 1e6, 5e6, 1e7];
Ke_array = [5e6, 1e7, 5e7, 1e8];

%figHdl = figure;
%figHdl.Position = [1 42 1280 663];
subplot(1,2,1);
hold on
subplot(1,2,2)
hold on

LW = 4;  % line width
FS = 48; % font size


for index = 1:length(Ke_array)
    Ke = Ke_array(index);
    fprintf('Ke=%f\n', Ke);
    demo_sphere_spinning;
    
    %     if index == 3 || index == 4
    %         myPlot = makePlot(t, omic_holder, 'time (sec)', '\omega (rad/s)', '', LW, FS);
    %         zoomOmic = [zoomOmic omic_holder];
    %         zoomColor = [zoomColor; myPlot.Color];
    %         hold on
    %     else
    subplot(1,2,1)
    makePlot(t, Psi, 'time (sec)', 'spinning history \Psi_{ij} (rad)', sprintf('steel ellipsoid, flat', radius), LW, FS);
    hold on
    subplot(1,2,2)
    makePlot(t, omic_holder, 'time (sec)', 'angular velocity (rad/s)', '', LW, FS);
    hold on
    %     xlabel('x')
    %     ylabel('y')
    %     zlabel('z')
    %     xlim([-1,1]);
    %     ylim([-1,1]);
    %     axis equal
    %   clearvars -except eta_spin_array omic_end_array LW FS zoomOmic zoomColor
    
end

subplot(1,2,2)
lgd = legend(sprintf('K_E =%.0e N/m', Ke_array(1)), ...
    sprintf('K_E =%.0e N/m', Ke_array(2)), ...
    sprintf('K_E =%.0e N/m', Ke_array(3)), ...
    sprintf('K_E =%.0e N/m', Ke_array(4)));
lgd.FontSize = 40;