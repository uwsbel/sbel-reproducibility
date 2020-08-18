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

% script disk down slope of various size
clear all
clc
close all
FontSize = 50;
LineWidth = 5;
MarkerSize = 10;

if strcmp(computer, 'MACI64')
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/core/');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/helper/');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/post_processing/');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/demos/');
    
    
else
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/core');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/helper');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/post_processing');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/demos/');
    
end

figure('units','normalized','outerposition',[0 0 1 1]);
hold on
eta_array = 0.2:0.005:0.6;
slopeAngleDeg_array = 1:0.1:30;

% eta_array = 0.2:0.1:0.6;
% slopeAngleDeg_array = 1:2:30;

state_val = zeros(length(eta_array), length(slopeAngleDeg_array));

TOL = 1e-10;
%for ii = 17:25
tic
for ii = 1:length(eta_array)
    eta = eta_array(ii);
    %    Ratio = [];
    %    Slope_angle = [];
    %    for jj = 34:38
    for jj = 1:length(slopeAngleDeg_array)
        slope_angle_deg = slopeAngleDeg_array(jj);
        
        %        Kr = 4*eta * 0.2^2 * Ke;
        
        demo_sphere_incline_up
        sliding_mode = myFrictionModel.slidingFr.mode;
        rolling_mode = myFrictionModel.rollingTr.mode;
        
        if sliding_mode == 's' && rolling_mode == 's'  %% stationary
            state_val(ii,jj) = 0;
        end
        
        if sliding_mode == 's' && rolling_mode == 'k'  %% pure rolling
            state_val(ii,jj) = 1;
        end
        
        if sliding_mode == 'k' && rolling_mode == 'k'  %% rolling with slipping
            state_val(ii,jj) = 2;
        end
        
        if sliding_mode == 'k' && rolling_mode == 's'  % pure sliding
            state_val(ii,jj) = 3;
        end
        
        fprintf('eta=%f, slope=%f, v_end=%g, omic_end=%g, ratio=%g, z=%f\n', eta, slope_angle_deg, norm(mySphere.velo), norm(mySphere.omic), state_val(ii,jj), mySphere.position(3));
        
    end
    
    %
    %     plot(Slope_angle, Ratio, '*', 'MarkerSize', MarkerSize, 'LineWidth', LineWidth);
    %     xlabel('incline angle (degree)', 'FontSize', FontSize);
    %     ylabel('$$\lim_{t\to \infty}{{x}}/{{\theta}}$$', 'FontSize', FontSize, 'Interpreter', 'latex');
    %     grid on
    %     set(gca, 'linewidth', LineWidth);
    %     a = get(gca, 'XTick');
    %     set(gca, 'FontSize', FontSize)
    %     xlim([1, 36]);
    %title('disk-spring-damper system', 'FontSize', FontSize)
end


FS = 45;


cfh = figure(1);
cfh.Units = 'normalized';
cfh.Position = [0 0 1 1];
%%

g = surf(slopeAngleDeg_array, eta_array, state_val);
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

% draw critical state line between pure rolling and stationary
eta_cr_array = eta_array;
alpha_cr_array = atan(3.5*mu_s - 5*eta_cr_array*mu_k)/3.1415926*180;

z_array = 5*ones(size(eta_cr_array));
figure(1);
hold on

plot3(alpha_cr_array, eta_cr_array, z_array, 'LineWidth', 5, 'Color', 'black')


% draw critical state line between pure rolling and rolling w/ slipping
eta_intercept = mu_s/2/mu_k;

eta_cr_array2 = eta_array(eta_array<eta_intercept);
alpha_cr_array2 = atan(2*eta_cr_array2*mu_k)/3.1415*180;
plot3(alpha_cr_array2, eta_cr_array2, 5*ones(size(eta_cr_array2)), 'LineWidth', 5, 'Color', 'black')


t1 = annotation('textbox');
textString1 = "$$\tan \alpha = 3 \mu_s - 4 \eta_r \mu_k$$";
t1.String = textString1;
t1.Interpreter = 'latex';
t1.FontSize = FontSize;
t1.EdgeColor = 'none';

t2 = annotation('textbox');
textString1 = "$$\tan \alpha = 2 \eta_r \mu_k$$";
t2.String = textString1;
t2.Interpreter = 'latex';
t2.FontSize = FontSize;
t2.EdgeColor = 'none';

dampingScale = 1;
t3 = annotation('textbox');
textString1 = sprintf('$$K_D = %.1f \\sqrt{mK_E}, D_r = %.1f \\sqrt{IK_r}$$', dampingScale, dampingScale);
t3.String = textString1;
t3.Interpreter = 'latex';
t3.FontSize = FontSize;
t3.EdgeColor = 'none';

t4 = annotation('textbox');
t4.String = 'zero initial velocity';
t4.FontSize = FontSize;
t4.EdgeColor = 'none';


toc
title(sprintf('$$\\mu_s = %.2f, \\mu_k=%.2f$$', mu_s, mu_k), 'FontSize', FontSize, 'Interpreter', 'latex')
