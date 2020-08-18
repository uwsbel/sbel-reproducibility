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

eta_array = [8 9 10 12] * 1e-4;

figHdl = figure;
figHdl.Position = [1 42 1280 663];
subplot(1,2,1);
hold on
subplot(1,2,2)
hold on

LW = 1;  % line width
FS = 20; % font size



for index = 1:length(eta_array)
    eta = eta_array(index);
    demo_elipsoid_rolling;
    
    
    KE = kineticEnergy_holder;
    rollingTr = rollingTr_holder;
    
    subplot(1,2,1)
    makePlot(t, KE, 't', 'Kinetic Energy (kg m^2/s^2)', '', LW, FS);
    
    subplot(1,2,2)
    makePlot(t, rollingTr, 't', 'Rolling Friction (Nm)', '', LW, FS);
end

legend(sprintf('\\eta=%g', eta_array(1)), sprintf('\\eta=%g', eta_array(2)), ...
    sprintf('\\eta=%g', eta_array(3)), sprintf('\\eta=%g', eta_array(4)))