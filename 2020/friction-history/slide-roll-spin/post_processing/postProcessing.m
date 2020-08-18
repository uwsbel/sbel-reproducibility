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

FontSize = 22;
LineWidth = 2;
%close all
% t = t(1:500);
% pos = pos(1:500,:);
% velo = velo(1:500,:);
% acc = acc(1:500,:);
% omic = omic(1:500,:);
% omic_dot = omic_dot(1:500,:);
% angular_pos = angular_pos(1:500);
% Fe_array = Fe_array(1:500);
% Te_array = Te_array(1:500);
% Fd_array = Fd_array(1:500);
% Td_array = Td_array(1:500);
% relative_slide = relative_slide(1:500);
% relative_roll = relative_roll(1:500);

    
    
% scenario sphere on plane with initial velocity
% figure('units','normalized','outerposition',[0 0 1 1]);
% subplot(2,3,1)
% makePlotYY(t,sqrt(pos(:,2).^2+pos(:,1).^2),t,angular_pos ,'time (sec)', 'position of CM (m)', 'angular position (rad)', '3D', LineWidth, FontSize)
% subplot(2,3,2)
% makePlotYY(t,sqrt(velo(:,2).^2+velo(:,1).^2),t,sqrt(omic(:,1).^2+omic(:,2).^2) ,'time (sec)', 'velocity of CM (m/s)', 'angular velocity (rad/s)', sprintf('\\mu_s=%.2f,\\mu_k=%.2f', mu_s, mu_k), LineWidth, FontSize)
% 
% subplot(2,3,3)
% makePlotYY(t,sqrt(acc(:,2).^2+acc(:,1).^2),t, sqrt(omic_dot(:,1).^2+omic(:,2).^2) ,'time (sec)', 'acceleration of CM (m/s^2)', 'angular acceleration (rad/s^2)', sprintf('dt=%g', dt), LineWidth, FontSize)
% %makePlotYY(t,sqrt(acc(:,2).^2),t, sqrt(omic_dot(:,1).^2) ,'time (sec)', 'acceleration of CM (m/s^2)', 'angular acceleration (rad/s^2)', sprintf('dt=%g', dt), LineWidth, FontSize)
% 
% subplot(2,3,4)
% makePlotYY(t, Fe_array, t, Te_array, 'time (sec)', 'F_E (N)', 'T_E (Nm)', sprintf('(Fr_s,Fr_k) = (%.0f,%.0f)N', mu_s*mass*gravity, mu_k*mass*gravity), LineWidth, FontSize)
% subplot(2,3,5)
% makePlotYY(t, Fd_array, t, Td_array, 'time (sec)', 'F_D (N)', 'T_D (Nm)',  sprintf('D_t=%.0fN/(m/s), D_r=%.0fNm/(rad/s)', sqrt(mass*Ke)*eta_t, Cr*eta_r), LineWidth, FontSize);
% subplot(2,3,6)
% makePlotYY(t, relative_slide*1e3, t, relative_roll*1e3, 'time (sec)', 'S_{ij} (mm)','\Theta_{ij} (\times10^{-3}rad)', 'relative motion', LineWidth, FontSize);

% scenario sphere on incline
% figure('units','normalized','outerposition',[0 0 1 1]);
% subplot(2,3,1)
% makePlotYY(t,(pos(:,2) * cos(slope_angle) + pos(:,3) * sin(slope_angle)),t,angular_pos ,'time (sec)', 'position of CM (m)', 'angular position (rad)', '3D', LineWidth, FontSize)
% subplot(2,3,2)
% makePlotYY(t,(velo(:,2) * cos(slope_angle) + velo(:,3) * sin(slope_angle)),t,(omic(:,1)) ,'time (sec)', 'velocity of CM (m/s)', 'angular velocity (rad/s)', sprintf('\\mu_s=%.2f,\\mu_k=%.2f', mu_s, mu_k), LineWidth, FontSize)
% subplot(2,3,3)
% makePlotYY(t,(acc(:,2) * cos(slope_angle) + acc(:,3) * sin(slope_angle)),t, (omic_dot(:,1)) ,'time (sec)', 'acceleration of CM (m/s^2)', 'angular acceleration (rad/s^2)', sprintf('sphere up %d deg slope', slope_angle_deg), LineWidth, FontSize)
% subplot(2,3,4)
% makePlotYY(t, Fe_array, t, Te_array, 'time (sec)', 'F_E (N)', 'T_E (Nm)', sprintf('(Fr_s,Fr_k) = (%.0f,%.0f)N', mu_s*mass*gravity, mu_k*mass*gravity), LineWidth, FontSize)
% subplot(2,3,5)
% makePlotYY(t, Fd_array, t, Td_array, 'time (sec)', 'F_D (N)', 'T_D (Nm)',  sprintf('D_t=%.0fN/(m/s), D_r=%.0fNm/(rad/s)', sqrt(mass*Ke)*eta_t, Cr*eta_r), LineWidth, FontSize);
% subplot(2,3,6)
% makePlotYY(t, relative_slide*1e3, t, relative_roll*1e3, 'time (sec)', 'S_{ij} (mm)','\Theta_{ij} (\times10^{-3}rad)', 'relative motion', LineWidth, FontSize);
% 
% figure;
% makePlotYY(t,(velo(:,2) * cos(slope_angle) + velo(:,3) * sin(slope_angle)),t,(omic(:,1)) ,'time (sec)', 'velocity of CM (m/s)', 'angular velocity (rad/s)', sprintf('\\mu_s=%.2f,\\mu_k=%.2f', mu_s, mu_k), LineWidth, FontSize)
% 
% figure;
% t0 = 3;  % starting time step to plot
% omic_x = omic(t0:end,1);
% velocity = (sqrt(velo(:,2).^2 + velo(:,3).^2));
% ratio = velocity(t0:end)./omic_x;
% makePlot(t(t0:end), ratio, 'time(sec)', 'x/omic','', LineWidth, FontSize);

% geodeisc of sphere on plane with initial velocity
xlim_low = 0.6;
xlim_high = 0.8;
% figure('units','normalized','outerposition',[0 0 1 1]);
% subplot(2,2,1)
% makePlotXYY(t, sqrt(wGeo_pos(:,2).^2+wGeo_pos(:,1).^2), t, sqrt(woGeo_pos(:,2).^2+woGeo_pos(:,1).^2), ...
%     'time (sec)', 'position of CM (m)', 'without geodesic', 'with geodesic', sprintf('initialVelo=%.2d', norm(velo(1,:))), LineWidth, FontSize)
% xlim([xlim_low, xlim_high]);
% subplot(2,2,2)
% makePlot(t, sqrt(wGeo_pos(:,2).^2+wGeo_pos(:,1).^2) - sqrt(woGeo_pos(:,2).^2+woGeo_pos(:,1).^2), ...
%     'time (sec)', 'difference', '', LineWidth, FontSize);
% xlim([xlim_low, xlim_high]);
% 
% fprintf('relative error [min, avg, max], average absolute error\n');
% 
% approx = sqrt(woGeo_pos(:,2).^2+woGeo_pos(:,1).^2);
% original = sqrt(wGeo_pos(:,2).^2+wGeo_pos(:,1).^2);
% [rErrorMin, rErrorAvg, rErrorMax] = relativeError(approx, original);
% aErrorAvg = sum(abs(approx - original))/length(approx);
% fprintf('position of CM: [%g, %g, %g]  %g\n', rErrorMin, rErrorAvg, rErrorMax, aErrorAvg);
% 
% subplot(2,2,3)
% makePlotXYY(t, wGeo_angularPos, t, woGeo_angularPos, ...
%     'time (sec)', 'angular position of CM (m)', 'without geodesic', 'with geodesic', sprintf('initialVelo=%.2d', norm(velo(1,:))), LineWidth, FontSize)
% xlim([xlim_low, xlim_high]);
% 
% subplot(2,2,4)
% makePlot(t, wGeo_angularPos - woGeo_angularPos, ...
%     'time (sec)', 'difference', '', LineWidth, FontSize);
% xlim([xlim_low, xlim_high]);
% 
% approx = woGeo_angularPos;
% original = wGeo_angularPos;
% [rErrorMin, rErrorAvg, rErrorMax] = relativeError(approx, original);
% aErrorAvg = sum(abs(approx - original))/length(approx);
% fprintf('angular position: [%g, %g, %g]  %g\n', rErrorMin, rErrorAvg, rErrorMax, aErrorAvg);



figure('units','normalized','outerposition',[0 0 1 1]);
subplot(2,2,1)
makePlotXYY(t, sqrt(wGeo_velo(:,2).^2+wGeo_velo(:,1).^2), t, sqrt(woGeo_velo(:,2).^2+woGeo_velo(:,1).^2), ...
    'time (sec)', 'velocity of CM (m/s)', 'without geodesic', 'with geodesic', sprintf('initialVelo=%.2d', norm(velo(1,:))), LineWidth, FontSize)
xlim([xlim_low, xlim_high]);

subplot(2,2,2)
makePlot(t, sqrt(wGeo_velo(:,2).^2+wGeo_velo(:,1).^2) - sqrt(woGeo_velo(:,2).^2+woGeo_velo(:,1).^2), ...
    'time (sec)', 'difference', '', LineWidth, FontSize);
xlim([xlim_low, xlim_high]);

approx = sqrt(woGeo_velo(:,2).^2+woGeo_velo(:,1).^2);
original = sqrt(wGeo_velo(:,2).^2+wGeo_velo(:,1).^2);
[rErrorMin, rErrorAvg, rErrorMax] = relativeError(approx, original);
aErrorAvg = sum(abs(approx - original))/length(approx);

fprintf('velocity of CM: [%g, %g, %g]  %g\n', rErrorMin, rErrorAvg, rErrorMax, aErrorAvg);


subplot(2,2,3)
makePlotXYY(t, sqrt(wGeo_omic(:,2).^2+wGeo_omic(:,1).^2), t, sqrt(woGeo_omic(:,2).^2+woGeo_omic(:,1).^2), ...
    'time (sec)', 'angular velocity (rad/s)', 'without geodesic', 'with geodesic', sprintf('initialVelo=%.2d', norm(velo(1,:))), LineWidth, FontSize)
xlim([xlim_low, xlim_high]);

subplot(2,2,4)
makePlot(t, sqrt(wGeo_omic(:,2).^2+wGeo_omic(:,1).^2) - sqrt(woGeo_omic(:,2).^2+woGeo_omic(:,1).^2), ...
    'time (sec)', 'difference', '', LineWidth, FontSize);
xlim([xlim_low, xlim_high]);

approx = sqrt(woGeo_omic(:,2).^2+woGeo_omic(:,1).^2);
original = sqrt(wGeo_omic(:,2).^2+wGeo_omic(:,1).^2);
[rErrorMin, rErrorAvg, rErrorMax] = relativeError(approx, original);
aErrorAvg = sum(abs(approx - original))/length(approx);

fprintf('angular position: [%g, %g, %g]  %g\n', rErrorMin, rErrorAvg, rErrorMax, aErrorAvg);



figure('units','normalized','outerposition',[0 0 1 1]);
subplot(2,2,1)
makePlotXYY(t, wGeo_Fe, t, woGeo_Fe, ...
    'time (sec)', 'F_E(N)', 'without geodesic', 'with geodesic', sprintf('initialVelo=%.2d', norm(velo(1,:))), LineWidth, FontSize)
xlim([xlim_low, xlim_high]);

subplot(2,2,2)
makePlot(t, wGeo_Fe - woGeo_Fe, ...
    'time (sec)', 'difference', '', LineWidth, FontSize);
xlim([xlim_low, xlim_high]);

approx = woGeo_Fe;
original = wGeo_Fe;
[rErrorMin, rErrorAvg, rErrorMax] = relativeError(approx, original);
aErrorAvg = sum(abs(approx - original))/length(approx);

fprintf('F_E = [%g, %g, %g]  %g\n', rErrorMin, rErrorAvg, rErrorMax, aErrorAvg);



subplot(2,2,3)
makePlotXYY(t, wGeo_Te, t, woGeo_Te, ...
    'time (sec)', 'T_E (N)', 'without geodesic', 'with geodesic', sprintf('initialVelo=%.2d', norm(velo(1,:))), LineWidth, FontSize)
xlim([xlim_low, xlim_high]);

subplot(2,2,4)
makePlot(t, wGeo_Te - woGeo_Te, ...
    'time (sec)', 'difference', '', LineWidth, FontSize);
xlim([xlim_low, xlim_high]);

approx = woGeo_Te;
original = wGeo_Te;
[rErrorMin, rErrorAvg, rErrorMax] = relativeError(approx, original);
aErrorAvg = sum(abs(approx - original))/length(approx);

fprintf('T_E: [%g, %g, %g]  %g\n', rErrorMin, rErrorAvg, rErrorMax, aErrorAvg);



figure('units','normalized','outerposition',[0 0 1 1]);
subplot(2,2,1)
makePlotXYY(t, wGeo_relativeSlide, t, woGeo_relativeSlide, ...
    'time (sec)', 'S_{ij} (m)', 'without geodesic', 'with geodesic', sprintf('initialVelo=%.2d', norm(velo(1,:))), LineWidth, FontSize)
xlim([xlim_low, xlim_high]);

subplot(2,2,2)
makePlot(t, wGeo_relativeSlide - woGeo_relativeSlide, ...
    'time (sec)', 'difference', '', LineWidth, FontSize);
xlim([xlim_low, xlim_high]);

approx = woGeo_relativeSlide;
original = wGeo_relativeSlide;
[rErrorMin, rErrorAvg, rErrorMax] = relativeError(approx, original);
aErrorAvg = sum(abs(approx - original))/length(approx);

fprintf('relative slide: [%g, %g, %g]  %g\n', rErrorMin, rErrorAvg, rErrorMax, aErrorAvg);


% 
subplot(2,2,3)
makePlotXYY(t, wGeo_relativeRoll, t, woGeo_relativeRoll, ...
    'time (sec)', '\Theta_{ij} (rad)', 'without geodesic', 'with geodesic', sprintf('initialVelo=%.2d', norm(velo(1,:))), LineWidth, FontSize)
xlim([xlim_low, xlim_high]);

subplot(2,2,4)
makePlot(t, wGeo_relativeRoll - woGeo_relativeRoll, ...
    'time (sec)', 'difference', '', LineWidth, FontSize);
xlim([xlim_low, xlim_high]);


approx = woGeo_relativeRoll;
original = wGeo_relativeRoll;
[rErrorMin, rErrorAvg, rErrorMax] = relativeError(approx, original);
aErrorAvg = sum(abs(approx - original))/length(approx);

fprintf('relative roll: [%g, %g, %g]  %g\n', rErrorMin, rErrorAvg, rErrorMax, aErrorAvg);





figure('units','normalized','outerposition',[0 0 1 1]);
subplot(2,2,1)
makePlotXYY(t, wGeo_Fd, t, woGeo_Fd, ...
    'time (sec)', 'F_D(N)', 'without geodesic', 'with geodesic', sprintf('initialVelo=%.2d', norm(velo(1,:))), LineWidth, FontSize)
xlim([xlim_low, xlim_high]);

subplot(2,2,2)
makePlot(t, wGeo_Fd - woGeo_Fd, ...
    'time (sec)', 'difference', '', LineWidth, FontSize);
xlim([xlim_low, xlim_high]);

approx = woGeo_Fd;
original = wGeo_Fd;
[rErrorMin, rErrorAvg, rErrorMax] = relativeError(approx, original);
aErrorAvg = sum(abs(approx - original))/length(approx);

fprintf('F_D = [%g, %g, %g]  %g\n', rErrorMin, rErrorAvg, rErrorMax, aErrorAvg);

subplot(2,2,3)
makePlotXYY(t, wGeo_Td, t, woGeo_Td, ...
    'time (sec)', 'T_D (N)', 'without geodesic', 'with geodesic', sprintf('initialVelo=%.2d', norm(velo(1,:))), LineWidth, FontSize)
xlim([xlim_low, xlim_high]);

subplot(2,2,4)
makePlot(t, wGeo_Td - woGeo_Td, ...
    'time (sec)', 'difference', '', LineWidth, FontSize);
xlim([xlim_low, xlim_high]);

approx = woGeo_Td;
original = wGeo_Td;
[rErrorMin, rErrorAvg, rErrorMax] = relativeError(approx, original);
aErrorAvg = sum(abs(approx - original))/length(approx);

fprintf('T_D: [%g, %g, %g]  %g\n', rErrorMin, rErrorAvg, rErrorMax, aErrorAvg);


