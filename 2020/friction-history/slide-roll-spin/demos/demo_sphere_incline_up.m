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

% demo sphere incline up
if strcmp(computer, 'MACI64')
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/core/');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/helper/');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/post_processing/');
    
else
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/core');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/helper');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/post_processing');
    
end


%eta = 0.4;
%slope_angle_deg = 2;
Ke = 1e5;
Kr = 4*eta * 0.2^2 * Ke;

% create a sphere
radius = 0.2;  mass = 5;
mySphere = sphereClass(radius, mass);

scenario = 'sphere_incline_up';
PI = 3.141592653589793238462643383279;

% create slope object
%slope_angle_deg = 20;
slope_angle = slope_angle_deg/180*PI;
planeNormal = [0;-sin(slope_angle);cos(slope_angle)];
planeOffset = radius * [0; sin(slope_angle); -cos(slope_angle)];
mySlope = planeClass(planeNormal, planeOffset);

% wall object for visualization
w1Origin = [-0.5; -0.5; 0]; w1Normal = [1; 0; 0]; wall1 = planeClass(w1Normal, w1Origin);
w2Normal = [0;-1;0]; w2Origin = [0.5; 0.5; 0]; wall2 = planeClass(w2Normal, w2Origin);

gravity = 9.8;
mu_s = 0.25; mu_k = 0.2;


useGeodesic = true; % whether to use geodesic or cut the corner

Tend = 0.8; dt = 1e-4;
mySimulation = simulationProcess(Tend, dt, scenario);
t = 0:mySimulation.dt:mySimulation.endTime;
ballTransparency = 0.5;
% initialize kinematics and dynamics info array
% pos_holder = zeros(length(t),3); velo_holder = zeros(length(t),3); acc_holder = zeros(length(t),3);
% eulerPar_holder = zeros(length(t),4); omic_holder = zeros(length(t),3); omic_dot_holder = zeros(length(t),3);
% Psi_holder = zeros(length(t),1);
% psi_holder = zeros(length(t),1);
% slidingFrE_holder = zeros(length(t), 3);
% rollingTrE_holder = zeros(length(t), 3);
% slidingFrP_holder = zeros(length(t), 3);
% rollingTrP_holder = zeros(length(t), 3);
% slidingFr_holder = zeros(length(t), 3);
% rollingTr_holder = zeros(length(t), 3);



% sliding_mode_holder = cell(length(t),1);


% rolling friction parameter
%Kr = 5000;
Ke = 1e5;
Kr =4*eta*radius^2*Ke;
etaSpin = 0;

% M_spin_holder = zeros(length(t),1);

% initial condition
veloMag = 0.5;
%veloMag = 0;

initial_velo = veloMag * [0; cos(slope_angle); sin(slope_angle)];
initial_omic = [0; 0; 0];
initial_pos = [0; 0; 0];
initial_orientation = [1 0 0; 0 cos(slope_angle) -sin(slope_angle); 0 sin(slope_angle) cos(slope_angle)];

mySphere.position = initial_pos;
mySphere.velo = initial_velo;
mySphere.orientation = initial_orientation;
mySphere.omic = initial_omic;
mySphere.renderTransparency = ballTransparency;


eulerPar_holder(1,:) = getPfromA(initial_orientation);
mySphere.eulerParameter = getPfromA(initial_orientation);


initiationOfContact = true;
isInContact = true;

%     gravity force
F_gravity = mass * [0; 0; -gravity];
%
%     contact force
F_normal = mass*gravity*cos(slope_angle)*[0; -sin(slope_angle); cos(slope_angle)];

mySimulation.generateFrameStruct(100);
mySimulation.generateMovie = false;

if mySimulation.generateMovie == true
    figure('units','normalized','outerposition',[0   0   0.76   0.78]);
end

for i = 1:length(t)-1
    
    if isInContact == true
        
        % initiation of the contact, create contact object
        if initiationOfContact == true
            initiationOfContact = false;
            
            myContact = spherePlaneContactModel(mySphere, mySlope);
            myFrictionModel = frictionModel(mu_s, mu_k, Ke, eta, 0, F_normal, mySphere, mySimulation.dt);
            myFrictionModel.etaSpin = etaSpin;
            
            % <==> need to rewrite this part, rewrite friction model
            myFrictionModel.spinningTr.stiffness = myFrictionModel.etaSpin * myFrictionModel.rollingTr.stiffness;
            myFrictionModel.spinningTr.dampingCr = 2*sqrt(mySphere.inertia * myFrictionModel.spinningTr.stiffness);
        else
            % contact continue, update contact object
            myContact.updateContactAtNextTimeStep(mySphere, mySlope, slope_angle_deg);
        end
        
    else
        % not in contact, delete contact object?
        isInContact = false;
        initiationOfContact = true;
        % get sum of all the forces and update kinematics
    end
    
    
    pi = geodesic(myContact.CP_prev_global_curr, myContact.CP_curr_global, ...
        myContact.CF_curr_global.n, mySphere.position, useGeodesic);  % body i for sphere
    
    pj = myContact.CP_curr_global - myContact.CP_prev_global; % body j for the ground
    
    % frame for the ground
    CF_ground_u1bar = myContact.CF_prev_global.u;
    CF_global_u1 = myContact.CF_curr_global.u;
    CF_global_n1 = myContact.CF_curr_global.n;
    
    % determine spin angle direction
    if dot(cross(CF_ground_u1bar, CF_global_u1), CF_global_n1) > 0
        psi =  norm(cross(CF_ground_u1bar, CF_global_u1));
    else
        psi = -norm(cross(CF_ground_u1bar, CF_global_u1));
    end
    
    
    
    delta = pj - pi;
    excursion = pi/mySphere.radius;
    myFrictionModel.updateFrictionParameters(delta, excursion, psi);
    myFrictionModel.evaluateForces;
    
    
    
    
    % do this last after evaluating of psi and delta Sij etc
    % replace previous contact frame and contact point
    myContact.replacePreviousContactFrame;
    myContact.replacePreviousContactPoint;
    
    
    
    
    
    
    
    % sliding friction
    Fr_sliding = - myFrictionModel.slidingFr.totalFriction;
    % sliding friction torque wrt center of mass
    M_fr = cross(myContact.CP_curr_global - mySphere.position, Fr_sliding);
    % normalized radius
    r_norm = (myContact.CP_curr_global - mySphere.position)/norm(myContact.CP_curr_global - mySphere.position);
    % rolling torque
    M_rolling = cross(r_norm, myFrictionModel.rollingTr.totalFriction);
    % spinning torque
    M_spinning = -myFrictionModel.spinningTr.totalFriction * myContact.CF_curr_global.n;
    
    % sum all the forces and moments
    F_total = F_gravity + F_normal + Fr_sliding;
    
    % sum all the moments
    M_total = M_fr + M_spinning - tensor(mySphere.omic') * mySphere.inertiaMatrix * mySphere.omic + M_rolling;
    
    % update acceleration at new timestep
    mySphere.acc = mySphere.massMatrix\F_total;
    mySphere.omic_dot = mySphere.inertiaMatrix\M_total;
    mySphere.updateKinematics(mySphere.acc, mySphere.omic_dot, mySimulation.dt);
    
    
    
    % recording data
%     omic_dot_holder(i+1,:) = mySphere.omic_dot';
%     acc_holder(i+1,:) = mySphere.acc';
%     velo_holder(i+1,:) = mySphere.velo';
%     pos_holder(i+1,:) = mySphere.position';
%     eulerPar_holder(i+1,:) = mySphere.eulerParameter';
    
    % print out data
%     sliding_Fr = myFrictionModel.slidingFr;
%     rolling_Tr = myFrictionModel.rollingTr;
%     spinning_Tr = myFrictionModel.spinningTr;
    %     fprintf('t=%.4f, Sij=%g, d_Sij=%g, Fr(%s):%g=%g+%g, Tr(%s):%g=%g+%g\n', ...
    %         t(i), norm(sliding_Fr.history), norm(sliding_Fr.increment), ...
    %         sliding_Fr.mode, norm(sliding_Fr.totalFriction), norm(sliding_Fr.elasticComponent), norm(sliding_Fr.plasticComponent), ...
    %         rolling_Tr.mode, norm(rolling_Tr.totalFriction), norm(rolling_Tr.elasticComponent), norm(rolling_Tr.plasticComponent));
    
    
    
%     omic_holder(i+1,:) = mySphere.omic';
%     slidingFrE_holder(i+1, :) = sliding_Fr.elasticComponent';
%     rollingTrE_holder(i+1, :) = cross(r_norm, myFrictionModel.rollingTr.elasticComponent)';
%     slidingFrP_holder(i+1, :) = sliding_Fr.plasticComponent';
%     rollingTrP_holder(i+1, :) = cross(r_norm, myFrictionModel.rollingTr.plasticComponent)';
%     slidingFr_holder (i+1, :) = sliding_Fr.totalFriction';
%     rollingTr_holder (i+1, :)= cross(r_norm, myFrictionModel.rollingTr.totalFriction)';
    
    
%     sliding_mode_holder{i+1} = sliding_Fr.mode;
    
    FS = 40;
    LW = 4;
    
    if mySimulation.generateMovie == true &&  mod(i, floor(length(t)/mySimulation.movieLoops)) == 0
        %   if mySimulation.generateMovie == true && i == 4900
        %    if mySimulation.generateMovie == true &&  i == length(t) - 1
        
%         if mySimulation.generateMovie == true
        mySphere.drawSphereSurface('checkerboard');
        hold on
%        wall1.drawPlaneColor(-0.5, 0.5, 1);
%        hold on
        
 %       wall2.drawPlaneColor(-0.5, 0.5, 1.2);
        
%         xlabel('x', 'FontSize', FS);
%         ylabel('y', 'FontSize', FS);
%         zlabel('z', 'FontSize', FS);
        mySphere.drawReferenceFrame('r');
        hold on
        mySlope.drawPlane(-1, 1);
%        mySlope.drawPlane(-1.6, -1);

view(84, 1);   % isotropic view
        %        view(0,90)   % top view
        %        view(90, 0);  % front view
        axis equal
        xlim([-1  , 1  ]);
        ylim([-0.25, 0.5]);
        zlim([-0.25, 0.2]);
%        xlim([-1.6  , 1  ]);
%        ylim([-0.25, 0.5]);
%        zlim([-0.8, -0.3]);

        textHdl = text(min(xlim), min(ylim), max(zlim), ...
            sprintf('time=%.4gsec, \\alpha=%.0f^o', ...
            t(i+1), slope_angle_deg));
        textHdl.FontSize = FS;
        set(gca, 'linewidth', LW);
        set(gca, 'FontSize', FS-3)
        hold on
        myContact.CF_curr_global.drawContactFrame(myContact.CP_curr_global, 'g', mySphere.radius*0.8);
        
        hold on
        myVelo = myVector3(mySphere.velo');
        myVelo.drawVectWithTextOrigin(mySphere.position', 1.2, 'b', ...
            sprintf('v_{CM}=%.2g m/s\n \\omega=%.2g rad/s \n v_{CM}/ \\omega=%.2g', ...
            mySphere.velo(2) * cos(slope_angle) + mySphere.velo(3) * sin(slope_angle), ...
            mySphere.omic(1), norm(mySphere.velo)/norm(mySphere.omic)), ...
            FS)
        
        hold on
        % draw sliding friction force with respect to the global reference
        % frame
        mySlidingFr = myVector3(-sliding_Fr.totalFriction);
        mySlidingFr.drawVectWithTextOrigin(myContact.CP_curr_global, 0.05, 'm', ...
            sprintf('Fr(%s)=%.2fN, Tr(%s)=%.2f Nm', ...
            sliding_Fr.mode, Fr_sliding(2) * cos(slope_angle) + Fr_sliding(3) * sin(slope_angle), ...
            rolling_Tr.mode, M_rolling(1)), ...
            FS);
        
        hdl = gca;
        hdl.XTick = [];
%        hdl.YTick = [];
%        hdl.ZTick = [];
        
        
        
        mySimulation.writeCurrentFrame;
        
        hold off
        grid on
    end
    
    
end

% if mySimulation.generateMovie == true
%     mySimulation.writeMovies(mySimulation.name);
% end

%%
% FontSize = 36;
% LineWidth = 3;
% 
% figure('units','normalized','outerposition',[0   0   1   0.7]);
% subplot(1,2,1)
% makePlotYY(t,(velo_holder(:,2) * cos(slope_angle) + velo_holder(:,3) * sin(slope_angle)),...
%     t,(omic_holder(:,1)) , ...
%     'time (sec)', 'v_{CM} (m/s)', '\omega (rad/s)', sprintf('\\alpha = %.0f ^o', slope_angle_deg), LW, FS)
% 
% subplot(1,2,2)
% makePlotYY(t, slidingFrE_holder(:,2) * cos(slope_angle) + slidingFrE_holder(:,3) * sin(slope_angle) ...
%     , t, rollingTrE_holder(:,1), ...
%     'time (sec)', 'F_E (N)', 'T_E (Nm)', '', LW, FS)
% 
% 
% i1 = 0.049;
% i2 = 0.0829;
% i3 = 0.0913;
% subplot(1,2,1)
% yy = -1:0.01:0.5;
% hold on
% plot(ones(size(yy))*i1,yy, 'black', 'LineStyle', '-', 'LineWidth', LineWidth-1)
% plot(ones(size(yy))*i2,yy, 'black', 'LineStyle', '-', 'LineWidth', LineWidth-1)
% plot(ones(size(yy))*i3,yy, 'black', 'LineStyle', '-', 'LineWidth', LineWidth-1)
% text(i1/2,            max(ylim)*0.8, 'I'  , 'FontSize', FontSize, 'LineStyle', 'none');
% text((i1+i2)/2,       max(ylim)*0.8, 'II' , 'FontSize', FontSize, 'LineStyle', 'none');
% text((i2+i3)/2,       max(ylim)*0.8, 'III', 'FontSize', FontSize, 'LineStyle', 'none');
% text((i3+max(xlim))/2,max(ylim)*0.8, 'IV' , 'FontSize', FontSize, 'LineStyle', 'none');
% 
% 
% % textHdl = text(min(xlim), max(ylim), 'Li
% %             sprintf('time=%.4gsec\n F_{total}=[%g, %g, %g]N\n M_{total}=[%g, %g, %g]', ...
% %             t(i), F_total(1), F_total(2), F_total(3), M_total(1), M_total(2), M_total(3)));
% % textHdl.FontSize = FS;
% subplot(1,2,2)
% yy = -20:0.01:20;
% hold on
% plot(ones(size(yy))*i1,yy, 'black', 'LineStyle', '-', 'LineWidth', LineWidth-1)
% plot(ones(size(yy))*i2,yy, 'black', 'LineStyle', '-', 'LineWidth', LineWidth-1)
% plot(ones(size(yy))*i3,yy, 'black', 'LineStyle', '-', 'LineWidth', LineWidth-1)
% text(i1/2,            max(ylim)*0.8, 'I'  , 'FontSize', FontSize, 'LineStyle', 'none');
% text((i1+i2)/2,       max(ylim)*0.8, 'II' , 'FontSize', FontSize, 'LineStyle', 'none');
% text((i2+i3)/2,       max(ylim)*0.8, 'III', 'FontSize', FontSize, 'LineStyle', 'none');
% text((i3+max(xlim))/2,max(ylim)*0.8, 'IV' , 'FontSize', FontSize, 'LineStyle', 'none');




% subplot(2,2,3)
% makePlotYY(t, slidingFrP_holder(:,2) * cos(slope_angle) + slidingFrP_holder(:,3) * sin(slope_angle) ...
%     , t, rollingTrP_holder(:,1), ...
%     'time (sec)', 'F_D (N)', 'T_D (Nm)', '', LW, FS)
% subplot(2,2,4)
% makePlotYY(t, slidingFr_holder(:,2) * cos(slope_angle) + slidingFr_holder(:,3) * sin(slope_angle) ...
%     , t, rollingTr_holder(:,1), ...
%     'time (sec)', '$$\sum F (N)$$', '$$\sum T (Nm)$$', '', LW, FS)