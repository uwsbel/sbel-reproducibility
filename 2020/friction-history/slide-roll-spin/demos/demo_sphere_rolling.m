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

clc
clear all
close all

addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/core/');
addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/helper/');
% create a sphere
radius = 0.2;  mass = 5;
mySphere = sphereClass(radius, mass);


% sliding friction parameter
Ke = 1e5;
% rolling friction parameter
eta_roll = 0.4;
Kr =4*eta_roll*mySphere.radius^2*Ke;

% spinning parameter
etaSpin = 0.06;



% create plane object
planeNormal = [0;0;1];
planeOffset = [0;0;0];
myPlane = planeClass(planeNormal, planeOffset);


gravity = 9.8;
mu_s = 0.25; mu_k = 0.2;


useGeodesic = true; % whether to use geodesic or cut the corner

scenario = sprintf('sphere_rolling_eta_%dE-1_spinning_eta_%gE-2', eta_roll*10, etaSpin*100);
Tend = 4; dt = 1e-4;
mySimulation = simulationProcess(Tend, dt, scenario);
t = 0:mySimulation.dt:mySimulation.endTime;

% tech_report = false;
% tech_report_dir = '/Users/lulu/Documents/TechReports/Friction3D/Images/';

% initialize kinematics array
pos_holder = zeros(length(t),3); velo_holder = zeros(length(t),3); acc_holder = zeros(length(t),3);
eulerPar_holder = zeros(length(t),4); omic_holder = zeros(length(t),3); omic_dot_holder = zeros(length(t),3);




Fe_array = zeros(length(t),1);    % elastic part of friction force magnitude
Te_array = zeros(length(t),1);    % elastic part of rolling torque magnitude
Fd_array = zeros(length(t),1);    % plastic part of friction force magnitude
Td_array = zeros(length(t),1);    % plastic part of rolling torque magnitude
Psi = zeros(length(t),1);
psi = zeros(length(t),1);
M_spin = zeros(length(t),1);



% initial condition
initial_omic = [2; 1; 3];
initial_velo = [0.4; 0.3; 0];
initial_pos = [0;0; mySphere.radius];
mySphere.position = initial_pos;
mySphere.omic = initial_omic;
mySphere.velo = initial_velo;
omic_holder(1,:) = initial_omic';


eulerPar_holder(1,:) = [1, 0, 0 , 0];



initiationOfContact = true;
isInContact = true;

%     gravity force
F_gravity = mass * [0; 0; -gravity];
%     
%     contact force
F_normal = -F_gravity;

mySimulation.generateFrameStruct(300);
hdl = figure(1);
hdl.Units = 'normalized';
hdl.Position = [0.28 0.28 0.4375 0.6275];

for i = 1:length(t)-1
        
    if isInContact == true
        
        % initiation of the contact, create contact object
        if initiationOfContact == true
            initiationOfContact = false;
            
            myContact = spherePlaneContactModel(mySphere, myPlane);
            myFrictionModel = frictionModel(mu_s, mu_k, Ke, eta_roll, 0, F_normal, mySphere, mySimulation.dt);
            myFrictionModel.etaSpin = etaSpin;
            
            myFrictionModel.spinningTr.stiffness = myFrictionModel.etaSpin * myFrictionModel.rollingTr.stiffness;
            myFrictionModel.spinningTr.dampingCr = 2*sqrt(mySphere.inertia * myFrictionModel.spinningTr.stiffness);
            
            
        else
            % contact continue, update contact object
            myContact.updateContactAtNextTimeStep(mySphere, myPlane, 0);
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
    
    % cos psi value
%    psi_cos = CF_ground_u1bar'*CF_global_u1/(norm(CF_ground_u1bar)*norm(CF_global_u1));
    psi_cos = CF_ground_u1bar'*CF_global_u1/sqrt(sum(CF_ground_u1bar.^2) * sum(CF_global_u1.^2));
    if psi_cos > 1
        psi_cos = 1;
    end

    if dot(cross(CF_ground_u1bar, CF_global_u1), CF_global_n1) > 0
        psi = acos(psi_cos);
    else
        psi = -acos(psi_cos);
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
    omic_dot_holder(i+1,:) = mySphere.omic_dot';
    acc_holder(i+1,:) = mySphere.acc';
    omic_holder(i+1,:) = mySphere.omic';
    velo_holder(i+1,:) = mySphere.velo';
    pos_holder(i+1,:) = mySphere.position';
    eulerPar_holder(i+1,:) = mySphere.eulerParameter';
    
    % print out data
    sliding_Fr = myFrictionModel.slidingFr;
    rolling_Tr = myFrictionModel.rollingTr;
    spinning_Tr = myFrictionModel.spinningTr;
%     fprintf('t=%.4f, Sij=%g, d_Sij=%g, Fr(%s):%g=%g+%g, Tr(%s):%g=%g+%g\n', ...
%         t(i), norm(sliding_Fr.history), norm(sliding_Fr.increment), ...
%         sliding_Fr.mode, norm(sliding_Fr.totalFriction), norm(sliding_Fr.elasticComponent), norm(sliding_Fr.plasticComponent), ...
%         rolling_Tr.mode, norm(rolling_Tr.totalFriction), norm(rolling_Tr.elasticComponent), norm(rolling_Tr.plasticComponent));

    fprintf('t=%.4f, Psi=%g, psi=%g, M_spin(%s):%g=%g+%g, omic=%g, M_spin=%g\n', ...
        t(i), spinning_Tr.history, spinning_Tr.increment, ...
        spinning_Tr.mode, spinning_Tr.totalFriction, spinning_Tr.elasticComponent, spinning_Tr.plasticComponent,...
        mySphere.omic(3), M_spinning(3));
    Psi(i+1) = spinning_Tr.history;
    psi(i+1) = spinning_Tr.increment;
    M_spin(i+1) = M_spinning(3);
    FS = 25;
    LW = 2;
    if mySimulation.generateMovie == true &&  mod(i, floor(length(t)/mySimulation.movieLoops)) == 0
        mySphere.drawSphereSurface;
        hold on 
        xlabel('x', 'FontSize', FS); 
        ylabel('y', 'FontSize', FS); 
        zlabel('z', 'FontSize', FS);
        hold on
        mySphere.drawReferenceFrame('r');
        hold on
        view(-29, 19);  % side view
%        view(0, 90);  % top view


    M_total = M_fr + M_spinning - tensor(mySphere.omic') * mySphere.inertiaMatrix * mySphere.omic + M_rolling;


        axis equal
        xlim([-0.3, 0.4]);
        ylim([-0.3, 0.5]);
        zlim([ 0,   0.5]);
        textHdl = text(min(xlim), max(ylim), max(zlim) * 1.1, ...
            sprintf(' time = %.4gsec, KE=%.3g kg m^2/s^2\n sliding mode:%s, |F_r|=%gN \n rolling mode:%s, |T_r|=%gNm \n spinning mode:%s, |T_s|=%gNm', ...
            t(i), mySphere.getKineticEnergy, ...
            sliding_Fr.mode,  norm(sliding_Fr.totalFriction), ...
            rolling_Tr.mode,  norm(rolling_Tr.totalFriction), ...
            spinning_Tr.mode, norm(spinning_Tr.totalFriction)));
        textHdl.FontSize = FS;
        set(gca, 'linewidth', LW);
        set(gca, 'FontSize', FS-3);
        hold on
        myContact.CF_curr_global.drawContactFrame(myContact.CP_curr_global, 'g', mySphere.radius*0.8);    
        hold on
        mySimulation.writeCurrentFrame;
        hold off
    end

    
end

mySimulation.writeMovies(mySimulation.name);

%%
% figure;
% subplot(2,2,1);
% plot(t(2:end), Psi(2:end));
% ylabel('history');
% 
% subplot(2,2,2)
% plot(t(2:end), omic_holder(2:end,3));
% ylabel('angular velocity');
% 
% subplot(2,2,3);
% plot(t(2:end), psi(2:end));
% ylabel('increment');
% 
% subplot(2,2,4)
% plot(t(2:end), M_spin(2:end));
% ylabel('spinning moment');
