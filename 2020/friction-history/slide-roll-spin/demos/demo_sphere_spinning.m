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

% clc
% clear all
% close all
addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/core');
addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/helper');
addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/2D');

PI = 3.141592653589793238462643383279;

% create a sphere
% glass ball
% E = 50*10^9;  % 200 Gpa
% radius = 0.02;  % 0.02m
% rho = 2.5*10^3; % rho = 8000 kg/m^3
% nu = 0.2;

% steel ball
E = 200*10^9;  % 200 Gpa
%radius = 0.02;  % 0.02m

rho = 8*10^3; % rho = 8000 kg/m^3
nu = 0.3; 


gravity = 9.8;



% ellipsoid upright
% ra = 0.02;
% rb = 0.02;
% rc = 0.05;

% ellipsoid flat
ra = 0.05;
rb = 0.02;
rc = 0.02;
radius = (ra * rb * rc)^(1/3);


% 
% 
volume = 4/3*PI*ra*rb*rc;
mass = volume * rho;
mySphere = sphereClass(ra, mass);
mySphere.youngsModulus = E;
mySphere.density = rho;
mySphere.poissonRatio = nu;

mySphere.inertia = 1/5 * mySphere.mass * (ra^2 + rb^2);

mySphere.massMatrix = mySphere.mass*eye(3,3);
mySphere.inertiaMatrix = mySphere.inertia*eye(3,3);



%figure('units','normalized','outerposition',[0 0 0.35 0.6]);

% font size and line width for plotting
FS = 36;
LW = 4.5;


% create plane object
planeNormal = [0;0;1];
planeOffset = [0;0;0];
myPlane = planeClass(planeNormal, planeOffset);



mu_s = 0.25; mu_k = 0.2;


useGeodesic = true; % whether to use geodesic or cut the corner
% sliding friction parameter
etaSpin = 0;

scenario = sprintf('sphere_spinning_physics_based');
Tend = 8; dt = 1e-4;
mySimulation = simulationProcess(Tend, dt, scenario);
t = 0:mySimulation.dt:mySimulation.endTime;


% initialize kinematics array
pos_holder = zeros(length(t),3); velo_holder = zeros(length(t),3); acc_holder = zeros(length(t),3);
eulerPar_holder = zeros(length(t),4); omic_holder = zeros(length(t),1); omic_dot_holder = zeros(length(t),3);


% rolling friction parameter
Kr = 1000;
%Ke = 5e6;


Fe_array = zeros(length(t),1);    % elastic part of friction force magnitude
Te_array = zeros(length(t),1);    % elastic part of rolling torque magnitude
Fd_array = zeros(length(t),1);    % plastic part of friction force magnitude
Td_array = zeros(length(t),1);    % plastic part of rolling torque magnitude
Psi = zeros(length(t),1);
psi = zeros(length(t),1);
M_spin = zeros(length(t),1);



% initial condition
initial_omic = [0; 0; 1];
initial_pos = [0;0; mySphere.radius];
mySphere.position = initial_pos;
mySphere.omic = initial_omic;
omic_holder(1) = initial_omic(3);


eulerPar_holder(1,:) = [1, 0, 0 , 0];



initiationOfContact = true;
isInContact = true;


mySimulation.generateFrameStruct(200);
mySimulation.generateMovie = false;




%     gravity force
F_gravity = mySphere.mass * [0; 0; -gravity];
%     
%     contact force
F_normal = -F_gravity;


for i = 1:length(t)-1
        
    if isInContact == true
        
        % initiation of the contact, create contact object
        if initiationOfContact == true
            initiationOfContact = false;
            
            myContact = spherePlaneContactModel(mySphere, myPlane);
            myFrictionModel = frictionModel(mu_s, mu_k, Ke, Kr, etaSpin, F_normal, mySphere, mySimulation.dt);
            
            % <==> need to rewrite this part, rewrite friction model
%             myFrictionModel.spinningTr.stiffness = myFrictionModel.etaSpin * myFrictionModel.rollingTr.stiffness;
%             myFrictionModel.spinningTr.dampingCr = 2*sqrt(mySphere.inertia * myFrictionModel.spinningTr.stiffness);
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
    
    psi = cross(CF_ground_u1bar, CF_global_u1);
    psi = psi(3);
    
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
    omic_holder(i+1) = mySphere.omic(3);
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
    
if mod(i, 1000) == 0
    
    fprintf('t=%.4f, Psi=%g, psi=%g, M_spin(%s):%g=%g+%g, omic=%g, acc=%g, M_spin=%g\n', ...
        t(i), spinning_Tr.history, spinning_Tr.increment, ...
        spinning_Tr.mode, spinning_Tr.totalFriction, spinning_Tr.elasticComponent, spinning_Tr.plasticComponent,...
        mySphere.omic(3), mySphere.omic_dot(3) ,M_spinning(3));
end
    Psi(i+1) = spinning_Tr.history;
    psi(i+1) = spinning_Tr.increment;
    M_spin(i+1) = M_spinning(3);
    if mySimulation.generateMovie == true &&  mod(i, floor(length(t)/mySimulation.movieLoops)) == 0
%    if mySimulation.generateMovie == true &&  i == 1
        mySphere.drawSphereSurface('checkerboard');
        hold on 
        myPlane.drawPlane(-0.3, 0.3);
%        xlabel('x', 'FontSize', FS); 
%        ylabel('y', 'FontSize', FS); 
%        zlabel('z', 'FontSize', FS); 
        mySphere.drawReferenceFrame('r');
        view(-31, 12);   % isotropic view
%        view(0,90)   % top view
        axis equal
        xlim([-1.5,1.5]*mySphere.radius);
        ylim([-1.5,1.5]*mySphere.radius);
        zlim([0, 2.5]*mySphere.radius);
        textHdl = text(min(xlim), max(ylim), max(zlim), ...
            sprintf('\\omega=%.2e rad/s \n T_{\\psi} = %.2e Nm', ...
            mySphere.omic(3), spinning_Tr.elasticComponent), ...
            'FontSize', FS);
        set(gca, 'linewidth', LW);
        set(gca, 'FontSize', FS-3)
        hold on
        myContact.CF_curr_global.drawContactFrame(myContact.CP_curr_global, 'g', mySphere.radius*0.8);
        grid on
        set(gca,'XTickLabel',[]);
        set(gca,'YTickLabel',[]);
        set(gca,'ZTickLabel',[])
        set(gca,'XTick',[]);
        set(gca,'YTick',[]);
        set(gca,'ZTick',[])
        set(gca,'ZColor',[1 1 1])

        mySimulation.writeCurrentFrame;
        if t(i+1) == 2 || t(i+1) == 4 || t(i+1) == 8
            mySaveFig(1, sprintf('sphere_spinning_eta_6e-3_%.0fsec', t(i+1)));
        end
        hold off
    end

    
end
if mySimulation.generateMovie == true
    mySimulation.writeMovies(mySimulation.name);
end

% subplot(1,3,1);
% hold on
% plot(t(2:end), Psi(2:end));
% ylabel('history');
% 
% subplot(1,3,2)
% hold on
% plot(t(2:end), omic_holder(2:end));
% ylabel('angular velocity');
% 
% subplot(1,3,3)
% hold on
% plot(t(2:end), M_spin(2:end));
% ylabel('spinning moment');
