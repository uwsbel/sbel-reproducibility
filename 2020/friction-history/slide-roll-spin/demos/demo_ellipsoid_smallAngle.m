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
addpath('/Users/lulu/Documents/Research/code/friction3DSimEngine/core');
addpath('/Users/lulu/Documents/Research/code/friction3DSimEngine/helper');
addpath('/Users/lulu/Documents/Research/code/friction3DSimEngine/post_processing');


%%%%% 10^7 for k in normal force  %%%%

% bowling ball parameters

PI = 3.141592653589793238462643383279;
mass = 5;

% ellipsoid
% radius_a = 0.2; radius_b = 0.18; radius_c = 0.15;
% c >> a = b
%radius_a = 0.5; radius_b = 0.2; radius_c = 0.2;
radius_a = 0.2; radius_b = 0.2; radius_c = 0.5;


% look up bowling isle statistics, a lot smaller friction coefficient
gravity = 9.8;
mu_s = 0.25;
mu_k = 0.2;

etaSpin = 0.05;
scenario = 'ellipsoid_spinning_eta_5e-2_frontView';
Tend = 8; dt = 1e-4;
mySimulation = simulationProcess(Tend, dt, scenario);
t = 0:mySimulation.dt:mySimulation.endTime;


% tech_report = false;
% tech_report_dir = '/Users/lulu/Documents/TechReports/Friction3D/Images/';

% initialize kinematics array
pos = zeros(length(t),3); velo = zeros(length(t),3); acc = zeros(length(t),3);
eulerPar = zeros(length(t),4); omic = zeros(length(t),3); omic_dot = zeros(length(t),3);

% sliding friction parameter
Ke = 5e4;

normalForceStiffness = 1e7;

% use eta here for the energy tie between rolling and sliding
eta = 0.001;  % choose eta to be about 1


M_spin = zeros(length(t),1);
F_slide = zeros(length(t),1);
M_roll = zeros(length(t),1);


% create an ellipsoid with initial condition
myEllipsoid = ellipsoidClass(radius_a, radius_b, radius_c, mass);

% initial condition
initialVelocity = [0; 0; 0];
initialPosition = [0; 0; radius_c - myEllipsoid.mass*gravity/normalForceStiffness];
%initialOmic = [2; 1; 3];
initialOmic = [0;0;2];


myEllipsoid.position = initialPosition;
myEllipsoid.velo = initialVelocity;
myEllipsoid.omic = initialOmic;

% create a plane using normal direction and offset
groundNormal = [0;0;1];
groundOrigin = [0;0;0];
myPlane = planeClass(groundNormal, groundOrigin);


% get contact point, project onto the plane
contactPoint_prev = myEllipsoid.findContactPointWithPlane(myPlane);

initiationOfContact = true;
isInContact = false;

forceGravity = myEllipsoid.mass * [0; 0; -gravity];



% holders


% holder for analysis of damping component of rolling resistence
% staticSlack_holder = zeros(length(t),1);
% kineticSlack_holder = zeros(length(t),1);


Tr_E_holder = zeros(length(t),1);
Tr_P_holder = zeros(length(t),1);
omic_holer = zeros(length(t),1);
omic_holer(1) = initialOmic(3);
history_holder = zeros(length(t), 1);
increment_holder = zeros(length(t), 1);


% figHdl = figure;
% figHdl.Position = [1 42 1280 663];
framesPerSecond = 100;
mySimulation.generateFrameStruct(framesPerSecond);
mySimulation.generateMovie = true;



% debugging
debug_begin = 1;
% debug_totalFrames = ceil((debug_end - debug_start)/dt);
%
% debugFrames(debug_totalFrames) = struct('cdata', [], 'colormap', []);
% frameCount = 1;

%

for i = 1:length(t)-1
    % in contact, calculate and sum all the forces
    if myEllipsoid.isInContactWithPlane(myPlane) == true
        % in contact
        isInContact = true;
        % get penetration depth
        penetrationDepth = myEllipsoid.getPenetrationDepth(myPlane);
        % get normal force
        forceNormal = penetrationDepth * normalForceStiffness * groundNormal;
        
        
        % initiation of the contact, create contact object
        if initiationOfContact == true
            initiationOfContact = false;
            myContact = ellipsoidPlaneContactModel(myEllipsoid, myPlane);
            curv = myEllipsoid.getCurvatureAtLocalPt(myContact.CP_curr_local);
            myFrictionModel = frictionModel(mu_s, mu_k, Ke, eta, forceNormal, myEllipsoid, mySimulation.dt, curv);
            myFrictionModel.etaSpin = etaSpin;
            myFrictionModel.updateFrictionStiffnessAndSlack(curv, myEllipsoid, forceNormal);
            myContact.CF_prev_global.printOut;
            
        else
            % contact continue, update contact object
            myContact.updateContactAtNextTimeStep(myEllipsoid, myPlane);
            curv = myEllipsoid.getCurvatureAtLocalPt(myContact.CP_curr_local);
            
            %             if t(i) > debug_begin
            %                 myContact.CF_prev_global.printOut;
            %             end
            
            
            % update slack and damping as well for ellipsoid
            myFrictionModel.updateFrictionStiffnessAndSlack(curv, myEllipsoid, forceNormal);
            
        end
        
        pi = geodesic(myContact.CP_prev_global_curr, myContact.CP_curr_global, ...
            myContact.CF_curr_global.n, myEllipsoid.position, false);  % body i for sphere
        pj_bar = myContact.CP_curr_global - myContact.CP_prev_global; % body j for the ground
        
        % frame for the ground
        CF_ground_u1bar = myContact.CF_prev_global.u;
        CF_global_u1    = myContact.CF_curr_global.u;
        CF_global_n1    = myContact.CF_curr_global.n;
        
        % calculate psi
        psi = cross(CF_ground_u1bar, CF_global_u1);
        psi = psi(3);
        
        
        % rotate pj_bar back by psi
        pj = rotationAboutAxis(pj_bar, groundNormal, -psi);
        
        % find relative slide and roll
        delta = pj - pi;
        excursion = pi * myEllipsoid.getCurvatureAtLocalPt(myContact.CP_curr_local);
        
        myFrictionModel.updateFrictionParameters(delta, excursion, psi);
        myFrictionModel.evaluateForces;
        % do this last after evaluating of psi and delta Sij etc
        % replace previous contact frame and contact point
        myContact.replacePreviousContactFrame;
        myContact.replacePreviousContactPoint;
        
        % get all the forces and moments from contact model
        
        % sliding friction
        Fr_sliding = - myFrictionModel.slidingFr.totalFriction;
        % sliding friction torque wrt center of mass
        M_slidingFr   = cross(myContact.CP_curr_global - myEllipsoid.position, Fr_sliding);
        % normal force torque wrt center of mass
        M_normalForce = cross(myContact.CP_curr_global - myEllipsoid.position, forceNormal);
        
        % normalized radius
        r_norm = (myContact.CP_curr_global - myEllipsoid.position)/norm(myContact.CP_curr_global - myEllipsoid.position);
        % rolling torque
        M_rolling = cross(r_norm, myFrictionModel.rollingTr.totalFriction);
        % spinning torque
        M_spinning = -myFrictionModel.spinningTr.totalFriction * myContact.CF_curr_global.n;
        
        % sum all the forces and moments
        F_total = forceGravity + forceNormal + Fr_sliding;
        
        % sum all the moments
        M_total = M_slidingFr + M_normalForce + M_spinning + M_rolling...
            - tensor(myEllipsoid.omic') * myEllipsoid.inertiaMatrix * myEllipsoid.omic;
        
        
    end
    
    
    
    % not in contact
    if myEllipsoid.isInContactWithPlane(myPlane) == false
        
        isInContact = false;
        initiationOfContact = true;
        F_total = forceGravity;
        M_total = - tensor(myEllipsoid.omic') * myEllipsoid.inertiaMatrix * myEllipsoid.omic;
        
        
        
    end
    
    
    % update acceleration at new timestep
    myEllipsoid.acc = myEllipsoid.massMatrix\F_total;
    myEllipsoid.omic_dot = myEllipsoid.inertiaMatrix\M_total;
    myEllipsoid.updateKinematics(myEllipsoid.acc, myEllipsoid.omic_dot, mySimulation.dt);
    
    if isInContact == true
        % print out data
        sliding_Fr = myFrictionModel.slidingFr;
        rolling_Tr = myFrictionModel.rollingTr;
        spinning_Tr = myFrictionModel.spinningTr;
        
        %        if t(i) > debug_begin
        %             fprintf('t=%.4f, mode=%s, omic=%g, spinTr=%g, Tr_D=%g, psi=%g, Psi=%g\n', ...
        %                 t(i), spinning_Tr.mode, ...
        %                 myEllipsoid.omic(3), M_spinning(3), spinning_Tr.plasticComponent, ...
        %                 spinning_Tr.increment, spinning_Tr.history);
        %        end
        
        history_holder(i+1) = spinning_Tr.history;
        
        Tr_E_holder(i+1) = spinning_Tr.elasticComponent;
        Tr_P_holder(i+1) = spinning_Tr.plasticComponent;
        omic_holer(i+1) = myEllipsoid.omic(3);
        history_holder(i+1) = spinning_Tr.history;
        increment_holder(i+1) = spinning_Tr.increment;
        
        
        
        
    end
    
    
    FS = 25;
    LW = 2;
    
    if mySimulation.generateMovie == true &&  mod(i, floor(length(t)/mySimulation.movieLoops)) == 0
        %            subplot(2,2,ii)
        myEllipsoid.drawReferenceFrame('r');
        hold on
        myEllipsoid.drawEllipsoid;
        %        myPlane.drawPlane(-0.5, 0.5);
        xlabel('x', 'FontSize', FS);
        ylabel('y', 'FontSize', FS);
        zlabel('z', 'FontSize', FS);
        %        view(-107, 16);
        %            view(viewIndexX(1), viewIndexY(1))
        view(-13, 15) % front view
        axis equal
        xlim([-0.5, 0.5]);
        ylim([-0.5, 0.5]);  %
        zlim([0, 1.2]);
        textHdl = text(min(xlim), max(ylim), max(zlim), ...
            sprintf('time = %.4g sec \n KE = %.3g kg m^2/s^2 \n rolling mode = %s', ...
            t(i), myEllipsoid.getKineticEnergy, spinning_Tr.mode));
        textHdl.FontSize = FS-2;
        set(gca, 'linewidth', LW);
        set(gca, 'FontSize', FS-3);
        grid on
        myContact.CF_curr_global.drawContactFrame(myContact.CP_curr_global, 'g', myEllipsoid.c*0.8);
        mySimulation.writeCurrentFrame;
        hold off
    end
    
end

fprintf('end of simulation\n')


% if mySimulation.generateMovie == true
%     mySimulation.writeMovies(mySimulation.name);
% end

% figure(1);
% hold on
% plot(t,omic_holer);

% vidObj = VideoWriter(strcat('debug', '.avi'));
% vidObj.Quality = 100;
% open(vidObj);
% writeVideo(vidObj, debugFrames);
% close(vidObj);

% Tstart = 1.4;
% Tend = 1.5;
% subplot(2,3,1);
% hold on
% makePlot(t, Tr_E_holder, '', 'elastic component(Nm)', '', LW, FS);
% xlim([Tstart,Tend]);
%
%
% subplot(2,3,2);
% hold on
% makePlot(t, Tr_P_holder, '', 'plastic component(Nm)', '', LW, FS);
% xlim([Tstart,Tend]);
%
% subplot(2,3,3);
% hold on
% makePlot(t, Tr_E_holder + Tr_P_holder, '', 'total torque', '', LW, FS);
% xlim([Tstart,Tend]);
%
% subplot(2,3,4);
% hold on
% makePlot(t, omic_holer, '', '$$ \omega (rad/s) $$', '', LW, FS);
% xlim([Tstart,Tend]);
%
% subplot(2,3,5);
% hold on
% makePlot(t, history_holder, '', 'history', '', LW, FS);
% xlim([Tstart,Tend]);
%
% subplot(2,3,6);
% hold on
% makePlot(t, increment_holder, '', 'increment', '', LW, FS);
% xlim([Tstart,Tend]);
