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

if strcmp(computer, 'MACI64')
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/core');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/helper');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/2D');
else
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/core');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/helper');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/2D');
end

%%%%% 10^7 for k in normal force  %%%%
PI = 3.1415926;
mass = 5;

% ellipsoid
radius_a = 0.2; radius_b = 0.2; radius_c = 0.5;

gravity = 9.8;
mu_s = 0.25;
mu_k = 0.2;

scenario = 'ellipsoid_spinning_eta_1e-2_frontView';
Tend = 110; dt = 1e-4;
mySimulation = simulationProcess(Tend, dt, scenario);
t = 0:mySimulation.dt:mySimulation.endTime;



% tech_report = false;
% tech_report_dir = '/Users/lulu/Documents/TechReports/Friction3D/Images/';

% initialize kinematics array
pos = zeros(length(t),3); velo = zeros(length(t),3); acc = zeros(length(t),3);
eulerPar = zeros(length(t),4); omic = zeros(length(t),3); omic_dot = zeros(length(t),3);

% sliding friction parameter
Ke = 1e5;

normalForceStiffness = 1e7;

% eta_roll and eta_spin 
eta_roll = 0;
%eta_spin = 0.006;


M_spin = zeros(length(t),1);
F_slide = zeros(length(t),1);
M_roll = zeros(length(t),1);


% create an ellipsoid with initial condition
myEllipsoid = ellipsoidClass(radius_a, radius_b, radius_c, mass);

% initial condition
initialVelocity = [0; 0; 0];
initialPosition = [0; 0; radius_c - myEllipsoid.mass*gravity/normalForceStiffness];
%initialOmic = [2; 1; 3];
initialOmic = [0;0;1];


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
spinningTr_holder = zeros(length(t),1);

kineticEnergy_holder = zeros(length(t),1);
kineticEnergy_holder(1) = myEllipsoid.getKineticEnergy;
omic_holer = zeros(length(t),1);
omic_holer(1) = 1;


% holder for analysis of damping component of rolling resistence
staticSlack_holder = zeros(length(t),1);
kineticSlack_holder = zeros(length(t),1);

spinningHistory_holder = zeros(length(t), 1);
product_holder = zeros(length(t), 1);
cospsiDiff_holder = zeros(length(t), 1);
psi_holder = zeros(length(t), 1);
dampingComponent_holder = zeros(length(t),1);


% figHdl = figure;
% figHdl.Units = 'normalized';
% figHdl.Position = [0.15 0.15 0.35 0.6];
framesPerSecond = 100;
mySimulation.generateFrameStruct(framesPerSecond);
mySimulation.generateMovie = false;



% debugging
debug_start = 1.985;
debug_end = 1.995;
debug_totalFrames = ceil((debug_end - debug_start)/dt);

debugFrames(debug_totalFrames) = struct('cdata', [], 'colormap', []);
frameCount = 1;

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
        
        %
        normalForce_holder(i+1) = norm(forceNormal);
        penetrationDepth_holder(i+1) = norm(penetrationDepth);
        
        % initiation of the contact, create contact object
        if initiationOfContact == true
            initiationOfContact = false;
            myContact = ellipsoidPlaneContactModel(myEllipsoid, myPlane);
            curv = myEllipsoid.evaluateCurv([myContact.CP_curr_local(3), myContact.CP_curr_local(2)], radius_c,radius_a);
            myFrictionModel = frictionModel(mu_s, mu_k, Ke, eta_roll, eta_spin, forceNormal, myEllipsoid, mySimulation.dt, curv);
            myContact.CF_prev_global.printOut;
        else
            % contact continue, update contact object
            myContact.updateContactAtNextTimeStep(myEllipsoid, myPlane);
            curv = myEllipsoid.evaluateCurv([myContact.CP_curr_local(3), myContact.CP_curr_local(2)], radius_c,radius_a);
            
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
        
        % calculate cos(psi) value
        % NOTE: small angle, cos(psi) close to 1
        
        %        psi_cos = min(1-1e-10, CF_ground_u1bar'*CF_global_u1/sqrt(sum(CF_ground_u1bar.^2) * sum(CF_global_u1.^2)));
        %        psi_cos = CF_ground_u1bar'*CF_global_u1/(norm(CF_ground_u1bar)*norm(CF_global_u1));
        psi_cos = CF_ground_u1bar'*CF_global_u1/(norm(CF_ground_u1bar)*norm(CF_global_u1));
        
        
        product_holder(i+1) = psi_cos;
        
        cos_psi_diff = psi_cos - 1;
        cospsiDiff_holder(i+1) = cos_psi_diff;
        
        
        if psi_cos - 1 > 0
            psi_cos = 1;
        end
        
        
        % determine spin angle direction
        if dot(cross(CF_ground_u1bar, CF_global_u1), CF_global_n1) > 0
            psi = acos(psi_cos);
        else
            psi = -acos(psi_cos);
        end
        
        % rotate pj_bar back by psi
        pj = rotationAboutAxis(pj_bar, groundNormal, -psi);
        
        % find relative slide and roll
        delta = pj - pi;
	   curv = myEllipsoid.evaluateCurv([myContact.CP_curr_local(3), myContact.CP_curr_local(2)], radius_c,radius_a);

        excursion = pi * curv;
        
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
        
        %        if t(i) > 1.5
%         fprintf('t=%.4f, mode=%s, omic=%g, sum_M=%g, spinTr=%g, psi=%g, Psi=%g\n', ...
%             t(i), spinning_Tr.mode, ...
%             myEllipsoid.omic(3), M_total(3), M_spinning(3), ...
%             spinning_Tr.increment, spinning_Tr.history);
        %        end
        if mod(i, 10000) == 0
        fprintf('t=%.4f, mode=%s, omic=%g, sum_M=%g, spinTr=%g, psi=%g, Psi=%g\n', ...
            t(i), spinning_Tr.mode, ...
            myEllipsoid.omic(3), M_total(3), M_spinning(3), ...
            spinning_Tr.increment, spinning_Tr.history);
        end            
        
        spinningTr_holder(i+1) = spinning_Tr.totalFriction;
        
        staticSlack_holder(i+1) = spinning_Tr.slackStatic;
        kineticSlack_holder(i+1) = spinning_Tr.slackKinetic;
        spinningHistory_holder(i+1) = spinning_Tr.history;
        
        
        
    end
    kineticEnergy_holder(i+1) = myEllipsoid.getKineticEnergy;
    
    
    FS = 36;
    LW = 4;
    
    if mySimulation.generateMovie == true &&  mod(i, floor(length(t)/mySimulation.movieLoops)) == 0
        %    if mySimulation.generateMovie == true
        
        %            subplot(2,2,ii)
%        wall1.drawPlaneColor(-0.5, 0.5, 1);        
 %       wall2.drawPlaneColor(-0.5, 0.5, 1.2);
        
        
        
        
        myEllipsoid.drawReferenceFrame('r');
        hold on
        myPlane.drawPlaneColor(-2.5,2.5, 0.8);
        grid off
        
        myEllipsoid.drawEllipsoid;
        grid off
        
        view(74, 14) % front view
        axis equal
        xlim([-0.3, 0.3]);
        ylim([-0.3, 0.3]);  %
        zlim([0, 1.2]);
        textHdl = text(min(xlim), min(ylim), max(zlim)-0.1, ...
            sprintf('time=%.2f sec, \n \\omega=%.3e rad/s \n T_{\\psi} = %.2e Nm', ...
            t(i+1), myEllipsoid.omic(3), spinning_Tr.elasticComponent), ...
            'FontSize', FS);
        set(gca, 'linewidth', LW);
        set(gca, 'FontSize', FS-3);
        myContact.CF_curr_global.drawContactFrame(myContact.CP_curr_global, 'g', myEllipsoid.c*0.8);
        grid off
        
        hdl = gca;
        hdl.XTick = [];
        hdl.YTick = [];
        hdl.ZTick = [];
        
        hold off
        mySimulation.writeCurrentFrame;
        
        
        hold off
    end

    
    
    
    
    omic_holer(i+1) = myEllipsoid.omic(3);
    
    psi_holder(i+1) = spinning_Tr.increment;
    dampingComponent_holder(i+1) = spinning_Tr.plasticComponent;
    
end

if mySimulation.generateMovie == true
    mySimulation.writeMovies(mySimulation.name);
end

% vidObj = VideoWriter(strcat('debug', '.avi'));
% vidObj.Quality = 100;
% open(vidObj);
% writeVideo(vidObj, debugFrames);
% close(vidObj);


%%
% figure;
% LW = 1;  % line width
% FS = 20; % font size
% MS = 8; % marker size
% % Tstart = 0;
% % Tend = Tend;
% %
% subplot(5,1,1);
% makePlot(t, omic_holer*1e4, '', '\omega (\times 10^{-4}) (rad/s)', ' ', LW, FS);
% xlim([debug_start,debug_end]);
% 
% subplot(5,1,2);
% 
% myPlot = makePlot(t(product_holder>1), product_holder(product_holder>1), '', '$$u^{T}\bar{u}$$', ' ', LW, FS);
% myPlot.Marker = 'o'; myPlot.LineStyle = 'none'; myPlot.MarkerSize = MS; myPlot.MarkerEdgeColor = 'r';
% hold on
% myPlot = makePlot(t(product_holder<=1), product_holder(product_holder<=1), '', '$$u^{T}\bar{u}$$', ' ', LW, FS);
% myPlot.Marker = 'o'; myPlot.LineStyle = 'none'; myPlot.MarkerSize = MS; myPlot.MarkerEdgeColor = 'b';
% legend('u^Tu > 1', 'u^Tu < 1');
% xlim([debug_start,debug_end]);
% 
% subplot(5,1,3);
% myPlot = makePlot(t(cospsiDiff_holder>0), cospsiDiff_holder(cospsiDiff_holder>0), '', '$$u^{T}\bar{u} - 1$$', ' ', LW, FS);
% myPlot.Marker = 'o'; myPlot.LineStyle = 'none'; myPlot.MarkerSize = MS; myPlot.MarkerEdgeColor = 'r';
% hold on
% myPlot = makePlot(t(cospsiDiff_holder<=0), cospsiDiff_holder(cospsiDiff_holder<=0), '', '$$u^{T}\bar{u} - 1$$', ' ', LW, FS);
% myPlot.Marker = 'o'; myPlot.LineStyle = 'none'; myPlot.MarkerSize = MS; myPlot.MarkerEdgeColor = 'b';
% legend('u^Tu - 1 > 0', 'u^Tu - 1 \leq 0');
% xlim([debug_start,debug_end]);
% 
% subplot(5,1,4);
% myPlot = makePlot(t(psi_holder==0), cospsiDiff_holder(psi_holder==0), '', '$$\psi(rad)$$', ' ', LW, FS);
% myPlot.Marker = 'o'; myPlot.LineStyle = 'none'; myPlot.MarkerSize = MS; myPlot.MarkerEdgeColor = 'r';
% hold on
% myPlot = makePlot(t(psi_holder~=0), cospsiDiff_holder(psi_holder~=0), '', '$$\psi(rad)$$', ' ', LW, FS);
% myPlot.Marker = 'o'; myPlot.LineStyle = 'none'; myPlot.MarkerSize = MS; myPlot.MarkerEdgeColor = 'b';
% legend('psi \neq 0', 'psi = 0');
% xlim([debug_start,debug_end]);
% 
% subplot(5,1,5);
% myPlot = makePlot(t(dampingComponent_holder==0), cospsiDiff_holder(dampingComponent_holder==0), '', '$$T_D{Nm}$$', ' ', LW, FS);
% myPlot.Marker = 'o'; myPlot.LineStyle = 'none'; myPlot.MarkerSize = MS; myPlot.MarkerEdgeColor = 'r';
% hold on
% myPlot = makePlot(t(dampingComponent_holder~=0), cospsiDiff_holder(dampingComponent_holder~=0), '', '$$T_D(Nm)$$', ' ', LW, FS);
% myPlot.Marker = 'o'; myPlot.LineStyle = 'none'; myPlot.MarkerSize = MS; myPlot.MarkerEdgeColor = 'b';
% legend('T_D \neq 0', 'T_D = 0');
% xlim([debug_start,debug_end]);




%
%
% subplot(2,4,2);
% makePlot(t, rollingTr_holder, '', 'total rolling torque(Nm)', '', LW, FS);
% xlim([Tstart,Tend]);
%
% subplot(2,4,3);
% makePlot(t, curv_holder, '', 'curvature(1/m)', '', LW, FS);
% xlim([Tstart,Tend]);
%
% subplot(2,4,4);
% makePlot(t, dampingCr_holder, '', 'damping coefficient(Nms/rad)', '', LW, FS);
% xlim([Tstart,Tend]);
%
% subplot(2,4,5);
% makePlotYY(t, staticSlack_holder, t, rollinghistory_holder, '', 'static slack (rad)', 'rolling history','', LW, FS);
% xlim([Tstart,Tend]);
%
% subplot(2,4,6);
% myPlot = makePlot(t, mode_holder, '', 'rolling mode', 'static mode == 1', LW, FS);
% myPlot.Marker = 'o'; myPlot.LineStyle = 'none'; myPlot.MarkerSize = MS;
% xlim([Tstart,Tend]);
%
% subplot(2,4,7);
% makePlotYY(t, normalForce_holder, t, penetrationDepth_holder, '', 'normal force(N)', 'penetration(m)', '', LW, FS);
% xlim([Tstart,Tend]);
