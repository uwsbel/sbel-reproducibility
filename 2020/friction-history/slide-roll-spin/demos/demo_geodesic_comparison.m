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

% geodesic comparison of sphere rolling on a plane

%clc
%clear all
%close all

%init_vx = 0.3*scale;
%init_vy = 0.4*scale;

scale = 0.5;
Tend = 2.4*scale;

init_vx = 0;
init_vy = 1*scale;


useGeodesic = true; % whether to use geodesic or cut the corner


% bowling ball parameters
mass = 5;
radius = 0.2;
inertia = 0.4*mass*radius^2;  % change inertia!!! should be 0.4mR^2
matrixM = mass*eye(3,3); % mass matrix
matrixJ = inertia*eye(3,3); % interita

% look up bowling isle statistics, a lot smaller friction coefficient
gravity = 9.8;
mu_s = 0.25;
mu_k = 0.2;

scenario = 'sphere_rolling_plane';
dt = 1e-4; t = 0:dt:Tend;

% tech_report = false;
% tech_report_dir = '/Users/lulu/Documents/TechReports/Friction3D/Images/';

% initialize kinematics array
pos = zeros(length(t),3); velo = zeros(length(t),3); acc = zeros(length(t),3);
eulerPar = zeros(length(t),4); omic = zeros(length(t),3); omic_dot = zeros(length(t),3);

% sliding friction
Ke = 1e5;
%Kd = 2*sqrt(mass*Ke);

Kd = 2*sqrt(mass*Ke);
slide_slack_s = mu_s*mass*gravity/Ke;
slide_slack_k = mu_k*mass*gravity/Ke;
Sij = 0;
slide_mode = 's';

% rolling friction model
%Kr = 2000;

Kr = 3000;
%Kr = 700; %oda experiment
C_cr = 2*sqrt(inertia*Kr);
eta_r = 1.5;
Cr = eta_r * C_cr;
rolling_mode = 's'; % initialze rolling mode
rolling_slack_s = slide_slack_s/(2*radius);  % slack for sphere on plane, static
rolling_slack_k = slide_slack_k/(2*radius); % slack for sphere on plane, kinetic
rolling_torque = 0; rolling_torque_array = zeros(length(t),1); % initialize rolling torque
rolling_history = 0; rolling_history_array = zeros(length(t),1); % relative rolling history
excursion = 0;


Fe_array = zeros(length(t),1); Fe_array(1) = 0;   % elastic part of friction force magnitude
Te_array = zeros(length(t),1); Te_array(1) = 0;   % elastic part of rolling torque magnitude
Fd_array = zeros(length(t),1); Fd_array(1) = 0;   % plastic part of friction force magnitude
Td_array = zeros(length(t),1); Td_array(1) = 0;   % plastic part of rolling torque magnitude
relative_slide = zeros(length(t),1); relative_slide(1) = 0;
relative_roll  = zeros(length(t),1); relative_roll(1)  = 0;
angular_pos = zeros(length(t),1);

isStaticSliding = zeros(length(t),1);
isStaticRolling = zeros(length(t),1);
isStaticSliding(1) = true;
isStaticRolling(1) = true;

% spinning friction


% initial condition
velo(1,2) = init_vy;
velo(1,1) = init_vx;


pos(1,:) = [0 0 radius];
eulerPar(1,:) = [1, 0, 0 , 0];
orientA_prev = getAfromP(eulerPar(1,:));
omic(1,3) = 0;

relative_spin = [0];



contactPointArray = [];


% get contact point, project onto the plane
contactPoint_prev = [pos(1,1) pos(1,2) 0]';
sCP_prev = contactPoint_prev - pos(1,:)';
sCP_prev_bar = orientA_prev'*sCP_prev;



% get velocity at contact point
sCP_curr = pos(1,:)' + orientA_prev*sCP_prev_bar; % contact point in global reference frame
vCP = velo(1,:)' + tensor(omic(1,:))*sCP_prev_bar; % get relative velocity
vCP(3) = 0; % project relative velocity onto contact plane
contactFrame_u_prev = vCP/norm(vCP);  % normalize to get contact frame in u direction

% previous contact frame
contactFrame_n_prev = [0;0;1];
contactFrame_w_prev = cross(contactFrame_n_prev, contactFrame_u_prev);

Ef_array = [];

for i = 1:length(t)-1
    
    if abs(t(i) - 0.6615) < 1e-5
        fprintf('time is %fsec\n', t(i));
    end
    
    % get orientation matrix from euler parameter
    orientA_curr = getAfromP(eulerPar(i,:));
    
    % get previous contact frame in current global coordinate
    contactFrame_n0 = orientA_curr * contactFrame_n_prev;
    contactFrame_u0 = orientA_curr * contactFrame_u_prev;
    contactFrame_w0 = orientA_curr * contactFrame_w_prev;
    
    
    % find current normal contact frame
    contactFrame_n1 = [0;0;1];
    
    % find current tangential contact frame using optimization method
    [contactFrame_u1, contactFrame_w1] = MaxCoordinate(contactFrame_u0, contactFrame_w0, contactFrame_n1);
    % rotation from n1 to n0, aligning normal contact frame
    rot_u = cross(contactFrame_n1, contactFrame_n0);
    
    if norm(rot_u) < 1e-8
        delta_psi = 0;
    else
        
        rot_u = rot_u/norm(rot_u);
        rot_Xi = acos(contactFrame_n1'*contactFrame_n0);

        contactFrame_u1bar = rotationAboutAxis(contactFrame_u1, rot_u, rot_Xi);
        contactFrame_w1bar = rotationAboutAxis(contactFrame_w1, rot_u, rot_Xi);

        delta_psi = acos(contactFrame_u1bar'*contactFrame_u0);
    end
    
    % get contact point, project onto the plane
    contactPoint_curr = [pos(i,1) pos(i,2) 0]';
    sCP_curr = contactPoint_curr - pos(i,:)';
    
    sCP_curr_bar = orientA_curr'*sCP_curr;
    sCP_prev_track = pos(i,:)' + orientA_curr*sCP_prev_bar;
    
    
        pi = geodesic(sCP_prev_track, contactPoint_curr, contactFrame_n1, pos(i,:)', useGeodesic);  % body i for sphere
    
    pj = contactPoint_curr - contactPoint_prev; % body j for the ground
    
    contactPoint_prev = contactPoint_curr;
    diff_pi_analytical = norm(pi) - norm(omic*dt*radius);
    
    sCP_prev_bar = sCP_curr_bar;
    
    % sliding friction model
    
    delta_Sij = pj(1:2) - pi(1:2);  % ground minus the sphere
    Sij = Sij + delta_Sij;
    
    slide_slack = norm(Sij);
    if slide_mode == 's'
        alpha_s = slide_slack/slide_slack_s;
        if alpha_s > 1
            Sij = Sij/alpha_s;
            slide_mode = 'k';
            Kd = 0;
        end
    else
        alpha_k = slide_slack/slide_slack_k;
        if alpha_k > 1
            %Sij = Sij/alpha_k;
            
            Sij = Sij/norm(Sij)*slide_slack_k;
            Kd = 0;
        else
            slide_mode = 's';
        end
    end
    
    
    % rolling friction kinematics
    excursion = pi/radius;
    rolling_history = rolling_history + excursion;
    
    % rolling friction mode and magnitude
    if rolling_mode == 's'
        alpha_s = norm(rolling_history)/rolling_slack_s;
%        torque_damping = Cr * excursion/dt;
        if alpha_s > 1
            rolling_history = rolling_history/alpha_s;
            rolling_mode = 'k';
%            torque_damping = 0;
        end
    else
        alpha_k = norm(rolling_history)/rolling_slack_k;
        if alpha_k > 1
            rolling_history = rolling_history/norm(rolling_history)*rolling_slack_k;
%            torque_damping = 0;
        else
            rolling_mode = 's';
%            torque_damping = Cr * excursion/dt;
        end
    end

    
    
    
    
    contactPointArray = [contactPointArray; contactPoint_curr(1) contactPoint_curr(2)];
    
    
    if abs(delta_psi) < 1e-5
        spin_fr_dir = [0;0;0];
    else
        
        
        % determine spin friction direction using cross product
        spin_fr_dir = cross(contactFrame_u1bar, contactFrame_u0)/norm(cross(contactFrame_u1bar,contactFrame_u0));
    end
    % TO DO
    % CHECK IF THE SPINNING ANGLE IS THE SAME AMONG DIFFERENT
    % COORDINATE PROJECTION (X AND Y)
    % pure spinning only but arbitrary reference frame, should be the
    % same spinning angle
    
    
    contactFrame_n_prev = contactFrame_n1;
    contactFrame_u_prev = contactFrame_u1;
    contactFrame_w_prev = contactFrame_w1;
    
    % gravity force
    F_gravity = mass * [0; 0; -gravity];
    
    % contact force
    F_normal = -F_gravity;
    
    % sliding friction force
        Ef = Ke * Sij;
        if slide_mode == 's'
            Kd = 2*sqrt(mass * Ke);
            Df = Kd * delta_Sij/dt;
            
            isStaticSliding(i+1) = true;
            
        else
            Df = 0;
            isStaticSliding(i+1) = false;
        end
    
    Fr_sliding = -(Ef + Df);
    
    Ef_array = [Ef_array; norm(Ef)];
    
    % rolling torque direction
    rolling_torque_dir = cross(contactPoint_curr - pos(i,:)', pi);
    
    if norm(rolling_torque_dir) > 1e-8
        rolling_torque_dir_u = rolling_torque_dir/norm(rolling_torque_dir);
    else
        rolling_torque_dir_u = [0;0;0];
    end
    
    % normalized radius
    radius_normalized = (contactPoint_curr - pos(i,:)')/norm(contactPoint_curr - pos(i,:)');
    % rolling friction torque
    Te = rolling_history * Kr;
    
    if rolling_mode == 's'
        torque_damping = Cr * excursion/dt;
        isStaticRolling(i+1) = true;
        
    else
        torque_damping = zeros(size(excursion));
        isStaticRolling(i+1) = false;
        
    end
    
    rolling_torque = cross(radius_normalized, Kr * rolling_history + torque_damping);

    
    % sliding friction torque wrt center of mass
    M_fr = tensor(sCP_curr_bar) * orientA_curr' * [Fr_sliding;0];
         M_fr = cross(sCP_curr, [Fr_sliding;0]);

    % spinning torque, opposite direction
    K_psi = 1e6;
    %    M_spin_mag = abs(delta_psi*K_psi);
    %    M_spin_max = 10;
    
    %    M_spin_mag = min(M_spin_max, M_spin_mag);
    
    M_spin = [0;0;0];
    
    
    % sum all the forces and moments
    F_total = F_gravity + F_normal + [Fr_sliding;0];
    
    
    % sum all the moments
    M_total = M_fr + M_spin - tensor(omic(i,:)) * matrixJ * omic(i,:)' + rolling_torque;
    
    % update acceleration at new timestep
    acc_new = matrixM\F_total;
    omic_dot_new = matrixJ\M_total;
    
    
    % Euler update of velocity
    omic_dot(i+1,:) = omic_dot_new';
    acc(i+1,:) = acc_new';
    omic(i+1,:) = omic(i,:) + omic_dot_new'*dt;
    velo(i+1,:) = velo(i,:) + acc_new'*dt;
    pos(i+1,:) = pos(i,:) + velo(i+1,:)*dt;
    
    
    eulerPar(i+1,:) = eulerParUpdate(omic(i,:), eulerPar(i,:), dt);
    
    
    orientA_prev = orientA_curr;
%     if t(i) > 0.99 && t(i) < 1
%        fprintf('t=%.4f, Kd=%g, Sij=%g, d_Sij=%g, Fr(%s):%g=%g+%g, Tr(%s):%g=%g+%g, theta_ij = %g\n', ...
%                 t(i), Kd, norm(Sij), norm(delta_Sij), slide_mode, norm(Fr_sliding), norm(Ef), norm(Df), rolling_mode, norm(rolling_torque), norm(Te), norm(torque_damping), norm(rolling_history));
 %   end
    Fe_array(i+1) = norm(Ef);   % elastic part of friction force magnitude
    Te_array(i+1) = norm(Te);   % elastic part of rolling torque magnitude
    Fd_array(i+1) = norm(Df);   % plastic part of friction force magnitude
    Td_array(i+1) = norm(torque_damping);   % plastic part of rolling torque magnitude

    
    
    relative_slide(i+1) = norm(Sij);
    relative_roll(i+1)  = norm(rolling_history);
    angular_pos(i+1) = acos(orientA_curr(3,3));
    
    
    
end


woGeo_pos = pos;
woGeo_angularPos = angular_pos;
woGeo_velo = velo;
woGeo_omic = omic;
woGeo_Fe = Fe_array;
woGeo_Te = Te_array;
woGeo_relativeSlide = relative_slide;
woGeo_relativeRoll = relative_roll;
woGeo_Fd = Fd_array;
woGeo_Td = Td_array;
woIsStaticSliding = isStaticSliding;
woIsStaticRolling = isStaticRolling;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% do it with geodesic  %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

useGeodesic = true; % whether to use geodesic or cut the corner
% look up bowling isle statistics, a lot smaller friction coefficient
gravity = 9.8;
mu_s = 0.25;
mu_k = 0.2;

scenario = 'sphere_rolling_plane';
dt = 1e-4; t = 0:dt:Tend;

% tech_report = false;
% tech_report_dir = '/Users/lulu/Documents/TechReports/Friction3D/Images/';

% initialize kinematics array
pos = zeros(length(t),3); velo = zeros(length(t),3); acc = zeros(length(t),3);
eulerPar = zeros(length(t),4); omic = zeros(length(t),3); omic_dot = zeros(length(t),3);

% sliding friction
Ke = 1e5;
slide_slack_s = mu_s*mass*gravity/Ke;
slide_slack_k = mu_k*mass*gravity/Ke;
Sij = 0;
slide_mode = 's';

% rolling friction model
%Kr = 700; %oda experiment
C_cr = 2*sqrt(inertia*Kr);
rolling_mode = 's'; % initialze rolling mode
rolling_slack_s = slide_slack_s/(2*radius);  % slack for sphere on plane, static
rolling_slack_k = slide_slack_k/(2*radius); % slack for sphere on plane, kinetic
rolling_torque = 0; rolling_torque_array = zeros(length(t),1); % initialize rolling torque
rolling_history = 0; rolling_history_array = zeros(length(t),1); % relative rolling history
excursion = 0;


Fe_array = zeros(length(t),1); Fe_array(1) = 0;   % elastic part of friction force magnitude
Te_array = zeros(length(t),1); Te_array(1) = 0;   % elastic part of rolling torque magnitude
Fd_array = zeros(length(t),1); Fd_array(1) = 0;   % plastic part of friction force magnitude
Td_array = zeros(length(t),1); Td_array(1) = 0;   % plastic part of rolling torque magnitude
relative_slide = zeros(length(t),1); relative_slide(1) = 0;
relative_roll  = zeros(length(t),1); relative_roll(1)  = 0;
angular_pos = zeros(length(t),1);
isStaticSliding = zeros(length(t),1);
isStaticRolling = zeros(length(t),1);
isStaticSliding(1) = true;
isStaticRolling(1) = true;


% spinning friction


% initial condition
%init_vx = 0.3*scale;
%init_vy = 0.4*scale;
init_vx = 0;
init_vy = 1*scale;

velo(1,1) = init_vx;
velo(1,2) = init_vy;
%velo(1,2) = 1.5;


pos(1,:) = [0 0 radius];
eulerPar(1,:) = [1, 0, 0 , 0];
orientA_prev = getAfromP(eulerPar(1,:));
omic(1,3) = 0;

relative_spin = [0];



contactPointArray = [];


% get contact point, project onto the plane
contactPoint_prev = [pos(1,1) pos(1,2) 0]';
sCP_prev = contactPoint_prev - pos(1,:)';
sCP_prev_bar = orientA_prev'*sCP_prev;



% get velocity at contact point
sCP_curr = pos(1,:)' + orientA_prev*sCP_prev_bar; % contact point in global reference frame
vCP = velo(1,:)' + tensor(omic(1,:))*sCP_prev_bar; % get relative velocity
vCP(3) = 0; % project relative velocity onto contact plane
contactFrame_u_prev = vCP/norm(vCP);  % normalize to get contact frame in u direction

% previous contact frame
contactFrame_n_prev = [0;0;1];
contactFrame_w_prev = cross(contactFrame_n_prev, contactFrame_u_prev);

Ef_array = [];


for i = 1:length(t)-1
    
    if abs(t(i) - 1.09) < 1e-5
        fprintf('time is %fsec\n', t(i));
    end
    
    % get orientation matrix from euler parameter
    orientA_curr = getAfromP(eulerPar(i,:));
    
    % get previous contact frame in current global coordinate
    contactFrame_n0 = orientA_curr * contactFrame_n_prev;
    contactFrame_u0 = orientA_curr * contactFrame_u_prev;
    contactFrame_w0 = orientA_curr * contactFrame_w_prev;
    
    
    % find current normal contact frame
    contactFrame_n1 = [0;0;1];
    
    % find current tangential contact frame using optimization method
    [contactFrame_u1, contactFrame_w1] = MaxCoordinate(contactFrame_u0, contactFrame_w0, contactFrame_n1);
    % rotation from n1 to n0, aligning normal contact frame
    rot_u = cross(contactFrame_n1, contactFrame_n0);
    
    if norm(rot_u) < 1e-8
        delta_psi = 0;
    else
        
        rot_u = rot_u/norm(rot_u);
        rot_Xi = acos(contactFrame_n1'*contactFrame_n0);

        contactFrame_u1bar = rotationAboutAxis(contactFrame_u1, rot_u, rot_Xi);
        contactFrame_w1bar = rotationAboutAxis(contactFrame_w1, rot_u, rot_Xi);

        delta_psi = acos(contactFrame_u1bar'*contactFrame_u0);
    end
    
    % get contact point, project onto the plane
    contactPoint_curr = [pos(i,1) pos(i,2) 0]';
    sCP_curr = contactPoint_curr - pos(i,:)';
    
    sCP_curr_bar = orientA_curr'*sCP_curr;
    sCP_prev_track = pos(i,:)' + orientA_curr*sCP_prev_bar;
    
    
        pi = geodesic(sCP_prev_track, contactPoint_curr, contactFrame_n1, pos(i,:)', useGeodesic);  % body i for sphere
    
    pj = contactPoint_curr - contactPoint_prev; % body j for the ground
    
    contactPoint_prev = contactPoint_curr;
    diff_pi_analytical = norm(pi) - norm(omic*dt*radius);
    
    sCP_prev_bar = sCP_curr_bar;
    
    % sliding friction model
    
    delta_Sij = pj(1:2) - pi(1:2);  % ground minus the sphere
    Sij = Sij + delta_Sij;
    
    slide_slack = norm(Sij);
    if slide_mode == 's'
        alpha_s = slide_slack/slide_slack_s;
        if alpha_s > 1
            Sij = Sij/alpha_s;
            slide_mode = 'k';
            Kd = 0;
        end
    else
        alpha_k = slide_slack/slide_slack_k;
        if alpha_k > 1
            %Sij = Sij/alpha_k;
            
            Sij = Sij/norm(Sij)*slide_slack_k;
            Kd = 0;
        else
            slide_mode = 's';
            Kd = 1.5*sqrt(mass*Ke);
        end
    end
    
    
    % rolling friction kinematics
    excursion = pi/radius;
    rolling_history = rolling_history + excursion;
    
    % rolling friction mode and magnitude
    if rolling_mode == 's'
        alpha_s = norm(rolling_history)/rolling_slack_s;
%        torque_damping = Cr * excursion/dt;
        if alpha_s > 1
            rolling_history = rolling_history/alpha_s;
            rolling_mode = 'k';
%            torque_damping = 0;
        end
    else
        alpha_k = norm(rolling_history)/rolling_slack_k;
        if alpha_k > 1
            rolling_history = rolling_history/norm(rolling_history)*rolling_slack_k;
%            torque_damping = 0;
        else
            rolling_mode = 's';
%            torque_damping = Cr * excursion/dt;
        end
    end

    
    
    
    
    contactPointArray = [contactPointArray; contactPoint_curr(1) contactPoint_curr(2)];
    
    
    if abs(delta_psi) < 1e-5
        spin_fr_dir = [0;0;0];
    else
        
        
        % determine spin friction direction using cross product
        spin_fr_dir = cross(contactFrame_u1bar, contactFrame_u0)/norm(cross(contactFrame_u1bar,contactFrame_u0));
    end
    % TO DO
    % CHECK IF THE SPINNING ANGLE IS THE SAME AMONG DIFFERENT
    % COORDINATE PROJECTION (X AND Y)
    % pure spinning only but arbitrary reference frame, should be the
    % same spinning angle
    
    
    contactFrame_n_prev = contactFrame_n1;
    contactFrame_u_prev = contactFrame_u1;
    contactFrame_w_prev = contactFrame_w1;
    
    % gravity force
    F_gravity = mass * [0; 0; -gravity];
    
    % contact force
    F_normal = -F_gravity;
    
    % sliding friction force
        Ef = Ke * Sij;
        if slide_mode == 's'
            Kd = 1.5*sqrt(mass*Ke);
            Df = Kd * delta_Sij/dt;
            isStaticSliding(i) = true;
            
        else
            Df = 0;
            isStaticSliding(i) = false;

        end
    
    Fr_sliding = -(Ef + Df);
    
    Ef_array = [Ef_array; norm(Ef)];
    
    % rolling torque direction
    rolling_torque_dir = cross(contactPoint_curr - pos(i,:)', pi);
    
    if norm(rolling_torque_dir) > 1e-8
        rolling_torque_dir_u = rolling_torque_dir/norm(rolling_torque_dir);
    else
        rolling_torque_dir_u = [0;0;0];
    end
    
    % normalized radius
    radius_normalized = (contactPoint_curr - pos(i,:)')/norm(contactPoint_curr - pos(i,:)');
    % rolling friction torque
    Te = rolling_history * Kr;
    
    if rolling_mode == 's'
        torque_damping = Cr * excursion/dt;
        isStaticRolling(i) = true;
        
    else
        torque_damping = zeros(size(excursion));
        isStaticRolling(i) = false;

    end
    
    rolling_torque = cross(radius_normalized, Kr * rolling_history + torque_damping);

    
    % sliding friction torque wrt center of mass
%    M_fr = tensor(sCP_curr_bar) * orientA_curr' * [Fr_sliding;0];
    
    M_fr = cross(sCP_curr, [Fr_sliding;0]);
    
%     orientA_curr *  sCP_curr_bar
%     sCP_curr
    
    % spinning torque, opposite direction
    K_psi = 1e6;
    %    M_spin_mag = abs(delta_psi*K_psi);
    %    M_spin_max = 10;
    
    %    M_spin_mag = min(M_spin_max, M_spin_mag);
    
    M_spin = [0;0;0];
    
%    M_spin
    % sum all the forces and moments
    F_total = F_gravity + F_normal + [Fr_sliding;0];
    
    
    % sum all the moments
    M_total = M_fr + M_spin - tensor(omic(i,:)) * matrixJ * omic(i,:)' + rolling_torque;
    
    % update acceleration at new timestep
    acc_new = matrixM\F_total;
    omic_dot_new = matrixJ\M_total;
    
    
    % Euler update of velocity
    omic_dot(i+1,:) = omic_dot_new';
    acc(i+1,:) = acc_new';
    omic(i+1,:) = omic(i,:) + omic_dot_new'*dt;
    velo(i+1,:) = velo(i,:) + acc_new'*dt;
    pos(i+1,:) = pos(i,:) + velo(i+1,:)*dt;
    
    
    eulerPar(i+1,:) = eulerParUpdate(omic(i,:), eulerPar(i,:), dt);
    
    
    orientA_prev = orientA_curr;
%     if t(i) > 0.99 && t(i) < 1
%        fprintf('t=%.4f, Kd=%g, Sij=%g, d_Sij=%g, Fr(%s):%g=%g+%g, Tr(%s):%g=%g+%g, theta_ij = %g\n', ...
%                 t(i), Kd, norm(Sij), norm(delta_Sij), slide_mode, norm(Fr_sliding), norm(Ef), norm(Df), rolling_mode, norm(rolling_torque), norm(Te), norm(torque_damping), norm(rolling_history));
%    end
    Fe_array(i+1) = norm(Ef);   % elastic part of friction force magnitude
    Te_array(i+1) = norm(Te);   % elastic part of rolling torque magnitude
    Fd_array(i+1) = norm(Df);   % elastic part of friction force magnitude
    Td_array(i+1) = norm(torque_damping);   % elastic part of rolling torque magnitude

    relative_slide(i+1) = norm(Sij);
    relative_roll(i+1)  = norm(rolling_history);
    angular_pos(i+1) = acos(orientA_curr(3,3));
    
    
    
end


wGeo_pos = pos;
wGeo_angularPos = angular_pos;
wGeo_velo = velo;
wGeo_omic = omic;
wGeo_Fe = Fe_array;
wGeo_Te = Te_array;
wGeo_Fd = Fd_array;
wGeo_Td = Td_array;
wGeo_relativeSlide = relative_slide;
wGeo_relativeRoll = relative_roll;
wIsStaticSliding = isStaticSliding;
wIsStaticRolling = isStaticRolling;
