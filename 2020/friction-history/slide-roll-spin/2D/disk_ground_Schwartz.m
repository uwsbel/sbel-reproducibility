clc
clear all
close all

% car wheel parameters

% oda paper
% rho = 1800;
% radius = 4e-3;
% mass = rho*4/3*pi*radius^3;
% inertia = 0.5*mass*radius^2;

eta_t = 1;
FontSize = 36;
LineWidth = 4;


mass = 5;
radius = 0.2;
inertia = 0.5*mass*radius^2;


gravity = 9.8;
mu_s = 0.25;
mu_k = 0.2;

scenario = 'disk_rolling';
solver = 'explicit';
Tend = 5; dt = 1e-4; t = 0:dt:Tend;


% initialize kinematics array
pos = zeros(length(t),1); velo = zeros(length(t),1); acc = zeros(length(t),1);
theta = zeros(length(t),1); omic = zeros(length(t),1); alpha = zeros(length(t),1);

% sliding/rolling friction array
Ef_array = zeros(length(t),1); Df_array = zeros(length(t),1);
Te_array = zeros(length(t),1); Td_array = zeros(length(t),1);

% initial condition
velo(1) = 0.5; omic(1) = 0;

% slide/roling friction parameter
% rolling friction parameters
%eta = 0.3;
%Kr = eta*R^2*Ke;   % rolling friction stiffness


% sliding friction model 
Ke = 1e5; % spring stiffness
%Ke = 4e7; % from Oda paper
sliding_mode = 'k'; % sliding friction mode
Sij = 10; Sij_array = zeros(length(t),1); Sij_array(1) = 0;  % relative displacement
delta_Sij = 0; dSij_array = zeros(length(t),1); dSij_array(1) = 0; % relative displacement increment 
sliding_friction = 0; % initialzie sliding friction
sliding_slack_s = mu_s*mass*gravity/Ke; % static slack
sliding_slack_k = mu_k*mass*gravity/Ke; % kinetic slack

% rolling friction model
rolling_fric_coefficient = 0.05;
rolling_torque = 0;

for i = 1:length(t)-1
    
    if solver == 'explicit'
        acc(i+1) = -sliding_friction/mass;
        alpha(i+1) = (sliding_friction*radius - rolling_torque)/inertia ;  % don't simplify the equation, use inertia instead of mass*radius, confusing...
    end
            
    % explicit kinematics update
    velo(i+1) = velo(i) + dt*acc(i+1);
    pos(i+1) = pos(i) + dt*velo(i+1);
    omic(i+1) = omic(i) + dt*alpha(i+1);
    theta(i+1) = theta(i) + dt*omic(i+1);
    
    pj = theta(i+1)*radius - theta(i)*radius;
    pi = pos(i+1) - pos(i);
    
    % relative slide and roll
    delta_Sij = pi-pj;  % ground minus the sphere
    Sij = Sij + delta_Sij;
    
    slide_slack = abs(Sij);
    Sij_old = Sij;

    % sliding friction mode and magnitude
    if sliding_mode == 's'
        Kd = 2*sqrt(mass*Ke);
        alpha_s = slide_slack/sliding_slack_s;
        if alpha_s > 1
            Sij = Sij/alpha_s;
            sliding_mode = 'k';
            Kd = 0;
        end
    else
        alpha_k = slide_slack/sliding_slack_k;
        if alpha_k > 1
            Sij = sign(Sij)*sliding_slack_k;
            Kd = 0;
        else
            sliding_mode = 's';
            Kd = eta_t*2*sqrt(mass*Ke);
        end
    end
    
    % rolling friction mode and magnitude
    if omic(i+1) >= 0
        rolling_torque =  rolling_fric_coefficient * mass *gravity * radius;
    else
        if omic(i+1) < 0
        rolling_torque = -  rolling_fric_coefficient * mass *gravity * radius;
        else
            rolling_torque = 0;
        end
    
    end
    
%    fprintf('time=%.5f, rolling_history=%g, rolling_slack_k=%g, rolling_slack_s=%g, Td=%g, omic=%g\n', t(i), rolling_history, rolling_slack_k, rolling_slack_s, torque_damping, omic(i+1));
    
    Ef = Ke * Sij;
    if sliding_mode == 's'
        Df = eta_t*Kd * delta_Sij/dt;
%         delta_Sij
%         Df
    else
        Df = 0;
    end
    
    sliding_friction = Ef + Df;
    
    Ef_array(i+1) = Ef;
    Df_array(i+1) = Df;
    Sij_array(i+1) = Sij;
    dSij_array(i+1) = delta_Sij;
    rolling_torque_array(i+1) = rolling_torque;
    
%    fprintf('time=%.5f, slide_mode=%s, excursion=%g\n', t(i), sliding_mode, excursion )

end


FontSize = 22;
LineWidth = 2;

figure('units','normalized','outerposition',[0 0 1 1]);
t_start = Tend - 0.001;

% subplot(2,3,1)
% makePlot(t,pos,'time (ms)','position of CM (m)',sprintf('%s solver', solver), LineWidth, FontSize);
% subplot(2,3,2)
% makePlot(t,velo,'time (ms)','velocity of CM (m/s)',sprintf('\\mu_s=%.2f,\\mu_k=%.2f', mu_s, mu_k), LineWidth, FontSize);
% subplot(2,3,3)
% makePlot(t,acc,'time (ms)','acceleration of CM (m/s^2)', '', LineWidth, FontSize);
% 
% subplot(2,3,4)
% makePlot(t,theta,'time (ms)','angular position (rad)', '', LineWidth, FontSize);
% subplot(2,3,5)
% makePlot(t,omic,'time (ms)','angular velocity (rad/s)', '', LineWidth, FontSize);
% subplot(2,3,6)
% makePlot(t,alpha,'time (ms)','angular acceleration (rad/s^2)', '', LineWidth, FontSize);
subplot(2,2,1)
makePlotYY(t,pos,t,theta ,'time (sec)', 'position of CM (m)', 'angular position (rad)', '2D', LineWidth, FontSize)
xlim([t_start, Tend]);

subplot(2,2,2)
makePlotYY(t,velo,t,omic ,'time (sec)', 'velocity of CM (m/s)', 'angular velocity (rad/s)', sprintf('\\mu_s=%.2f,\\mu_k=%.2f', mu_s, mu_k), LineWidth, FontSize)
xlim([t_start, Tend]);

subplot(2,2,3)
makePlotYY(t,acc,t,alpha ,'time (sec)', 'acceleration of CM (m/s^2)', 'angular acceleration (rad/s^2)', sprintf('dt=%g', dt), LineWidth, FontSize)
xlim([t_start, Tend]);

subplot(2,2,4)
makePlot(t, rolling_torque, 'time(sec)', 'rolling torque', '', LineWidth, FontSize);
xlim([t_start, Tend]);

%xlim([9.6,10.5]);
% more plots on friction