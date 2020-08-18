%clc
%clear all
%close all
% disk rolling down a slope of angle alpha, with rolling and sliding friction model
% enabled

scenario = 'disk on slope';

FontSize = 36;
LineWidth = 4.5;
plotFigures = false;
verbose = false;


% slide/roling friction parameter



% disk parameters
mass = 5;
radius = 0.2;
inertia = 0.5*mass*radius^2; % parallel axis theorem

gravity = 9.8;
mu_s = 0.23;
mu_k = 0.18;

% slope_angle_deg = 30;
% eta_r = 0.6;

Ke = 1e5; % spring stiffness
slope_angle = slope_angle_deg/180*3.14 ;


solver = 'explicit';
Tend = 0.5; dt = 1e-4; t = 0:dt:Tend;

% initialize kinematics array
pos = zeros(length(t),1); velo = zeros(length(t),1); acc = zeros(length(t),1);
theta = zeros(length(t),1); omic = zeros(length(t),1); alpha = zeros(length(t),1);

% sliding/rolling friction array
Ef_array = zeros(length(t),1); Df_array = zeros(length(t),1);
Te_array = zeros(length(t),1); Td_array = zeros(length(t),1);

% initial condition
pos(1) = 0; omic(1) = 0;
velo(1) = 0;



% sliding friction model
sliding_mode = 's'; % sliding friction mode
Sij = 0; Sij_array = zeros(length(t),1); Sij_array(1) = 0;  % relative displacement
delta_Sij = 0; dSij_array = zeros(length(t),1); dSij_array(1) = 0; % relative displacement increment
sliding_friction = 0; % initialzie sliding friction
sliding_slack_s = mu_s*mass*gravity*cos(slope_angle)/Ke; % static slack
sliding_slack_k = mu_k*mass*gravity*cos(slope_angle)/Ke; % kinetic slack


dampingScale = 1;  % coefficient for crtical damping 
% rolling friction model
Kr = 4*eta_r*radius^2*Ke;
%Kr = 2000;
Cr = dampingScale*sqrt(inertia*Kr);


rolling_mode = 's'; % initialze rolling mode
rolling_slack_s = sliding_slack_s/(2*radius);  % slack for sphere on plane, static
rolling_slack_k = sliding_slack_k/(2*radius); % slack for sphere on plane, kinetic
rolling_torque = 0; rolling_torque_array = zeros(length(t),1); % initialize rolling torque
rolling_history = 0; rolling_history_array = zeros(length(t),1); % relative rolling history
excursion = 0;

check_excursion = zeros(length(t),1);
for i = 1:length(t)-1
    
    if solver == 'explicit'
        
        acc(i+1) = (-sliding_friction + mass*gravity*sin(slope_angle))/mass;
%        rolling_torque = 0;
        alpha(i+1) = (sliding_friction*radius - rolling_torque)/inertia;
    end
    
    % explicit kinematics update
    velo(i+1) = velo(i) + dt*acc(i+1);
    pos(i+1) = pos(i) + dt*velo(i+1);
    omic(i+1) = omic(i) + dt*alpha(i+1);
    theta(i+1) = theta(i) + dt*omic(i+1);
    
%     if abs(theta(i+1)) > 0.5
%         theta(i+1) = sign(theta(i+1))*0.5;
%     end
    
    pj = theta(i+1)*radius - theta(i)*radius;
    pi = pos(i+1) - pos(i);
    
    % relative slide and roll
    delta_Sij = pi-pj;
    Sij = Sij + delta_Sij;
    excursion = (pj)/radius;
    rolling_history = rolling_history + excursion;
    
    slide_slack = abs(Sij);
    Sij_old = Sij;
    rolling_history_old = rolling_history;
    
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
            Kd = 2*sqrt(mass*Ke);
        end
    end
    
    % rolling friction mode and magnitude
    if rolling_mode == 's'
        alpha_s = abs(rolling_history)/rolling_slack_s;
        torque_damping = Cr * excursion/dt;
       
        if alpha_s > 1
            rolling_history = rolling_history/alpha_s;
            rolling_mode = 'k';
            torque_damping = 0;
        end
    else
        alpha_k = abs(rolling_history)/rolling_slack_k;
        if alpha_k > 1
            rolling_history = sign(rolling_history)*rolling_slack_k;
            torque_damping = 0;
        else
            rolling_mode = 's';
            torque_damping = Cr * excursion/dt;
        end
    end
    
    %    fprintf('time=%.5f, rolling_history=%g, rolling_slack_k=%g, rolling_slack_s=%g, Td=%g, omic=%g\n', t(i), rolling_history, rolling_slack_k, rolling_slack_s, torque_damping, omic(i+1));
    
%     Ef = Ke * Sij;
%     if sliding_mode == 's'
%         Df = eta_t*Kd * delta_Sij/dt;
%     else
%         Df = 0;
%     end
    
    Ef = Ke * Sij;
    Df = dampingScale * Kd * delta_Sij/dt;

    sliding_friction = Ef + Df;
    
    rolling_torque = Kr * rolling_history + torque_damping;
    Ef_array(i+1) = Ef;
    Df_array(i+1) = Df;
    Sij_array(i+1) = Sij;
    dSij_array(i+1) = delta_Sij;
    Te_array(i+1) = Kr * rolling_history;
    Td_array(i+1) = torque_damping;
    rolling_torque_array(i+1) = rolling_torque;
    rolling_history_array(i+1) = rolling_history;
    
     if t(i) > 0 && verbose == true
        fprintf('t=%g, Sij=%g, d_Sij=%g, Fr(%s):%g=%g+%g, Tr(%s):%g=%g+%g, theta_ij=%g, velo=%g, omic=%g\n',...
                 t(i), Sij,    delta_Sij, sliding_mode, -sliding_friction, -Ef, -Df, rolling_mode, rolling_torque, Kr*rolling_history, torque_damping, rolling_history, velo(i), omic(i));
     end
    
    if excursion > 0
        check_excursion(i) = 1;
    else
        check_excursion(i) = -1;
    end
    
end

fprintf('velo=%.2e, omic=%.2e, Fr(%s)=%.2e, Tr(%s)=%.2e, mu_sN=%.2e, mu_kN=%.2e, Tr_max=%.2e, mgsin(a)=%.2e\n',...
                       velo(end), omic(end), sliding_mode, sliding_friction, ...
                       rolling_mode, rolling_torque, ...
                       mu_s*mass*gravity*cos(slope_angle), mu_k*mass*gravity*cos(slope_angle), ...
                       2*eta_r*radius*mu_k*mass*gravity*cos(slope_angle), mass*gravity*sin(slope_angle));
    


if plotFigures == true

figure('units','normalized','outerposition',[0 0 1 1]);

subplot(2,3,1)
grid on
makePlotYY(t,pos,t,theta ,'time (sec)', '$$x$$ (m)', '$$\theta$$ (rad)', '', LineWidth, FontSize)
ax=gca;
ax.YAxis(1).Exponent=-2;
set(gca, 'XLabel', []);
set(gca, 'XTickLabel', []);
subplot(2,3,2)
grid on
makePlotYY(t,velo,t,omic ,'time (sec)', '$$\dot{x}$$ (m/s)', '$$\dot{\theta}$$ (rad/s)', '', LineWidth, FontSize)
ax=gca;
ax.YAxis(1).Exponent=-1;
ax.YAxis(2).Exponent=0;

set(gca, 'XLabel', []);
set(gca, 'XTickLabel', []);

subplot(2,3,3)
grid on
makePlotYY(t,acc,t,alpha ,'time (sec)', '$$\ddot{x}$$ (m/s$$^2$$)', '$$\ddot{\theta}$$ (rad/s$$^2$$)', '', LineWidth, FontSize)
set(gca, 'XLabel', []);
set(gca, 'XTickLabel', []);

% more plots on friction
subplot(2,3,4)
grid on
%makePlotYY(t, Ef_array, t, Te_array, 'time (sec)', 'F_E (N)', 'T_E (Nm)', sprintf('(Fr_s,Fr_k)=(%.0f,%.0f), (Tr_s,Tr_k)=(%.2f,%.2f)', mu_s*mass*gravity*cos(slope_angle), mu_k*mass*gravity*cos(slope_angle), Kr*rolling_slack_s, Kr*rolling_slack_k), LineWidth, FontSize)
makePlotYY(t, Ef_array, t, Te_array, 'time (sec)', '$$\mathbf{F}_e$$ (N)', '$$\mathcal{T}_e$$ (N)', '', LineWidth, FontSize)

subplot(2,3,5)
grid on
makePlotYY(t, Df_array, t, Td_array, 'time (sec)', '$$\mathbf{F}_d$$ (N)', '$$\mathcal{T}_d$$ (N)', '', LineWidth, FontSize);
subplot(2,3,6)
grid on
makePlotYY(t, Sij_array, t, rolling_history_array, 'time (sec)', 'S_{ij} (m)','\Theta_{ij} (m)', '', LineWidth, FontSize);
ax=gca;
ax.YAxis(1).Exponent=-5;
ax.YAxis(2).Exponent=-4;

end
