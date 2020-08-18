clc
clear all
close all
% disk rolling on friction ground, with spring and damper attached
% position compared with analytical solution (assumption always rolling without
% slipping linearized (sin(theta) = theta)

scenario = 'disk-spring-damper system';


tech_report = false;
tech_report_dir = '/Users/lulu/Documents/TechReports/Friction3D/Images/';

FontSize = 29;
LineWidth = 2.5;

% disk parameters
mass = 5;
radius = 0.2;
inertia = 0.5*mass*radius^2; % parallel axis theorem
%distance_b = 0*radius;
distance_b = radius;
spring_k = 1000;
%damper_c = 72.9;
damper_c = 32.9;

F_ext = 100;


gravity = 9.8;
mu_s = 0.25;
mu_k = 0.2;


% calculate analytical solution (using rolling w/o slipping assumption) 
% and compare with numerical ones
figure('units','normalized','outerposition',[0 0 1 1]);
NUM = F_ext;
m_eff = 3*mass/2;
c_eff = damper_c*(1+distance_b/radius)^2;
k_eff = spring_k*(1+distance_b/radius)^2;
DEN = [m_eff, c_eff, k_eff];
sys = tf(NUM, DEN);
[pos_analytical, t_analytical] = step(sys);
subplot(1,2,1)
plot(t_analytical,pos_analytical, 'LineWidth', LineWidth);


solver = 'explicit';
Tend = t_analytical(end); dt = 1e-4; t = 0:dt:Tend;

% initialize kinematics array
pos = zeros(length(t),1); velo = zeros(length(t),1); acc = zeros(length(t),1);
theta = zeros(length(t),1); omic = zeros(length(t),1); alpha = zeros(length(t),1);

% sliding/rolling friction array
Ef_array = zeros(length(t),1); Df_array = zeros(length(t),1);
Te_array = zeros(length(t),1); Td_array = zeros(length(t),1);

% initial condition
pos(1) = 0; omic(1) = 0;



% sliding friction model
Ke = 1e5; % spring stiffness
sliding_mode = 's'; % sliding friction mode
Sij = 0; Sij_array = zeros(length(t),1); Sij_array(1) = 0;  % relative displacement
delta_Sij = 0; dSij_array = zeros(length(t),1); dSij_array(1) = 0; % relative displacement increment
sliding_friction = 0; % initialzie sliding friction
sliding_slack_s = mu_s*mass*gravity/Ke; % static slack
sliding_slack_k = mu_k*mass*gravity/Ke; % kinetic slack
eta_t = 1;


% rolling friction model
Kr = 200;
C_cr = 2*sqrt(inertia*Kr);
eta_r = 1;
Cr = eta_r * C_cr;
rolling_mode = 's'; % initialze rolling mode
rolling_slack_s = sliding_slack_s/(2*radius);  % slack for sphere on plane, static
rolling_slack_k = sliding_slack_k/(2*radius); % slack for sphere on plane, kinetic
rolling_torque = 0; rolling_torque_array = zeros(length(t),1); % initialize rolling torque
rolling_history = 0; rolling_history_array = zeros(length(t),1); % relative rolling history
excursion = 0;

check_excursion = zeros(length(t),1);
for i = 1:length(t)-1
    
    if solver == 'explicit'
        F_spring = -spring_k * (pos(i) + distance_b * sin(theta(i)));
        F_damper = -damper_c * (velo(i) + distance_b * cos(theta(i))*omic(i));
        
        
        acc(i+1) = -sliding_friction/mass + F_spring/mass + F_damper/mass + F_ext/mass;
        alpha(i+1) = sliding_friction*radius/inertia - rolling_torque/inertia + F_spring*distance_b*cos(theta(i))/inertia + F_damper*distance_b*cos(theta(i))/inertia ;
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
            Kd = eta_t*2*sqrt(mass*Ke);
        end
    end
    
    % rolling friction mode and magnitude
    if rolling_mode == 's'
        alpha_s = abs(rolling_history)/rolling_slack_s;
        torque_damping = Cr * omic(i+1);
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
            torque_damping = Cr * omic(i+1);
        end
    end
    
    %    fprintf('time=%.5f, rolling_history=%g, rolling_slack_k=%g, rolling_slack_s=%g, Td=%g, omic=%g\n', t(i), rolling_history, rolling_slack_k, rolling_slack_s, torque_damping, omic(i+1));
    
    Ef = Ke * Sij;
    if sliding_mode == 's'
        Df = eta_t*Kd * delta_Sij/dt;
    else
        Df = 0;
    end
    
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
    
    fprintf('time=%.5f, slide_mode=%s, excursion=%g\n', t(i), sliding_mode, excursion )
    
    if excursion > 0
        check_excursion(i) = 1;
    else
        check_excursion(i) = -1;
    end
    
end


if tech_report == false

% subplot(2,3,1)
% makePlot(t,pos,'time (sec)','position of CM (m)',sprintf('%s solver', solver), LineWidth, FontSize);
% subplot(2,3,2)
% makePlot(t,velo,'time (sec)','velocity of CM (m/s)',sprintf('\\mu_s=%.2f,\\mu_k=%.2f', mu_s, mu_k), LineWidth, FontSize);
% subplot(2,3,3)
% makePlot(t,acc,'time (sec)','acceleration of CM (m/s^2)', '', LineWidth, FontSize);
%
% subplot(2,3,4)
% makePlot(t,theta,'time (sec)','angular position (rad)', '', LineWidth, FontSize);
% subplot(2,3,5)
% makePlot(t,omic,'time (sec)','angular velocity (rad/s)', '', LineWidth, FontSize);
% subplot(2,3,6)
% makePlot(t,alpha,'time (sec)','angular acceleration (rad/s^2)', '', LineWidth, FontSize);
subplot(2,3,1)
makePlotYY(t,pos,t,theta ,'time (sec)', 'position of CM (m)', 'angular position (rad)', scenario, LineWidth, FontSize)
subplot(2,3,2)
makePlotYY(t,velo,t,omic ,'time (sec)', 'velocity of CM (m/s)', 'angular velocity (rad/s)', sprintf('\\mu_s=%.2f,\\mu_k=%.2f', mu_s, mu_k), LineWidth, FontSize)
subplot(2,3,3)
makePlotYY(t,acc,t,alpha ,'time (sec)', 'acceleration of CM (m/s^2)', 'angular acceleration (rad/s^2)', sprintf('%s solver, dt=%g', solver, dt), LineWidth, FontSize)
% more plots on friction
subplot(2,3,4)
makePlotYY(t, Ef_array, t, Te_array, 'time (sec)', 'F_E (N)', 'T_E (Nm)', sprintf('(Fr_s,Fr_k)=(%.0f,%.0f), (Tr_s,Tr_k)=(%.2f,%.2f)', mu_s*mass*gravity, mu_k*mass*gravity, Kr*rolling_slack_s, Kr*rolling_slack_k), LineWidth, FontSize)
subplot(2,3,5)
makePlotYY(t, Df_array, t, Td_array, 'time (sec)', 'F_D (N)', 'T_D (Nm)',  sprintf('D_t=%.0fN/(m/s), D_r=%.0fNm/(rad/s)', sqrt(mass*Ke)*eta_t, Cr*eta_r), LineWidth, FontSize);
subplot(2,3,6)
makePlotYY(t, Sij_array*1e3, t, rolling_history_array*1e3, 'time (sec)', 'S_{ij} (mm)','\Theta_{ij} (\times10^{-3}rad)', 'relative motion', LineWidth, FontSize);
end

if tech_report == true
FontSize = FontSize + 7;
LineWidth = LineWidth + 1.5;
plotHeight = 0.7;
figure('units','normalized','outerposition',[0 0 1 plotHeight]);
str_figname = 'disk_spring_damper_pos.png';
subplot(1,2,1)
makePlotYY(t,pos,t,theta ,'time (sec)', '$$x$$', '$$\theta$$', '', LineWidth, FontSize)
subplot(1,2,2)
makePlotYY(t, Ef_array, t, Te_array, 'time (sec)', '$$\mathbf{E}_f$$', '$$\mathcal{T}_E$$', '', LineWidth, FontSize)
print(gcf, strcat(tech_report_dir, str_figname), '-dpng', '-r300');

figure('units','normalized','outerposition',[0 0 1 plotHeight]);
str_figname = 'disk_spring_damper_velo.png';
subplot(1,2,1)
makePlotYY(t,velo,t,omic ,'time (sec)', '$$\dot{x}$$', '$$\dot{\theta}$$', '' , LineWidth, FontSize)
subplot(1,2,2)
makePlotYY(t, Df_array, t, Td_array, 'time (sec)', '$$\mathbf{D}_f$$', '$$\mathcal{T}_D$$',  '', LineWidth, FontSize);
print(gcf, strcat(tech_report_dir, str_figname), '-dpng', '-r300');


figure('units','normalized','outerposition',[0 0 1 plotHeight]);
str_figname = 'disk_spring_damper_acc.png';
subplot(1,2,1)
makePlotYY(t,acc,t,alpha ,'time (sec)', '$$\ddot{x}$$', '$$\ddot{\theta}$$', '', LineWidth, FontSize)
subplot(1,2,2)
makePlotYY(t, Sij_array*1e3, t, rolling_history_array*1e3, 'time (sec)', '$$S_{ij}(\times10^{-3})$$','$$\Theta_{ij}(\times10^{-3})$$', '', LineWidth, FontSize);
print(gcf, strcat(tech_report_dir, str_figname), '-dpng', '-r300');

    
end


figure;
subplot(1,2,1)
hold on
plot(t_analytical,pos_analytical, 'LineWidth', LineWidth);
plot(t, pos, 'LineWidth', LineWidth);
lgd = legend('analytical (rolling w/o slipping assumption)', 'rolling with slipping', 'Location', 'best');
lgd.FontSize = FontSize-4;
xlabel('time(sec)', 'FontSize', FontSize);
ylabel('position of CM (m)', 'FontSize', FontSize);

set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize)
xlim([0, Tend]);
title('disk-spring-damper system', 'FontSize', FontSize)

subplot(1,2,2)
hold on
plot(t_analytical,pos_analytical/radius, 'LineWidth', LineWidth);
plot(t, theta, 'LineWidth', LineWidth);
hold on
xlabel('time(sec)', 'FontSize', FontSize);
ylabel('angular position (rad)', 'FontSize', FontSize);

set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize)
xlim([0, Tend]);
title(sprintf('$$\\zeta=%.2f, b=%.2f$$', damper_c/(2*sqrt(spring_k*mass)), distance_b), 'Interpreter', 'latex','FontSize', FontSize);

