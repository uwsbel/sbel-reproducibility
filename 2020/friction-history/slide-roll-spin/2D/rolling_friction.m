clc
clear all
close all

% car wheel parameters
mass = 5;
radius = 0.2;
inertia = 0.5*mass*radius^2;


gravity = 9.8;
mu_s = 0.25;
mu_k = 0.2;

scenario = 'disk_rolling';
solver = 'explicit';
Tend = 10; dt = 1e-4; t = 0:dt:Tend;

tech_report = true;
tech_report_dir = '/Users/lulu/Documents/TechReports/Friction3D/Images/';

% initialize kinematics array
pos = zeros(length(t),1); velo = zeros(length(t),1); acc = zeros(length(t),1);
theta = zeros(length(t),1); omic = zeros(length(t),1); alpha = zeros(length(t),1);

% sliding/rolling friction array
Ef_array = zeros(length(t),1); Df_array = zeros(length(t),1);
Te_array = zeros(length(t),1); Td_array = zeros(length(t),1);

% initial condition
velo(1) = 2; omic(1) = 0;

% slide/roling friction parameter
% rolling friction parameters
%eta = 0.3;
%Kr = eta*R^2*Ke;   % rolling friction stiffness


% sliding friction model 
Ke = 1e5; % spring stiffness
sliding_mode = 's'; % sliding friction mode
Sij = 0; Sij_array = zeros(length(t),1); Sij_array(1) = 0;  % relative displacement
delta_Sij = 0; dSij_array = zeros(length(t),1); dSij_array(1) = 0; % relative displacement increment 
sliding_friction = 0; % initialzie sliding friction
sliding_slack_s = mu_s*mass*gravity/Ke; % static slack
sliding_slack_k = mu_k*mass*gravity/Ke; % kinetic slack

% rolling friction model
Kr = 2000;
C_cr = 2*sqrt(inertia*Kr);
eta_r = 0.8;
eta_t = 1;
Cr = eta_r * C_cr;
rolling_mode = 's'; % initialze rolling mode
rolling_slack_s = sliding_slack_s/(4*radius);  % slack for sphere on plane, static
rolling_slack_k = sliding_slack_k/(4*radius); % slack for sphere on plane, kinetic
rolling_torque = 0; rolling_torque_array = zeros(length(t),1); % initialize rolling torque
rolling_history = 0; rolling_history_array = zeros(length(t),1); % relative rolling history
excursion = 0;

check_excursion = zeros(length(t),1);

for i = 1:length(t)-1
    
    if solver == 'explicit'
        acc(i+1) = -sliding_friction/mass;
        alpha(i+1) = (sliding_friction*radius - rolling_torque)/inertia ;  % don't simplify the equation, use inertia instead of mass*radius, confusing...
    end
    
    if t(i) > 3
        fprintf('time=%g, Sij=%g, delta_Sij=%g, friction=%g, Ef=%g, Df=%g\n', t(i), Sij, delta_Sij, -sliding_friction, -Ef, -Df);
    end
        
    % explicit kinematics update
    velo(i+1) = velo(i) + dt*acc(i+1);
    pos(i+1) = pos(i) + dt*velo(i+1);
    omic(i+1) = omic(i) + dt*alpha(i+1);
    theta(i+1) = theta(i) + dt*omic(i+1);
    
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
    
%    fprintf('time=%.5f, slide_mode=%s, excursion=%g\n', t(i), sliding_mode, excursion )
    
    if excursion > 0
        check_excursion(i) = 1;
    else
        check_excursion(i) = -1;
    end
    
end


if tech_report == false
FontSize = 22;
LineWidth = 2;

figure('units','normalized','outerposition',[0 0 1 1]);

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
makePlotYY(t,pos,t,theta ,'time (sec)', 'position of CM (m)', 'angular position (rad)', sprintf('%s solver', solver), LineWidth, FontSize)
subplot(2,3,2)
makePlotYY(t,velo,t,omic ,'time (sec)', 'velocity of CM (m/s)', 'angular velocity (rad/s)', sprintf('\\mu_s=%.2f,\\mu_k=%.2f', mu_s, mu_k), LineWidth, FontSize)
subplot(2,3,3)
makePlotYY(t,acc,t,alpha ,'time (sec)', 'acceleration of CM (m/s^2)', 'angular acceleration (rad/s^2)', sprintf('dt=%g', dt), LineWidth, FontSize)
xlim([9.6,10.5]);
% more plots on friction
subplot(2,3,4)
makePlotYY(t, Ef_array, t, Te_array, 'time (sec)', 'F_E (N)', 'T_E (Nm)', sprintf('(Fr_s,Fr_k) = (%.0f,%.0f)N', mu_s*mass*gravity, mu_k*mass*gravity), LineWidth, FontSize)
subplot(2,3,5)
makePlotYY(t, Df_array, t, Td_array, 'time (sec)', 'F_D (N)', 'T_D (Nm)',  sprintf('D_t=%.0fN/(m/s), D_r=%.0fNm/(rad/s)', sqrt(mass*Ke)*eta_t, Cr*eta_r), LineWidth, FontSize);
subplot(2,3,6)
makePlotYY(t, Sij_array*1e3, t, rolling_history_array*1e3, 'time (sec)', 'S_{ij} (mm)','\Theta_{ij} (\times10^{-3}rad)', 'relative motion', LineWidth, FontSize);
end

if tech_report == true
    FontSize = 29;
    LineWidth = 3.5;
plotHeight = 0.7;
figure('units','normalized','outerposition',[0 0 1 plotHeight]);
str_figname = strcat(scenario, '_pos.png');
subplot(1,2,1)
makePlotYY(t,pos,t,theta ,'time (sec)', '$$x$$', '$$\theta$$', '', LineWidth, FontSize)
subplot(1,2,2)
makePlotYY(t, Ef_array, t, Te_array, 'time (sec)', '$$\mathbf{E}_f$$', '$$\mathcal{T}_E$$', '', LineWidth, FontSize)
%print(gcf, strcat(tech_report_dir, str_figname), '-dpng', '-r300');

figure('units','normalized','outerposition',[0 0 1 plotHeight]);
str_figname = strcat(scenario, '_velo.png');
subplot(1,2,1)
makePlotYY(t,velo,t,omic ,'time (sec)', '$$\dot{x}$$', '$$\dot{\theta}$$', '' , LineWidth, FontSize)


% divided plot for large range
hAxis(1)=subplot(2,4,3);
yyaxis left
plot(t(1:7), Df_array(1:7), 'LineWidth', LineWidth);
ylim([0.5,1.01*max(Df_array)])
hAxes = gca;
hAxes.YRuler.Exponent = 0; % get rid of scientific notation on y axis
ytickformat('%.0e');



yyaxis right
plot(t(2:60), Td_array(2:60), 'LineWidth', LineWidth);
ylim([2,1.01*max(Td_array)])
xlim([0,0.006])

set(gca, 'Xcolor', 'w')
set(gca, 'XTick', []); % hide x coordinates
box off % get rid of box
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-5)
set(gca, 'Ycolor', 'w')
set(gca, 'YTick', []);


hAxis(2)=subplot(2,4,7);
yyaxis left
plot(t, Df_array, 'LineWidth', LineWidth);
h_ylabel_left = ylabel('$$\mathbf{D}_f$$', 'Interpreter', 'latex', 'FontSize', FontSize);
ylim([-100,5])
xlim([0,0.006])
hAxes = gca;
hAxes.XRuler.Exponent = 0;
xtickformat('%.0e');

box off
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-5)


yyaxis right
plot(t, Td_array, 'LineWidth', LineWidth);
ylim([-100,5])
xlim([0,0.006])
set(gca, 'Ycolor', 'w')
set(gca, 'YTick', []);

box off
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-5)
set(hAxis(2), 'Position', hAxis(2).Position + [0 1 0 0]*0.05);


set(h_ylabel_left, 'Units', 'normalized')  % change position of components to normalized units 
set(h_ylabel_left, 'Position', get(h_ylabel_left, 'Position') +[0.15, 0.65 0]) % % move x y labels relatively


hAxis(3)=subplot(2,4,8);
set(hAxis(3), 'Position', hAxis(3).Position + [0 1 0 0]*0.05);
yyaxis left
plot(t, Df_array, 'LineWidth', LineWidth);
ylim([-100,5])
xlim([9.8,11])
set(gca, 'Ycolor', 'w')
set(gca, 'YTick', []);
box off
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-5)

yyaxis right
plot(t, Td_array, 'LineWidth', LineWidth);
ylim([-100,5])
xlim([9.8,11])
% set(gca, 'Ycolor', 'w')
% set(gca, 'YTick', []);
box off
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-5)

h_xlabel = xlabel('time(sec)', 'FontSize', FontSize);
h_ylabel_right = ylabel('$$\mathcal{T}_D$$', 'Interpreter', 'latex', 'FontSize', FontSize);
set(h_ylabel_right, 'Units', 'normalized')
set(h_ylabel_right, 'Position', get(h_ylabel_right, 'Position') +[-0.15, 0.65 0])
set(h_xlabel, 'Units', 'normalized')
set(h_xlabel, 'Position', get(h_xlabel, 'Position') +[-0.25, 0 0])


subplot(2,4,4);
yyaxis left
set(gca, 'Ycolor', 'w')
set(gca, 'YTick', []);

box off
yyaxis right
%plot(t, [], 'LineWidth', LineWidth);
ylim([2,1.01*max(Td_array)])
xlim([9.8,11])


set(gca, 'Xcolor', 'w')
set(gca, 'XTick', []);
box off
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-5)


annotation(figure(2), 'line', [0.5421875 0.5421875],...
    [0.506186858316222 0.57905544147844],'Color',[0 0.45 0.74],'LineWidth',3,...
    'LineStyle',':');
annotation(figure(2), 'line',[0.904687500000001 0.904687500000001],...
    [0.506186858316222 0.579055441478441],'Color',[0.85 0.33 0.1],'LineWidth',3,...
    'LineStyle',':');
annotation(figure(2), 'line', [0.69765625 0.7515625],...
    [0.160164271047228 0.160164271047228],'LineWidth',3,'LineStyle',':');
% plots connected
annotation(figure(2),'line',[0.701562500000001 0.746875000000001],...
    [0.484279445874241 0.483420709814762],'Color',[0.85 0.33 0.1],'LineWidth',3,...
    'LineStyle','--');
annotation(figure(2),'line',[0.690625 0.690625],...
    [0.580196581196581 0.505982905982906],'Color',[0.85 0.33 0.1],'LineWidth',3,...
    'LineStyle','--');
annotation(figure(2),'line',[0.557812500000001 0.557812500000001],...
    [0.580196581196581 0.505982905982906],'Color',[0 0.45 0.74],'LineWidth',3,...
    'LineStyle','--');
annotation(figure(2),'line',[0.54453125 0.54453125],...
    [0.583615384615385 0.509401709401709],'Color',[0.85 0.33 0.1],'LineWidth',3,...
    'LineStyle','--');

%print(gcf, strcat(tech_report_dir, str_figname), '-dpng', '-r300');
set(gcf,'color','w');
img = getframe(gcf);
%imwrite(img.cdata, strcat(tech_report_dir, str_figname)); % save figure exactly how it is

figure('units','normalized','outerposition',[0 0 1 plotHeight]);
str_figname = strcat(scenario, '_acc.png');

plotHeight = 0.7;
figure('units','normalized','outerposition',[0 0 1 plotHeight]);

hAxis(1)=subplot(1,4,1);
yyaxis left
plot(t, acc, 'LineWidth', LineWidth);
%ylim([-1, 1.1*max(acc)])
xlim([0,0.006])

ylabel('$$\ddot{x}$$', 'Interpreter', 'latex', 'FontSize', FontSize);

yyaxis right
plot(t, alpha, 'LineWidth', LineWidth);
%ylim([5, 1.1*max(alpha)])
xlim([0,0.006])
hAxes = gca;
yaxes = get(hAxes,'YAxis');
yaxes(2).Exponent = 3; % use scientific notation for y axis

set(gca, 'linewidth', LineWidth);
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-5)
h_ylabel = ylabel('$$\ddot{\theta}$$', 'Interpreter', 'latex', 'FontSize', FontSize);
xlabel('time(sec)');
set(h_ylabel, 'Units', 'normalized')  % change position of components to normalized units 
set(h_ylabel, 'Position', get(h_ylabel, 'Position') - [0.55, 0, 0]) % % move x y labels relatively

Xaxes = get(hAxes,'XAxis');
Xaxes.Exponent = 0; % use scientific notation for y axis

hAxis(2)=subplot(1,4,2);
yyaxis left
plot(t, acc, 'LineWidth', LineWidth);
xlim([0.0061,max(t)])
h_ylabel = ylabel('$$\ddot{x}$$', 'Interpreter', 'latex', 'FontSize', FontSize);
set(h_ylabel, 'Units', 'normalized')  % change position of components to normalized units 
set(h_ylabel, 'Position', get(h_ylabel, 'Position') + [0.15, 0, 0]) % % move x y labels relatively

yyaxis right
plot(t, alpha, 'LineWidth', LineWidth);
xlim([0.0061,max(t)])
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-5)
set(gca, 'linewidth', LineWidth);
set(gca, 'FontSize', FontSize-5)
ylabel('$$\ddot{\theta}$$', 'Interpreter', 'latex', 'FontSize', FontSize);
xlabel('time(sec)');




subplot(1,2,2);
makePlotYY(t, Sij_array*1e3, t, rolling_history_array*1e3, 'time (sec)', '$$S_{ij}(\times10^{-3})$$','$$\Theta_{ij}(\times10^{-3})$$', '', LineWidth, FontSize);
set(gcf,'color','w');
img = getframe(gcf);
%imwrite(img.cdata, strcat(tech_report_dir, str_figname)); % save figure exactly how it is

end
