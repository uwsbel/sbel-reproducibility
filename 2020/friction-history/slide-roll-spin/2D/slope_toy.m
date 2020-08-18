clc
clear all
close all

m = 1;
g = 9.8;
k = 1e3;
mu_s = 0.25;
mu_k = 0.2;
slope_theta = 0.18;

%solver = 'implicit';
solver = 'explicit';
fr_model_damping = false;

Tend = 0.2;
dt = 1e-4;
t = 0:dt:Tend;

x = zeros(length(t),1);
v = zeros(length(t),1);
acc = zeros(length(t),1);
Ef_array = zeros(length(t),1);
Df_array = zeros(length(t),1);
dSij_array = zeros(length(t),1);

x(1) = 0;
x_init = x(1);
v(1) = 0;
acc(1) = g*sin(slope_theta);
Ef_array(1) = 0;
Df_array(1) = 0;
dSij_array(1) = 0;

Ke = 1e5;
%Ke = 0;
%Kd = 2*sqrt(m*Ke);
%Kd = 0;

slide_mode = 's';
%define relative displacement history
Sij = 0;
Sij_array = zeros(length(t),1);


theta_array =  zeros(length(t),1);
%    slope_theta = max(slope_theta - 0.001, 0.18);

theta_array(1) = slope_theta;
Sij_array(1) = 0;
% slide_slack_s = mu_s*m*g*cos(slope_theta)/Ke;
% slide_slack_k = mu_k*m*g*cos(slope_theta)/Ke;
friction = 0;
pj = 0;
delta_Sij = 0;
Kd=0;
for i = 1:length(t)-1
    
    %    slope_theta = max(slope_theta - 0.001, 0.18);
    slide_slack_s = mu_s*m*g*cos(slope_theta)/Ke;
    slide_slack_k = mu_k*m*g*cos(slope_theta)/Ke;
    
    if solver == 'implicit'
        if fr_model_damping == true
            Kd = 2*sqrt(m*Ke);
        else
            Kd = 0;
        end
        
        if slide_mode == 's'
            acc(i+1) = (m*g*sin(slope_theta) - Ke*Sij - Ke*dt*v(i) - Kd*v(i))/(m + k*dt^2 + Ke*dt^2 + Kd*dt);
        else
            
            acc(i+1) = (m*g*sin(slope_theta) - Ke*Sij)/(m + k*dt^2);
        end
        
    else
        
        acc(i+1) = (m*g*sin(slope_theta) - friction)/m;
    end
    
    
    
    v(i+1) = v(i) + dt*acc(i+1);
    x(i+1) = x(i) + dt*v(i+1);
    
    pi = x(i+1) - x(i);
    delta_Sij = pi-pj;
    Sij = Sij + delta_Sij;
    
    slide_slack = abs(Sij);
    
    Sij_old = Sij;
    if slide_mode == 's'
        
        if fr_model_damping == true
            Kd = 2*sqrt(m*Ke);
        else
            Kd = 0;
        end
        
        
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
            
            Sij = sign(Sij)*slide_slack_k;
            Kd = 0;
        else
            slide_mode = 's';
            if fr_model_damping == true
                Kd = 2*sqrt(m*Ke);
            else
                Kd = 0;
            end
        end
    end
    
    Ef = Ke * Sij;
    
    Df = Kd * delta_Sij/dt;
    
    friction = Ef + Df;
    
    
    Ef_array(i+1) = Ef;
    Df_array(i+1) = Df;
    Sij_array(i+1) = Sij;
    theta_array(i+1) = slope_theta;
    dSij_array(i+1) = delta_Sij;
    
    fprintf('time=%.5f, slide_mode=%s, sij_old=%g, sij=%g, dsij=%g\n', t(i), slide_mode, Sij_old, Sij, delta_Sij)
    
    
    
end

FontSize = 22;
LineWidth = 2;

figure('units','normalized','outerposition',[0 0 1 1]);
subplot(2,3,1)
plot(t, x, 'LineWidth', LineWidth);
xlabel('time (sec)', 'FontSize', FontSize);
ylabel('position (m)', 'FontSize', FontSize);
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-3)
xlim([0,Tend])
title(sprintf('%s solver', solver));

subplot(2,3,4)
plot(t, v, 'LineWidth', LineWidth);
xlabel('time (m/s)', 'FontSize', FontSize);
ylabel('velocity (m/s)', 'FontSize', FontSize);
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-3)
title(sprintf('\\mu_s=%.2f, \\mu_k=%.2f', mu_s, mu_k));
xlim([0,Tend])

% subplot(2,3,3)
% plot(t, theta_array, 'LineWidth', LineWidth)
% xlabel('time (sec)', 'FontSize', FontSize);
% ylabel('incline angle (rad)', 'FontSize', FontSize);
% set(gca, 'linewidth', LineWidth);
% a = get(gca, 'XTick');
% set(gca, 'FontSize', FontSize-3)
% title(sprintf('tan^{-1} \\mu_s = %.2frad, tan^{-1} \\mu_k = %.2frad', atan(mu_s), atan(mu_k)));
% xlim([0, Tend])


subplot(2,3,2)
plot(t, Ef_array, 'LineWidth', LineWidth)
xlabel('time (sec)', 'FontSize', FontSize);
ylabel('F_r elastic part (N)', 'FontSize', FontSize);
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-3)
title(sprintf('max static fr = %.3fN, slide fr = %.3fN', mu_s*m*g*cos(slope_theta), mu_k*g*cos(slope_theta)));
xlim([0, Tend])

subplot(2,3,5)
plot(t, Df_array, 'LineWidth', LineWidth)
%title(sprintf('Ke=%.0gN/m, Kd=%.2fNs/m, dt = %gsec', Ke, Kd,dt));
xlabel('time (sec)', 'FontSize', FontSize);
ylabel('F_r damping part (N)', 'FontSize', FontSize);
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-3)
xlim([0,Tend])
title(sprintf('Kd=%.2f',Kd));


subplot(2,3,3)
plot(t, Sij_array, 'LineWidth', LineWidth)
xlabel('time (sec)', 'FontSize', FontSize);
ylabel('S_{ij}', 'FontSize', FontSize);
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-3)
xlim([0, Tend])
title(sprintf('incline=%.2frad', slope_theta));


subplot(2,3,6)
plot(t, dSij_array, 'LineWidth', LineWidth)
%title(sprintf('Ke=%.0gN/m, Kd=%.2fNs/m, dt = %gsec', Ke, Kd,dt));
xlabel('time (sec)', 'FontSize', FontSize);
ylabel('\Delta S_{ij}', 'FontSize', FontSize);
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize-3)
xlim([0,Tend])


