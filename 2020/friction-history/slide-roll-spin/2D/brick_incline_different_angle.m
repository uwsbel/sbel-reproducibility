clc
clear all
close all

%% do alpha = atan(mu_s) and atan(mu_k)

% brick on an incline of three different angles
% alpha = atan(mu_s), atan(mu_k), and one larger than atan(mu_k)

tech_report = false;
tech_report_dir = '/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/results/figsForPapers/';

m = 1;
g = 9.8;
k = 0;
mu_s = 0.25;
mu_k = 0.2;
slope_theta = atan(mu_k);
%slope_theta = 0.25;

%solver = 'implicit';
solver = 'explicit';
% change friction mode here, whether or not to include damping
fr_model_damping = true;

Tend = 0.04;
dt = 1e-4;
t = 0:dt:Tend;

x = zeros(length(t),1); v = zeros(length(t),1); acc = zeros(length(t),1);
Ef_array = zeros(length(t),1);
Df_array = zeros(length(t),1);
dSij_array = zeros(length(t),1);

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

FontSize = 36;
LineWidth = 4;

plotHeight = 0.7;
figure('units','normalized','outerposition',[0 0 1 plotHeight]);

subplot(1,2,1)
makeplot(t, x, 'time', '$$x$$', '', LineWidth, FontSize);
subplot(1,2,2)
makeplot(t, v, 'time', '$$\dot{x}$$', '', LineWidth, FontSize);

figure('units','normalized','outerposition',[0 0 1 plotHeight]);
subplot(1,2,1)
makeplot(t, Ef_array, 'time', '$$\mathbf{E}_f$$', '', LineWidth, FontSize)
subplot(1,2,2)
makeplot(t, Df_array, 'time', '$$\mathbf{D}_f$$','', LineWidth, FontSize);


figure('units','normalized','outerposition',[0 0 1 plotHeight]);
subplot(1,2,1)
makeplot(t, Sij_array, 'time', '$$S_{ij}$$', '', LineWidth, FontSize);
subplot(1,2,2)
makeplot(t, dSij_array, 'time', '$$\Delta S_{ij}$$', '', LineWidth, FontSize);



slope_theta = atan(mu_s);
%slope_theta = 0.25;

%solver = 'implicit';
solver = 'explicit';
% change friction mode here, whether or not to include damping
fr_model_damping = true;

Tend = 0.04;
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

% FontSize = 22;
% LineWidth = 2;
% 
% if tech_report == true
%     FontSize = FontSize + 7;
%     LineWidth = LineWidth + 1.5;
% end


plotHeight = 0.7;
figure(1);
subplot(1,2,1)
hold on

makeplot(t, x, 'time', '$$x$$', '', LineWidth, FontSize);
subplot(1,2,2)
hold on

makeplot(t, v, 'time', '$$\dot{x}$$', '', LineWidth, FontSize);

figure(2);
hold on
subplot(1,2,1)
hold on

makeplot(t, Ef_array, 'time', '$$\mathbf{E}_f$$', '', LineWidth, FontSize)
subplot(1,2,2)
hold on

makeplot(t, Df_array, 'time', '$$\mathbf{D}_f$$','', LineWidth, FontSize);


figure(3);
hold on
subplot(1,2,1)
hold on

makeplot(t, Sij_array, 'time', '$$S_{ij}$$', '', LineWidth, FontSize);
subplot(1,2,2)
hold on

makeplot(t, dSij_array, 'time', '$$\Delta S_{ij}$$', '', LineWidth, FontSize);




fr_model_damping = true;
slope_theta = 0.25;
Tend = 0.04;
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


if tech_report == true
    
    figure(1)
    subplot(1,2,1)
    hold on
    makeplot(t, x, 'time (sec)', '$$x$$', '', LineWidth, FontSize);
    
    subplot(1,2,2)
    hold on
    makeplot(t, v, 'time', '$$\dot{x}$$', '', LineWidth, FontSize);
    str_figname = sprintf('brick_incline_%s_kinematics.png','large');
    lgd = legend( '\alpha = tan^{-1}\mu_k = 0.197', '\alpha = tan^{-1}\mu_s = 0.244', '\alpha = 0.25', 'location', 'best');
    
    lgd.FontSize = FontSize-4;
    print(gcf, strcat(tech_report_dir, str_figname), '-dpng', '-r300');
    
    figure(2)
    subplot(1,2,1)
    hold on
    makeplot(t, Ef_array, 'time', '$$\mathbf{E}_f$$', '', LineWidth, FontSize)
    
    subplot(1,2,2)
    hold on
    makeplot(t, Df_array, 'time', '$$\mathbf{D}_f$$','', LineWidth, FontSize);
    lgd = legend( '\alpha = tan^{-1}\mu_k = 0.197', '\alpha = tan^{-1}\mu_s = 0.244', '\alpha = 0.25', 'location', 'best');

    lgd.FontSize = FontSize-4;    
    str_figname = sprintf('brick_incline_%s_friction.png','large');
    print(gcf, strcat(tech_report_dir, str_figname), '-dpng', '-r300');
    
    figure(3)
    subplot(1,2,1)
    hold on
    makeplot(t, Sij_array, 'time', '$$S_{ij}$$', '', LineWidth, FontSize);
    
    subplot(1,2,2)
    hold on
    makeplot(t, dSij_array, 'time', '$$\Delta S_{ij}$$', '', LineWidth, FontSize);
    lgd = legend( '\alpha = tan^{-1}\mu_k = 0.197', '\alpha = tan^{-1}\mu_s = 0.244', '\alpha = 0.25', 'location', 'best');
    
    lgd.FontSize = FontSize-4;
    str_figname = sprintf('brick_incline_%s_relativeMotion.png','large');
    print(gcf, strcat(tech_report_dir, str_figname), '-dpng', '-r300');
else
    
    figure(1)
    subplot(1,2,1)
    hold on
    grid on
    makeplot(t, x, 'time (sec)', '$$x$$', '', LineWidth, FontSize);
    
    subplot(1,2,2)
    hold on
    grid on

    makeplot(t, v, 'time', '$$\dot{x}$$', '', LineWidth, FontSize);
    str_figname = sprintf('brick_incline_%s_kinematics.png','large');
    lgd = legend( '\alpha = tan^{-1}\mu_k = 0.197', '\alpha = tan^{-1}\mu_s = 0.244', '\alpha = 0.25', 'location', 'best');
    
    lgd.FontSize = FontSize-4;
%    print(gcf, strcat(tech_report_dir, str_figname), '-dpng', '-r300');
    
    figure(2)
    subplot(1,2,1)
    hold on
    grid on
    makeplot(t, Ef_array, 'time', '$$\mathbf{E}_f$$', '', LineWidth, FontSize)
    hold on
    
    
    subplot(1,2,2)
    hold on
    grid on
    makeplot(t, Df_array, 'time', '$$\mathbf{D}_f$$','', LineWidth, FontSize);
    lgd = legend( '\alpha = tan^{-1}\mu_k = 0.197', '\alpha = tan^{-1}\mu_s = 0.244', '\alpha = 0.25', 'location', 'best');

    lgd.FontSize = FontSize-4;    
    str_figname = sprintf('brick_incline_%s_friction.png','large');
%    print(gcf, strcat(tech_report_dir, str_figname), '-dpng', '-r300');
    subplot(1,2,1)
    staticFr  = mu_s * m * g * cos(slope_theta);
    kineticFr = mu_k * m * g * cos(slope_theta);
    hold on
    plot(t(1:186), staticFr * ones(size(t(1:186))), '-.', 'LineWidth', LineWidth);
    plot(t(1:186), kineticFr * ones(size(t(1:186))), '-.', 'LineWidth', LineWidth);
    
    figure(3)
    subplot(1,2,1)
    hold on
    grid on
    makeplot(t, Sij_array, 'time', '$$S_{ij}$$', '', LineWidth, FontSize);
    
    subplot(1,2,2)
    hold on
    grid on
    makeplot(t, dSij_array, 'time', '$$\Delta S_{ij}$$', '', LineWidth, FontSize);
    lgd = legend( '\alpha = tan^{-1}\mu_k = 0.197', '\alpha = tan^{-1}\mu_s = 0.244', '\alpha = 0.25', 'location', 'best');
    
    lgd.FontSize = FontSize-4;
    str_figname = sprintf('brick_incline_%s_relativeMotion.png','large');
%    print(gcf, strcat(tech_report_dir, str_figname), '-dpng', '-r300');

    subplot(1,2,1)
    static_Sij  = mu_s * m * g * cos(slope_theta)/Ke;
    kinetic_Sij = mu_k * m * g * cos(slope_theta)/Ke;
    hold on
    plot(t(1:186), static_Sij * ones(size(t(1:186))), '-.', 'LineWidth', LineWidth);
    plot(t(1:186), kinetic_Sij * ones(size(t(1:186))), '-.', 'LineWidth', LineWidth);
    
    text(-0.01, static_Sij*0.9, '$$\mu_s N/K_e$$', 'FontSize', FontSize, 'Interpreter', 'latex');
    text(-0.01, kinetic_Sij*0.9, '$$\mu_k N/K_e$$', 'FontSize', FontSize, 'Interpreter', 'latex');
    


    
    
end



function makeplot(varargin)
        x = varargin{1}; y = varargin{2};
        x_str = varargin{3}; y_str = varargin{4}; title_str = varargin{5};
        LW = varargin{6}; FS = varargin{7};
        plot(x, y, 'LineWidth', LW);
        if contains(x_str, '$$')
            xlabel(x_str, 'FontSize', FS, 'Interpreter', 'latex');
        else
            xlabel(x_str, 'FontSize', FS);
        end
        
        
        if contains(y_str, '$$')
            ylabel(y_str, 'FontSize', FS, 'Interpreter', 'latex');
        else
            ylabel(y_str, 'FontSize', FS);
        end
        
        if contains(title_str, '$$')
            title(title_str, 'FontSize', FS, 'Interpreter', 'latex');
        else
            title(title_str, 'FontSize', FS);
        end
        
        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        xlim([0,max(x)])
  

end




