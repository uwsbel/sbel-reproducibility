clc
clear all
close all

% set brick on an incline, compare kinematic and friction force
% with damping and without damping

FontSize = 36;
LineWidth = 4.5;


result_dir = '/home/luning/Source/projectlets/friction-contact/slide-roll-spin/results/figsForPapers/';
image_dir = '/home/luning/Papers/2020/FrictionHistory/Images/';


plotHeight = 0.7;
figure('units','normalized','outerposition',[0 0 1 plotHeight]);
hold on
figure('units','normalized','outerposition',[0 0 1 plotHeight]);
hold on
figure('units','normalized','outerposition',[0 0 1 plotHeight]);
hold on

m = 1;
g = 9.8;
k = 1e3;
mu_s = 0.25;
mu_k = 0.2;
slope_theta = 0.18;

fr_model_damping = [true; false];


Tend = 0.1;
dt = 1e-4;
t = 0:dt:Tend;


for kk = 1:2
    
    x = zeros(length(t),1);
    v = zeros(length(t),1);
    acc = zeros(length(t),1);
    Ef_array = zeros(length(t),1);
    Df_array = zeros(length(t),1);
    dSij_array = zeros(length(t),1);
    
    acc(1) = g*sin(slope_theta);
    Ef_array(1) = 0;
    Df_array(1) = 0;
    dSij_array(1) = 0;
    
    Ke = 1e5;
    
    slide_mode = 's';
    
    %define relative displacement history
    Sij = 0;
    Sij_array = zeros(length(t),1);
    
    
    
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
        
        acc(i+1) = (m*g*sin(slope_theta) - friction)/m;
        
        v(i+1) = v(i) + dt*acc(i+1);
        x(i+1) = x(i) + dt*v(i+1);
        
        pi = x(i+1) - x(i);
        delta_Sij = pi-pj;
        Sij = Sij + delta_Sij;
        
        slide_slack = abs(Sij);
        Sij_old = Sij;
        
        if slide_mode == 's'
            
            if fr_model_damping(kk) == true
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
                if fr_model_damping(kk) == true
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
        dSij_array(i+1) = delta_Sij;
        
        fprintf('time=%.5f, slide_mode=%s, sij_old=%g, sij=%g, dsij=%g\n', t(i), slide_mode, Sij_old, Sij, delta_Sij)
        
        
        
        
    end
    
    figure(1)
    subplot(1,2,1)
    grid on
    hold on
    makeplot(t, x, 'time (sec)', '$$x$$ (m)', '', LineWidth, FontSize);
    subplot(1,2,2)
    hold on
    makeplot(t, v, 'time (sec)', '$$\dot{x}$$ (m/s)', '', LineWidth, FontSize);
    
    figure(2)
    subplot(1,2,1)
    grid on
    hold on
    makeplot(t, Ef_array, 'time (sec)', '$$\mathbf{E}_f$$ (N)', '', LineWidth, FontSize)
    subplot(1,2,2)
    hold on
    makeplot(t, Df_array, 'time (sec)', '$$\mathbf{D}_f$$ (N)','', LineWidth, FontSize);
    
    
    figure(3)
    subplot(1,2,1)
    grid on
    hold on
    makeplot(t, Sij_array, 'time (sec)', '$$S_{ij}$$ (m)', '', LineWidth, FontSize);
    subplot(1,2,2)
    hold on
    makeplot(t, dSij_array, 'time (sec)', '$$\Delta S_{ij}$$ (m)', '', LineWidth, FontSize);
    
    
end

figure(1)
lgd = legend( 'K_D = 632 Ns/m', 'K_D = 0', 'location', 'best');
lgd.FontSize = FontSize-2;
grid on
str_figname = 'brick_incline_small_kinematics';
print(gcf, strcat(image_dir, str_figname, '.png'), '-dpng', '-r300');
savefig(strcat(result_dir, str_figname, '.fig'));


figure(2)
lgd = legend( 'K_D = 632 Ns/m', 'K_D = 0', 'location', 'best');
lgd.FontSize = FontSize-2;
grid on
str_figname = 'brick_incline_small_friction';
print(gcf, strcat(image_dir, str_figname, '.png'), '-dpng', '-r300');
savefig(strcat(result_dir, str_figname, '.fig'));



figure(3)
lgd = legend( 'K_D = 632 Ns/m', 'K_D = 0', 'location', 'best');
lgd.FontSize = FontSize-2;
grid on
str_figname = sprintf('brick_incline_small_relativeMotion');
print(gcf, strcat(image_dir, str_figname, '.png'), '-dpng', '-r300');
savefig(strcat(result_dir, str_figname, '.fig'));




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