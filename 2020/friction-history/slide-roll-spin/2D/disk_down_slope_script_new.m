close all
clc
clear all

% script disk down slope of various size

FontSize = 50;
LineWidth = 5;
MarkerSize = 10;

figure('units','normalized','outerposition',[0 0 1 1]);
hold on

eta_min = 0.2;
eta_max = 0.6;
slope_angle_min = 1;
slope_angle_max = 35;
eta_array = eta_min:0.01:eta_max;
slope_angle_array = slope_angle_min:0.2:slope_angle_max;

%eta_array = 0.2;

for eta_r = eta_array
    for slope_angle_deg = slope_angle_array
        
        fprintf('eta=%.3f, slope=%.2f, ', eta_r, slope_angle_deg)
        disk_down_slope;
        
        if sliding_mode == 's' && rolling_mode == 's'
            scatter(slope_angle_deg, eta_r, '*', 'r');
        end
        
        if sliding_mode == 's' && rolling_mode == 'k'
            scatter(slope_angle_deg, eta_r, '*', 'b');
        end
        
        if sliding_mode == 'k' && rolling_mode == 'k'
            scatter(slope_angle_deg, eta_r, '*', 'g');
        end
        
        if sliding_mode == 'k' && rolling_mode == 's'
            scatter(slope_angle_deg, eta_r, '*', 'm');
        end
        
    end
    

end

eta_intercept = mu_s/2/mu_k;

eta_cr = eta_array(eta_array<eta_intercept);
alpha_cr = atan(2*eta_cr*mu_k)/3.1415*180;

% for i = 1:length(eta_cr)
%     if eta_cr(i) > 0.5
%         alpha_cr(i) = atan(mu_k)/3.1415*180;
%     end
% end
plot(alpha_cr, eta_cr, 'LineWidth', 4, 'Color', 'black')

%eta_cr_2 = eta_array;
eta_cr_2 = eta_array(eta_array<eta_intercept);

alpha_cr_2 = atan(3 * mu_s - 4*eta_cr_2*mu_k)/3.1415*180;
hold on
plot(alpha_cr_2, eta_cr_2, 'LineWidth', 4, 'Color', 'black')

t1 = annotation('textbox');
textString1 = "$$\tan \alpha = 3 \mu_s - 4 \eta_r \mu_k$$";
t1.String = textString1;
t1.Interpreter = 'latex';
t1.FontSize = FontSize;
t1.EdgeColor = 'none';

t2 = annotation('textbox');
textString1 = "$$\tan \alpha = 2 \eta_r \mu_k$$";
t2.String = textString1;
t2.Interpreter = 'latex';
t2.FontSize = FontSize;
t2.EdgeColor = 'none';

t3 = annotation('textbox');
textString1 = sprintf('$$K_D = %.1f \\sqrt{mK_E}, D_r = %.1f \\sqrt{IK_r}$$', dampingScale, dampingScale);
t3.String = textString1;
t3.Interpreter = 'latex';
t3.FontSize = FontSize;
t3.EdgeColor = 'none';


xlabel('slope angle (deg)', 'FontSize', FontSize);
ylabel('\eta_r', 'FontSize', FontSize);

set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize)
ylim([eta_min, eta_max]);
xlim([slope_angle_min, slope_angle_max]);

title(sprintf('disk on incline, \\mu_s = %.2f, \\mu_k=%.2f', mu_s, mu_k), 'FontSize', FontSize)


%
% t2 = annotation('textbox');
% textString2 = sprintf('$$K_R = 4\\eta R^2 K_E$$');
% t2.String = textString2;
% t2.Interpreter = 'latex';
% t2.FontSize = FontSize;
%
% t3 = annotation('textbox');
% textString3 = sprintf('$$R = 0.2m$$');
% t3.String = textString3;
% t3.Interpreter = 'latex';
% t3.FontSize = FontSize;
%
%
% lgd=legend(sprintf('4\\eta=%g',eta_array(1)), sprintf('4\\eta=%g',eta_array(2)), sprintf('4\\eta=%g',eta_array(3)), sprintf('4\\eta=%g',eta_array(4)),sprintf('4\\eta=%g',eta_array(5)), 'Location', 'best');
% lgd.FontSize = FontSize;
