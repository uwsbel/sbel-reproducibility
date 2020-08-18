% script disk down slope of various size
FontSize = 50;
LineWidth = 5;
MarkerSize = 10;

figure('units','normalized','outerposition',[0 0 1 1]);
hold on

eta_array = 0.4:0.2:1.2;

for eta_r = eta_array
Ratio = [];
Slope_angle = [];
for slope_angle_deg = 1:0.1:30

    fprintf('eta=%f, slope=%f\n', eta_r, slope_angle_deg)
disk_down_slope;

Slope_angle = [Slope_angle;slope_angle_deg];
Ratio = [Ratio;pos(end)/theta(end)];

end


plot(Slope_angle, Ratio, '*', 'MarkerSize', MarkerSize, 'LineWidth', LineWidth);
xlabel('incline angle (degree)', 'FontSize', FontSize);
ylabel('$$\lim_{t\to \infty}{{x}}/{{\theta}}$$', 'FontSize', FontSize, 'Interpreter', 'latex');
grid on
set(gca, 'linewidth', LineWidth);
a = get(gca, 'XTick');
set(gca, 'FontSize', FontSize)
xlim([1, 36]);
%title('disk-spring-damper system', 'FontSize', FontSize)
end

t2 = annotation('textbox');
textString1 = sprintf('$$K_E = %.1eN/m$$', Ke);
t2.String = textString1;
t2.Interpreter = 'latex';
t2.FontSize = FontSize;

t2 = annotation('textbox');
textString2 = sprintf('$$K_R = 4\\eta R^2 K_E$$');
t2.String = textString2;
t2.Interpreter = 'latex';
t2.FontSize = FontSize;

t3 = annotation('textbox');
textString3 = sprintf('$$R = 0.2m$$');
t3.String = textString3;
t3.Interpreter = 'latex';
t3.FontSize = FontSize;


lgd=legend(sprintf('4\\eta=%g',eta_array(1)), sprintf('4\\eta=%g',eta_array(2)), sprintf('4\\eta=%g',eta_array(3)), sprintf('4\\eta=%g',eta_array(4)),sprintf('4\\eta=%g',eta_array(5)), 'Location', 'best');
lgd.FontSize = FontSize;
