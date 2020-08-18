function drawEllipse(a,b,c, color)
if b == 0
    x_pos = -a:a/20:a;
    z = sqrt(1 - (x_pos./a).^2)*c;
    
    plot3(x_pos, zeros(size(x_pos)), z, color, 'LineWidth', 4);
    
    hold on
    plot3(x_pos, zeros(size(x_pos)), -z, color, 'LineWidth', 4);
    grid on
end

if a == 0
    x_pos = -b:b/20:b;
    z = sqrt(1 - (x_pos./b).^2)*c;
    
    plot3(zeros(size(x_pos)), x_pos,  z, color, 'LineWidth', 4);
    
    hold on
    plot3(zeros(size(x_pos)), x_pos, -z, color, 'LineWidth', 4);
end