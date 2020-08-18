posInfo = csvread('/home/luning/Source/projectlets/friction-contact/ellipsoid_friction/build/snowMan.csv');
addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/core');
addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/helper');
PI = 3.141592653589793238462643383279;

timeSteps = length(posInfo);
movieLoops = timeSteps;
frames(movieLoops) = struct('cdata', [], 'colormap', []);


ball0_pos  = [0;0;4.1975];
ball0_quat = [0.9142;-0.3382;0.2209;0];
R0 = 0.2;
ball1_pos  = [0;0;2];
ball1_quat = [0.7295,-0.0782,-0.1242,0.6680486];
R1 = 2;

ball0 = sphereClass(R0, 1, ball0_pos);
ball0.updateEulerParameter(ball0_quat);

ball1 = sphereClass(R1, 1, ball1_pos);
ball1.updateEulerParameter(ball1_quat);
close all
figure;
axis equal
grid off
ball0.drawSphereSurface('checkerboard');
hold on
ball1.drawSphereSurface('watermelon');

%view(-52,3);
view(107,4);
xlim([-0.5, 0.5]);
ylim([-0.5, 0.5]);
zlim([3, 5]);

frames(1) = getframe(gcf);

FS = 20;

for i = 2:timeSteps
    if mod(i,10) == 0
        hold off
        ball0.position = posInfo(i, 3:5)';
        ball0.updateEulerParameter(posInfo(i, 6:9)');
        
        ball1.position = posInfo(i, 11:13)';
        ball1.updateEulerParameter(posInfo(i, 14:17)');
        
        ball0.drawSphereSurface('checkerboard');
        hold on
        ball1.drawSphereSurface('watermelon');
        
        view(107,4);
        xlim([-0.25, 0.25]);
        ylim([-0.25, 0.25]);
        zlim([3.5, 4.5]);
        
        xlabel('x');
        ylabel('y');
        zlabel('z');
        textHdl = text(min(xlim), min(ylim), min(zlim), ...
            sprintf('time=%.4g sec', posInfo(i,1)));
        textHdl.FontSize = FS;
        getframe(gcf);
    end
    
end