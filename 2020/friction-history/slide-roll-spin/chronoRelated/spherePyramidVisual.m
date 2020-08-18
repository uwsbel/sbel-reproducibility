% read position info and generate a movie
% 3 balls stacking in a pyramid
% data being read: bottom left ball(pos_x, pos_z, e0, e1, e2, e3)
%                  bottom right ball (......)
%                  top ball (......)

%close all
clear all
clc
if strcmp(computer, 'MACI64') == true
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/post_processing/');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/core/');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/helper/');
    
end
if strcmp(computer, 'GLNXA64') == true
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/post_processing/');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/core/');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/helper/');
    addpath('/home/luning/Source/projectlets/friction-contact/slide-roll-spin/2D/');
end

mu_s = 0.25;
mu_k = 0.2;
eta = 0.35;
massArray = [1.57, 1.58];

if strcmp(computer, 'MACI64') == true
    dataDirectory =sprintf( '/Users/luning/Data/pyramidTest/gap30e-2R/eta_%2.0fe-2/', eta*100);
%    dataDirectory = sprintf('/Users/luning/Data/pyramidTest/smallerStep/eta%2.0fe-2/', eta*100);
end
if strcmp(computer, 'GLNXA64') == true
    dataDirectory =sprintf( '/home/luning/Build/chrono-dev/BallStack/widerPos/eta_%2.0fe-2/', eta*100);
end
    
videoFilename = sprintf('rolling_eta_%2.0fe-2_95e-2_96e-2kg', eta*100);
halfDist = 0.15;

LineWidth = 4;
FontSize = 36;
gcf = figure;
gcf.Units = 'normalized';
gcf.Position = [0 0 1 1];
radius = 0.15;
timeStepsPerFrame = 200;
tEnd = 1.5;
dT = 1e-4;
frames(floor(tEnd/dT * length(massArray)/timeStepsPerFrame)) = struct('cdata', [], 'colormap', []);

count = 0;
saveFigCount = 1;

for mass = massArray
    
    fileName  = strcat(dataDirectory, sprintf('ballsStackHertzMultiStepTopmass%.6eHalfDist%.6e.csv', mass, halfDist));
    forceFile = strcat(dataDirectory, sprintf('ballsStackHertzMultiStepForceFileTopmass%.6eHalfDist%.6e.csv', mass, halfDist));
    
    info   = dlmread(fileName,  ',');
    forces = dlmread(forceFile, ',');
    time = info(:,1);
    p1_x = info(:,2);
    p1_z = info(:,3);
    ball1_e0 = info(:,4);
    ball1_e1 = info(:,5);
    ball1_e2 = info(:,6);
    ball1_e3 = info(:,7);
    
    p2_x = info(:,8);
    p2_z = info(:,9);
    ball2_e0 = info(:,10);
    ball2_e1 = info(:,11);
    ball2_e2 = info(:,12);
    ball2_e3 = info(:,13);
    
    p3_x = info(:,14);
    p3_z = info(:,15);
    ball3_e0 = info(:,16);
    ball3_e1 = info(:,17);
    ball3_e2 = info(:,18);
    ball3_e3 = info(:,19);
    
    % initialize
    Fr_10 = [];
    Fr_13 = [];
    Tr_10_i = [];
    Tr_10_j = [];
    Tr_13_i = [];
    Tr_13_j = [];
    N_10 = [];
    N_13 = [];
    
    for ii = 1:length(forces)
        
        body_i = forces(ii, 3); % body index i
        body_j = forces(ii, 4); % body index j
        if (body_i == 0 && body_j == 1)
            Fr_10   = [Fr_10;   forces(ii, 1)  forces(ii, 8)  forces(ii,5)  forces(ii,6)  forces(ii,7)];
            Tr_10_i = [Tr_10_i; forces(ii, 1)  forces(ii, 12) forces(ii,9)  forces(ii,10) forces(ii,11)];
            Tr_10_j = [Tr_10_j; forces(ii, 1)  forces(ii, 16) forces(ii,13) forces(ii,14) forces(ii,15)];
            N_10    = [N_10;    forces(ii, 1)  forces(ii, 17) forces(ii,18) forces(ii,19)];
        end
        if (body_i == 1 && body_j == 3)
            Fr_13   = [Fr_13;   forces(ii, 1)  forces(ii, 8)  forces(ii,5)  forces(ii,6)  forces(ii,7)];
            Tr_13_i = [Tr_13_i; forces(ii, 1)  forces(ii, 12) forces(ii,9)  forces(ii,10) forces(ii,11)];
            Tr_13_j = [Tr_13_j; forces(ii, 1)  forces(ii, 16) forces(ii,13) forces(ii,14) forces(ii,15)];
            N_13    = [N_13;    forces(ii, 1)  forces(ii, 17) forces(ii,18) forces(ii,19)];
            
        end
        
        
        
    end
    
    
    t_10 = 1;
    t_13 = 1;
    
    %%
    for i = 1:ceil(tEnd/dT)
        if mod(i,timeStepsPerFrame) == 0
            count = count + 1;
            hold off
            
            % increment t_10 so the time match
            while (Fr_10(t_10, 1) < time(i) && t_10 < length(Fr_10))
                t_10 = t_10 + 1;
            end
            % increment t_13 so the time match
            while (Fr_13(t_13, 1) < time(i) && t_13 < length(Fr_13))
                t_13 = t_13 + 1;
            end
            
            
            
            
            circle(p1_x(i), p1_z(i), radius);
            hold on
            circle(p2_x(i), p2_z(i), radius);
            hold on
            circle(p3_x(i), p3_z(i), radius);
            axis equal
            
            quat1 = [ball1_e0(i), ball1_e1(i), ball1_e2(i), ball1_e3(i)];
            drawReferenceFrame(quat1, [p1_x(i), p1_z(i)], radius*0.8, 'b');
            hold on
            
            quat2 = [ball2_e0(i), ball2_e1(i), ball2_e2(i), ball2_e3(i)];
            drawReferenceFrame(quat2, [p2_x(i), p2_z(i)], radius*0.8, 'r');
            hold on
            
            quat3 = [ball3_e0(i), ball3_e1(i), ball3_e2(i), ball3_e3(i)];
            drawReferenceFrame(quat3, [p3_x(i), p3_z(i)], radius*0.8, 'y');
            hold on
            
            xlim([-0.6,0.6]);
            ylim([0, 0.6]);
            
            txtHdl1 = text(min(xlim), max(ylim)*0.75, sprintf('time=%.4fsec\n m_{top}^c = %.2f kg \n\\eta_r = %.2f\n\\mu_s = %.2f, \\mu_k = %.2f', ...
                time(i)+0.0001, mass, eta, mu_s, mu_k));
            
            %             txtHdl2 = text(-0.4, -0.04, sprintf('sliding: %s, %.3e< %.3e N \n rolling: i %s j %s, (%.3e, %.3e) < %.3e Nm \n normal F: %.3e N', ...
            %                 getMode(Fr_10(t_10,2)), norm(Fr_10(t_10,3:5)), 0.25*norm(N_10(t_10, 2:4)), ...
            %                 getMode(Tr_10_i(t_10,2)), getMode(Tr_10_j(t_10,2)),...
            %                 norm(Tr_10_i(t_10,3:5)), norm(Tr_10_j(t_10,3:5)),...
            %                 2*eta*radius*0.25*norm(N_10(t_10, 2:4)), norm(N_10(t_10, 2:4))));
            txtHdl2 = text(min(xlim), 0.04, sprintf('Fr(%s) = %.2eN \nTr(%s) = %.2e Nm \nFn = %.2e N', ...
                getMode(Fr_10(t_10,2)), norm(Fr_10(t_10,3:5)), ...
                getMode(Tr_10_i(t_10,2)), norm(Tr_10_i(t_10,3:5)), norm(N_10(t_10, 2:4))));
    
            if (t_13 <= length(Fr_13))
%                 txtHdl3 = text(0.2, 0.4, sprintf('sliding: %s, %.3e< %.3e N \n rolling: i %s j %s\n Tr: (%.3e, %.3e) < %.3e Nm \n normal F: %.3e N', ...
%                 getMode(Fr_13(t_13,2)), norm(Fr_13(t_13,3:5)), 0.25*norm(N_13(t_13, 2:4)), ...
%                 getMode(Tr_13_i(t_13,2)), getMode(Tr_13_j(t_13,2)),...
%                 norm(Tr_13_i(t_13,3:5)), norm(Tr_13_j(t_13,3:5)),...
%                 eta*radius*0.25*norm(N_13(t_13, 2:4)), norm(N_13(t_13, 2:4))));

                txtHdl3 = text(0.2, 0.4, sprintf('Fr(%s) = %.2eN \nTr(%s) = %.2e Nm \nFn = %.2e N', ...
                getMode(Fr_13(t_13,2)), norm(Fr_13(t_13,3:5)), ...
                getMode(Tr_13_i(t_13,2)), norm(Tr_13_i(t_13,3:5)), norm(N_13(t_13, 2:4))));
                
                txtHdl3.FontSize = 48;
                
                
            end
            txtHdl1.FontSize = 48;
            txtHdl2.FontSize = 48;
            
            set(gca, 'linewidth', LineWidth);
            a = get(gca, 'XTick');
            set(gca, 'FontSize', FontSize)
            
            hdl = gca;
            set(hdl, 'XTick', [])
            set(hdl, 'YTick', [])
            hdl.Units = 'normalized';
            hdl.Position = [0.05 0.05 0.9 0.9];
            
            
            frames(count) = getframe(gcf);
        
            
            
        end
        %         if time(i) == 0.2 || time(i) == 0.3 || time(i) == 0.4 || time(i) == 0.6 || time(i) == 1.5
        %             mySaveFig(1, sprintf('pyramid_mass_14_mu_25_eta_3_t_%d', saveFigCount));
        %             saveFigCount = saveFigCount + 1;
        %         end
        
    end
    
end



% write the file
vidObj = VideoWriter(strcat(videoFilename, '.avi'));
vidObj.Quality = 100;
open(vidObj);
writeVideo(vidObj, frames);
close(vidObj);


% input: quaternion, center of mass, radius, color
% draw reference frame in 2D (assuming it spins in x-z plane only)
% this need to be changed later on
function drawReferenceFrame(quat, cm, r, c)
A_33 = getAfromP(quat);  % get 3 by 3 rotation matrix from quaternion

A_22 = [A_33(1,1) A_33(1,3); A_33(3,1) A_33(3,3)];  % extract 2 by 2 rotation matrix (x-z plane)
u = A_22(:,1); w = A_22(:,2);
hdl = quiver(cm(1), cm(2), r*u(1), r*u(2), c);
hdl.LineWidth = 4;
hold on
hdl = quiver(cm(1), cm(2), r*w(1), r*w(2), c);
hdl.LineWidth = 4;
hold off
end





function h = circle(x,y,r)
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit,'LineWidth', 4);
end

function s = getMode(x)
if (x == 1)
    s = 'k';
else
    s = 's';
end
end
