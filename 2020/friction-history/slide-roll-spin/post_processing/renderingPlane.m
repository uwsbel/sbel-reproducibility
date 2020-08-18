% =============================================================================
  % SIMULATION-BASED ENGINEERING LAB (SBEL) - http://sbel.wisc.edu
  %
  % Copyright (c) 2019 SBEL
  % All rights reserved.
  %
  % Use of this source code is governed by a BSD-style license that can be found
  % at https://opensource.org/licenses/BSD-3-Clause
  %
  % =============================================================================
  % Contributors: Luning Fang
  % =============================================================================

% disk rolling with/without slipping rendering
% clear all
fontSize = 20;
lineWidth = 3;
close all
figure;
hold on

cmPoint = pos(1,:);
linearVelo = velo(1,:);
angularVelo = omic(1,1);

orientA = getAfromP(eulerPar(1,:));
PI = 3.14115926;
th = 0:PI/10:2*PI;
psi = 0:PI/10:PI; % coordinate z
drawSphere = zeros(length(th)*length(psi), 3);

radius = 0.2;
count = 1;
for ii = 1:length(psi)
    for jj = 1:length(th)
        drawSphere(count,1) = cmPoint(1) + radius*cos(th(jj))*sin(psi(ii));
        drawSphere(count,2) = cmPoint(2) + radius*sin(th(jj))*sin(psi(ii));
        drawSphere(count,3) = cmPoint(3) + radius*cos(psi(ii));
        count = count + 1;
    end
end

sphereHdl = plot3(drawSphere(:,1), drawSphere(:,2), drawSphere(:,3),'.');

%view(36.87,0)
view(0,90)  % top view

hold on
LFxArrow = quiver3(cmPoint(1), cmPoint(2), cmPoint(3), radius*orientA(1,1), radius*orientA(2,1), radius*orientA(3,1));
LFyArrow = quiver3(cmPoint(1), cmPoint(2), cmPoint(3), radius*orientA(1,2), radius*orientA(2,2), radius*orientA(3,2));
LFzArrow = quiver3(cmPoint(1), cmPoint(2), cmPoint(3), radius*orientA(1,3), radius*orientA(2,3), radius*orientA(3,3));

% plot sliding friction force and label its description
% contactPoint = contactPointArray(1,:);
% slidingFrScale = 0.25;
% slidingFr = slidingFrScale*Fr_array(1,:);
% slidingFrArrow = quiver3(contactPoint(1), contactPoint(2), contactPoint(3), slidingFr(1), slidingFr(2), slidingFr(3),0);
% slidingFrArrow.LineWidth = lineWidth;
% slidingFrText = text(contactPoint(1), contactPoint(3), contactPoint(3), ...
%     sprintf('slidingMode=%s, Fr=%.2gN, \n rollingMode=%s, Tr=%.2gNm', slidingMode_array{1}, norm(Fr_array(1,:)), rollingMode_array{1}, Tr_array(1)), ...
%     'FontSize', fontSize);

% plot translational velocity and label its description
veloArrow = quiver3(cmPoint(1),cmPoint(2),cmPoint(3),linearVelo(1), linearVelo(2), linearVelo(3),0);
veloArrow.LineWidth = lineWidth;
veloRatio = nonZeroDivide(norm(linearVelo), norm(angularVelo));
veloText = text(cmPoint(1)+linearVelo(1), cmPoint(2)+linearVelo(2), cmPoint(3)+linearVelo(3), ...
    sprintf('velo=%.2g m/s, \n omic=%.2g rad/s \n ratio=%.2g', norm(linearVelo), norm(angularVelo), veloRatio), ...
    'FontSize', fontSize);


% plot angular velocity initialize
angularVeloArcHdl = plot3(ones(20,1), ones(20,1), ones(20,1), 'LineWidth', lineWidth);
angularVeloArrow = scatter3(1, 1, 1, 100, '<');

timeText = text(0, min(pos(:,2)) - 2*radius, max(pos(:,3)) + 2*radius,'time=0sec', 'FontSize', fontSize);
xlabel('x'); ylabel('y'); zlabel('z');
slidingFrDescprtion = sprintf('max Fr_s = %.2gN, Fr_k = %.2gN', slide_slack_s * Ke, slide_slack_k * Ke);
title(slidingFrDescprtion, 'FontSize', fontSize);


% plot slope
% [slope_x slope_y] = meshgrid(-0.7:0.1:0.4);
% slope_z = tan(slope_angle)*slope_y - radius*cos(slope_angle) - radius*tan(slope_angle)*sin(slope_angle);
% surf(slope_x, slope_y, slope_z);


dk = 5;
kStart = dk + 1;
kEnd = length(pos);


for k = kStart:dk:kEnd
    cmPoint = pos(k,:);
    linearVelo = velo(k,:);
    angularVelo = omic(k,1);
    veloRatio = nonZeroDivide(norm(linearVelo), norm(angularVelo));
    
    contactPoint = contactPointArray(k,:);
%     slidingFr = slidingFrScale * Fr_array(k,:);
    sphereHdl.XData = sphereHdl.XData + pos(k,1) - pos(k-dk,1);
    sphereHdl.YData = sphereHdl.YData + pos(k,2) - pos(k-dk,2);
    sphereHdl.ZData = sphereHdl.ZData + pos(k,3) - pos(k-dk,3);
    orientA = getAfromP(eulerPar(k,:));
    
    % update orientation frame    
    arrowUpdate(LFxArrow, cmPoint(1), cmPoint(2), cmPoint(3), radius*orientA(1,1), radius*orientA(2,1), radius*orientA(3,1));
    arrowUpdate(LFyArrow, cmPoint(1), cmPoint(2), cmPoint(3), radius*orientA(1,2), radius*orientA(2,2), radius*orientA(3,2));
    arrowUpdate(LFzArrow, cmPoint(1), cmPoint(2), cmPoint(3), radius*orientA(1,3), radius*orientA(2,3), radius*orientA(3,3));
    
    % update sliding friction force and its description
%    arrowUpdate(slidingFrArrow, contactPoint(1), contactPoint(2), contactPoint(3), slidingFr(1), slidingFr(2), slidingFr(3));    
%    textUpdate(slidingFrText, contactPoint(1), contactPoint(2), contactPoint(3), ...
%        sprintf('slidingMode=%s, Fr=%.2gN, \n rollingMode=%s, Tr=%.2gNm', ...
%        slidingMode_array{k}, norm(Fr_array(k,:)), rollingMode_array{k}, Tr_array(k)));
    
    % update translational velocity and its description
    arrowUpdate(veloArrow, cmPoint(1), cmPoint(2), cmPoint(3), linearVelo(1), linearVelo(2), linearVelo(3));
    veloText.Position = [cmPoint(1)+linearVelo(1) cmPoint(2)+linearVelo(2) cmPoint(3)+linearVelo(3)];    
    veloText.String =  sprintf('velo=%.2g m/s, \n omic=%.2g rad/s \n ratio=%.2g', norm(linearVelo), norm(angularVelo), veloRatio);
    
    % update angular velocity
    angularVeloArc = generateArc(0.3*abs(angularVelo), 1.5*radius, pos(k,:));  % arc length, radius, center
    angularVeloArcHdl.XData = angularVeloArc.Xdata;
    angularVeloArcHdl.YData = angularVeloArc.Ydata;
    angularVeloArcHdl.ZData = angularVeloArc.Zdata;
    
    if angularVelo > 0
        angularVeloArrow.XData = angularVeloArc.Xdata(end);
        angularVeloArrow.YData = angularVeloArc.Ydata(end);
        angularVeloArrow.ZData = angularVeloArc.Zdata(end);
        angularVeloArrow.Marker = '<';

    else
        if angularVelo < 0
        angularVeloArrow.XData = angularVeloArc.Xdata(1);
        angularVeloArrow.YData = angularVeloArc.Ydata(1);
        angularVeloArrow.ZData = angularVeloArc.Zdata(1);
        angularVeloArrow.Marker = '>';
        
        end
    end
    
    % update time 
    timeText.String = sprintf('time=%.4fsec', t(k));
    
    xlim([-radius*2, radius*2]);
    ylim([min(pos(:,2)) - 2*radius, max(pos(:,2)) + 2*radius]);
    zlim([min(pos(:,3)) - 2*radius, max(pos(:,3)) + 2*radius]);
    axis equal
    pause(1e-3);
end


% update the position and magnitude of quiver function
function arrowUpdate(arrowHdl, posX, posY, posZ, arrowX, arrowY, arrowZ)
arrowHdl.XData = posX;
arrowHdl.YData = posY;
arrowHdl.ZData = posZ;
arrowHdl.UData = arrowX;
arrowHdl.VData = arrowY;
arrowHdl.WData = arrowZ;
end

% update position and text content of text box function
function textUpdate(textHdl, posX, posY, posZ, string)
textHdl.Position = [posX posY posZ];
textHdl.String = string;
end

% customized division where denominator is set to machine error when it is
% zero
function s = nonZeroDivide(num,den)
if den == 0
    den = 1e-12;
end
s = num/den;
end
