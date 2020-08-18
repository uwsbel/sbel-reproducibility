if strcmp(computer, 'MACI64')
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/core');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/helper');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/2D');
    addpath('/Users/luning/Sources/projectlets/friction-contact/slide-roll-spin/post_processing/');

end
clear all
close all

% a = 0.02;
% b = 0.02;
% c = 0.05;

a = 0.05;
b = 0.02;
c = 0.02;

mass = 1;
verticalEllipsoid = ellipsoidClass(a,b,c,1);
ellipsoidHdl = verticalEllipsoid.drawEllipsoid;
axis equal

ellipsoidHdl.FaceAlpha = 0.3;
hold on
FS = 35;
xlabel('x', 'FontSize', FS)
ylabel('y', 'FontSize', FS)
zlabel('z', 'FontSize', FS)
%tickHdl = get(gca, 'XTick');
set(gca, 'FontSize', FS-3)

hdl = gca;
hdl.XTick='';
hdl.YTick='';
hdl.ZTick='';

grid on

drawEllipse(a, 0, c, 'r');
hold on
drawEllipse(0, b, c, 'g');

