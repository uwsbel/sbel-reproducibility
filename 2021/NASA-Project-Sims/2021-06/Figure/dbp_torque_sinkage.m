% clc;
% clear all;

load /home/whu59/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1003/chrono-dev-io/003/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_P0.7/results.txt;
wheel = results;

xdata = [-0.7 -0.5 -0.3 -0.1 0.0 0.1 0.3 0.5 0.7]';

startID = 30000;
endID = 300000;
meanF = mean(wheel(startID:endID,11));
meanT = mean(wheel(startID:endID,16));
meanS = 0.1 + 0.28 - mean(wheel(startID:endID,4));

id = 9;
Fdata(id) = meanF;
Tdata(id) = meanT;
Sdata(id) = meanS;


figure('color','w');
plot(xdata(:),Fdata(:),'b*','MarkerSize',10,'LineWidth',2);hold on;

figure('color','w');
plot(xdata(:),-Tdata(:),'b*','MarkerSize',10,'LineWidth',2);hold on;

figure('color','w');
plot(xdata(:),1000* Sdata(:),'b*','MarkerSize',10,'LineWidth',2);hold on;


nn=size(wheel,1);
dn=1;

% plot(wheel(1:dn:nn,1),-wheel(1:dn:nn,14),'r-','LineWidth',2);hold on;
% plot(wheel(1:dn:nn,1)-wheel(1,1),wheel(1:dn:nn,16),'r-','LineWidth',2);hold on;
% plot(wheel(1:dn:nn,1)-wheel(1,1),wheel(1:dn:nn,2)-wheel(1,2),'r-','LineWidth',2);hold on;

plot(wheel(1:dn:nn,1),wheel(1:dn:nn,11),'r-','LineWidth',2);hold on;

grid on
% axis off
box on
% axis equal
axis([0 50 -500 500])
fs = 16;%FontSize
xlabel('Slip','FontSize',fs,'FontWeight','bold')
ylabel('DrawBar-PUll (N)','FontSize',fs,'FontWeight','bold')
set(gca,'fontsize',fs,'FontWeight','bold')
% legend('Fx','Fy','Fz','FontSize',fs,'FontWeight','bold')