clc;
clear all;

% wheel_data = importdata('/home/whu59/euler_bak/chrono_fsi_granular_1003/Figure/single_wheel_test_experiment_results/2021-04-12_17-42-09_data_modified.csv');
wheel_data = importdata('/home/whu59/euler_bak/chrono_fsi_granular_1003/Figure/SR_TestSet1_108kg_30April2020_Data/2020-04-30_10-43-47_data_modified.csv');
wheel = wheel_data.data;

% load /home/whu59/euler_bak/chrono_fsi_granular_1003/chrono-cosim-build/bin/DEMO_OUTPUT/RIG_COSIM/RIGID_SCM/RIG_ang=0.048/results.dat;
% wheel = results;

figure('color','w');

nn=size(wheel,1);
dn=1;

% plot(wheel(1:dn:nn,1),-wheel(1:dn:nn,14),'r-','LineWidth',2);hold on;
% plot(wheel(1:dn:nn,1)-wheel(1,1),wheel(1:dn:nn,16),'r-','LineWidth',2);hold on;
% plot(wheel(1:dn:nn,1)-wheel(1,1),wheel(1:dn:nn,2)-wheel(1,2),'r-','LineWidth',2);hold on;

plot(wheel(1:dn:nn,16),wheel(1:dn:nn,9),'r-','LineWidth',2);hold on;

grid on
% axis off
box on
% axis equal
% axis([0 120 0 100])
fs = 16;%FontSize
xlabel('Slip','FontSize',fs,'FontWeight','bold')
ylabel('DrawBar-PUll (N)','FontSize',fs,'FontWeight','bold')
set(gca,'fontsize',fs,'FontWeight','bold')
% legend('Fx','Fy','Fz','FontSize',fs,'FontWeight','bold')