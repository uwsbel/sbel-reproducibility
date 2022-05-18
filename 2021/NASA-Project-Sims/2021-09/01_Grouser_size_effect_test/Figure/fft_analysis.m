clc;
clear all;

% wheel_data = importdata('~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/002_12mm/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_0.0/results.txt');
% wheel = wheel_data.data;

load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/021_12mmGrouser_6mmSPH_2X_AngVel/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_N0.5/results.txt;
wheel = results;

figure('color','w');
nn=200001;%size(wheel,1)-1;
dn=1;


Ts = 0.0001;
t = 0:Ts:20-Ts;   
y = fft(wheel(1:nn-1,16));   
fs = 1/Ts;
f = (0:length(y)-1)*fs/length(y);

plot(f,abs(y))

grid on
% axis off
box on
% axis equal
% axis([0.3 10 0 50])
fs = 16;%FontSize
xlabel('Frequency (Hz)','FontSize',fs,'FontWeight','bold')
% ylabel('Drawbar-pull (N)','FontSize',fs,'FontWeight','bold')
% ylabel('Torque (Nm)','FontSize',fs,'FontWeight','bold')
ylabel('Magnitude','FontSize',fs,'FontWeight','bold')
set(gca,'fontsize',fs,'FontWeight','bold')
% legend('Fx','Fy','Fz','FontSize',fs,'FontWeight','bold')