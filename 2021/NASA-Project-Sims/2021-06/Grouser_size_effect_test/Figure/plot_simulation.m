if(1==0)
    clc;
    clear all;

    % wheel_data = importdata('~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/002_12mm/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_0.0/results.txt');
    % wheel = wheel_data.data;

    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/002_12mmGrouser_12mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_N0.5/results.txt;
    wheel01 = results;
    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/002_12mmGrouser_12mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_0.0/results.txt;
    wheel02 = results;
    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/002_12mmGrouser_12mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_P0.5/results.txt;
    wheel03 = results;

    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/003_10mmGrouser_10mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_N0.5/results.txt;
    wheel04 = results;
    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/003_10mmGrouser_10mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_0.0/results.txt;
    wheel05 = results;
    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/003_10mmGrouser_10mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_P0.5/results.txt;
    wheel06 = results;

    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/004_08mmGrouser_08mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_N0.5/results.txt;
    wheel07 = results;
    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/004_08mmGrouser_08mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_0.0/results.txt;
    wheel08 = results;
    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/004_08mmGrouser_08mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_P0.5/results.txt;
    wheel09 = results;

    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/005_06mmGrouser_06mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_N0.5/results.txt;
    wheel10 = results;
    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/005_06mmGrouser_06mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_0.0/results.txt;
    wheel11 = results;
    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/005_06mmGrouser_06mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_P0.5/results.txt;
    wheel12 = results9(25314:end,:);

    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/006_04mmGrouser_04mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_N0.5/results.txt;
    wheel13 = results;
    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/006_04mmGrouser_04mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_0.0/results.txt;
    wheel14 = results;
    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/006_04mmGrouser_04mmSPH/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_P0.5/results.txt;
    wheel15 = results;
end

n01=size(wheel01,1);
n02=size(wheel02,1);
n03=size(wheel03,1);
n04=size(wheel04,1);
n05=size(wheel05,1);
n06=size(wheel06,1);
n07=size(wheel07,1);
n08=size(wheel08,1);
n09=size(wheel09,1);
n10=size(wheel10,1);
n11=size(wheel11,1);
n12=size(wheel12,1);
n13=size(wheel13,1);
n14=size(wheel14,1);
n15=size(wheel15,1);

figure('color','w');

dn=200;
id = 4; % 4 sinkage, 11 dbp, 16 torque
slip = 0.5;

sign = 1.0;
offset = 0;
small = 0;
if(id ==4)
    sign = -1;
    offset = 0.38;
    small = 0.004;
end
if(id ==16)
    sign = -1;
end

if(slip==-0.5)
   plot(wheel01(1:dn:n01,1), sign * wheel01(1:dn:n01,id) + offset - small,'r-','LineWidth',2);hold on;
   plot(wheel04(1:dn:n04,1), sign * wheel04(1:dn:n04,id) + offset,'b-','LineWidth',2);hold on;
   plot(wheel07(1:dn:n07,1), sign * wheel07(1:dn:n07,id) + offset - small,'g-','LineWidth',2);hold on;
   plot(wheel10(1:dn:n10,1), sign * wheel10(1:dn:n10,id) + offset - small,'k-','LineWidth',2);hold on;
   plot(wheel13(1:dn:n13,1), sign * wheel13(1:dn:n13,id) + offset,'c-','LineWidth',2);hold on;
end

if(slip==0.0)
   plot(wheel02(1:dn:n02,1), sign * wheel02(1:dn:n02,id) + offset - small,'r-','LineWidth',2);hold on;
   plot(wheel05(1:dn:n05,1), sign * wheel05(1:dn:n05,id) + offset,'b-','LineWidth',2);hold on;
   plot(wheel08(1:dn:n08,1), sign * wheel08(1:dn:n08,id) + offset - small,'g-','LineWidth',2);hold on;
   plot(wheel11(1:dn:n11,1), sign * wheel11(1:dn:n11,id) + offset - small,'k-','LineWidth',2);hold on;
   plot(wheel14(1:dn:n14,1), sign * wheel14(1:dn:n14,id) + offset,'c-','LineWidth',2);hold on;
end

if(slip==0.5)
   plot(wheel03(1:dn:n03,1), sign * wheel03(1:dn:n03,id) + offset - small,'r-','LineWidth',2);hold on;
   plot(wheel06(1:dn:n06,1), sign * wheel06(1:dn:n06,id) + offset,'b-','LineWidth',2);hold on;
   plot(wheel09(1:dn:n09,1), sign * wheel09(1:dn:n09,id) + offset - small,'g-','LineWidth',2);hold on;
   plot(wheel12(1:dn:n12,1), sign * wheel12(1:dn:n12,id) + offset - small,'k-','LineWidth',2);hold on;
   plot(wheel15(1:dn:n15,1), sign * wheel15(1:dn:n15,id) + offset,'c-','LineWidth',2);hold on;
end

% height = 0.1;%0.096;0.096;0.1
% radi = 0.13;%0.128;0.126;0.126;
% 0;0.006;0.008;0.004
% plot(wheel(1:dn:nn,1),wheel(1:dn:nn,16),'c-','LineWidth',2);hold on;

grid on
box on
% axis off
% axis equal
fs = 16;%FontSize
xlabel('Time (s)','FontSize',fs,'FontWeight','bold')
t_sta = 0;
t_end = 40;
if(id==11)
    axis([t_sta t_end -500 200])
    ylabel('Drawbar-pull (N)','FontSize',fs,'FontWeight','bold')
end
if(id==16)
    axis([t_sta t_end -100 200])
    ylabel('Torque (Nm)','FontSize',fs,'FontWeight','bold')
end
if(id==4)
    axis([t_sta t_end 0.01 0.06])
    ylabel('Sinkage (m)','FontSize',fs,'FontWeight','bold')
end

legend('12mm','10mm','8mm','6mm','4mm','FontSize',fs,'FontWeight','bold')
set(gca,'fontsize',fs,'FontWeight','bold')



