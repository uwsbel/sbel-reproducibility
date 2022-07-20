clc;
clear all;

% wheel_data = importdata('/home/whu59/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/002_12mm/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_0.0/results.txt');
% wheel = wheel_data.data;

load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1005_control/chrono-dev-io/711_flat_up_flat_real_mass/results.txt;
wheel = results;

% aa = 1;
% bb = [1/4 1/4 1/4 1/4];
% y = filter(bb,aa,wheel(:,11:10:11));

% 701/711: 15.5 8745 8045 92.0%  1632 1346 82%
% 702/712: 18.0 
% 703/713: 24.0 
% 704/714: 32.0 
% 721/731: 16.5 10876 10005 92.0%
% 722/732: 18.5 14140 13141 93.0%
% 723/733: 24.0 23393 21649 92.5%
% 724/734: 28.0 28227 25904 91.7%


figure('color','w');

nn=size(wheel,1);
dn=10;
sta_t = 2;
end_t = 5; %15.5 18 24 32
dt = 0.00025;
power = zeros(10,2);
energy = 0;
np = 0;
for (i=1:dn:nn)
    if (wheel(i,1) > sta_t && wheel(i,1) < end_t)
        np = np+1;
        power(np,1) = np * dt * dn;
        power(np,2) = 0;
        for nw=1:6
            power(np,2) = power(np,2) + abs(wheel(i, 13*(nw-1) + 13) * wheel(i, 13*(nw-1) + 9));
        end
        energy = energy + power(np,2) * dt * dn;
    end
end

% plot(wheel(1:dn:nn,1),wheel(1:dn:nn,11:10:61),'-','LineWidth',2);hold on;
plot(power(1:dn:np,1),power(1:dn:np,2),'-','LineWidth',2);hold on;

grid on
% axis off
box on
% axis equal
% axis([1 16 -1500 1500])
fs = 16;%FontSize
xlabel('Time (s)','FontSize',fs,'FontWeight','bold')
ylabel('Power (W)','FontSize',fs,'FontWeight','bold')
set(gca,'fontsize',fs,'FontWeight','bold')
% legend('Fx','Fy','Fz','FontSize',fs,'FontWeight','bold')