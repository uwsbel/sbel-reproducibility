if( 1 == 0 )
    clc;
    clear all;
    % wheel_data = importdata('/home/whu59/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1004/chrono-dev-io/002_12mm/DEMO_OUTPUT/FSI_Rover_Single_Wheel/Wheel_slip_0.0/results.txt');
    % wheel = wheel_data.data;
    load ~/research/server/euler/research/sbel/d_chrono_fsi_granular/chrono_fsi_granular_1005_control/chrono-dev-io/734_flat_up_flat_real_mass/results.txt;
    wheel = results;
end

figure('color','w');

nn=size(wheel,1);
slip = zeros(nn,6);
velocity = zeros(nn,6);
power = zeros(nn,6);
for i=1:nn
    for j=1:6
        pos = wheel(i,13*(j-1)+2:13*(j-1)+4);
        vel = wheel(i,13*(j-1)+5:13*(j-1)+7);
        omega = wheel(i,13*(j-1)+8:13*(j-1)+10);
        torque = wheel(i,13*(j-1)+11:13*(j-1)+13);
        radius = 0.28;
        if(pos(1,1)>3.3 || pos(1,1)<0.0)
            radius = 0.28;
        end
        slip(i,j) = 1.0 - (vel(1,1)^2 + vel(1,2)^2+ vel(1,3)^2)^0.5/(radius*abs(omega(1,2)));
        velocity(i,j) = norm(vel);
        power(i,j) = abs(omega(1,2) * torque(1,3));
    end
    
end

dn = 101;
id = -1;
if(id > 1)
    sign = 1;
    if(id == 9)
       sign=-1; 
    end
    if(id == 13)
       sign=-1; 
    end
    plot(wheel(1:dn:nn,1),sign*wheel(1:dn:nn,13*(1-1) + id),'r-','LineWidth',2);hold on;
    plot(wheel(1:dn:nn,1),sign*wheel(1:dn:nn,13*(3-1) + id),'g--','LineWidth',2);hold on;
    plot(wheel(1:dn:nn,1),sign*wheel(1:dn:nn,13*(5-1) + id),'b-.','LineWidth',2);hold on;
end

if(id==0)
    plot(wheel(1:dn:nn,1),velocity(1:dn:nn,1),'r-','LineWidth',2);hold on;
    plot(wheel(1:dn:nn,1),velocity(1:dn:nn,3),'g--','LineWidth',2);hold on;
    plot(wheel(1:dn:nn,1),velocity(1:dn:nn,5),'b-.','LineWidth',2);hold on;
end

if(id==1)
    plot(wheel(1:dn:nn,1),slip(1:dn:nn,1),'r-','LineWidth',2);hold on;
    plot(wheel(1:dn:nn,1),slip(1:dn:nn,3),'g--','LineWidth',2);hold on;
    plot(wheel(1:dn:nn,1),slip(1:dn:nn,5),'b-.','LineWidth',2);hold on;
end

if(id==-1)
    plot(wheel(1:dn:nn,1),power(1:dn:nn,1),'r-','LineWidth',2);hold on;
    plot(wheel(1:dn:nn,1),power(1:dn:nn,3),'g--','LineWidth',2);hold on;
    plot(wheel(1:dn:nn,1),power(1:dn:nn,5),'b-.','LineWidth',2);hold on;
end



box on
grid on
% axis([0 16.5 -1 1])
% axis off
% axis equal
fs = 16;%FontSize

xlabel('Time (s)','FontSize',fs,'FontWeight','bold')
t_sta = 1;
t_end = 28;
if(id==-1)
    axis([t_sta t_end 0 600])
    ylabel('Power (W)','FontSize',fs,'FontWeight','bold')
end
if(id==0)
    axis([t_sta t_end 0 0.8])
    ylabel('Velocity (m/s)','FontSize',fs,'FontWeight','bold')
end
if(id==1)
    axis([t_sta t_end -1 1])
    ylabel('Slip','FontSize',fs,'FontWeight','bold')
end
if(id==4)
    axis([t_sta t_end 0 1.5])
    ylabel('Z-displacement (m)','FontSize',fs,'FontWeight','bold')
end
if(id==5)
    axis([t_sta t_end -3 3])
    ylabel('X-velocity (m/s)','FontSize',fs,'FontWeight','bold')
end
if(id==6)
    axis([t_sta t_end -0.2 0.2])
    ylabel('Y-velocity (m/s)','FontSize',fs,'FontWeight','bold')
end
if(id==7)
    axis([t_sta t_end -0.2 0.2])
    ylabel('Z-velocity (m/s)','FontSize',fs,'FontWeight','bold')
end
if(id==9)
    axis([t_sta t_end 1 2])
    ylabel('Angular velocity (rad/s)','FontSize',fs,'FontWeight','bold')
end
if(id==13)
    axis([t_sta t_end -100 400])
    ylabel('Torque (Nm)','FontSize',fs,'FontWeight','bold')
end
if(id==14)
    axis([t_sta t_end -1 1])
    ylabel('Slip','FontSize',fs,'FontWeight','bold')
end
% if(id==4)
%     ylabel('Sinkage (m)','FontSize',fs,'FontWeight','bold')
% end
% if(id==4)
%     ylabel('Drawbar-pull (N)','FontSize',fs,'FontWeight','bold')
% end
legend('Front','Middle','Rear','FontSize',fs,'FontWeight','bold')
set(gca,'fontsize',fs,'FontWeight','bold')




