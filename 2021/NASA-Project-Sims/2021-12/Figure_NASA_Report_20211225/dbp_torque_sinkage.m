clc;
clear all;


xdata = [-0.7 -0.5 -0.3 -0.1 0.0 0.1 0.3 0.5 0.7]';
Fdata = [-87.7 -80.8 -66.4 -46.3 -26.9 5.1 30.9 44.4 67]';
Tdata = [-12.6 -12.2 -9.7 -4.8 0.4 9.1 16.3 21.1 27.5]';
Sdata = [0.01 0.013 0.0135 0.017 0.019 0.023 0.029 0.046 0.08]';


% figure('color','w');
% plot(xdata(:),Fdata(:),'b*','MarkerSize',10,'LineWidth',2);hold on;

% figure('color','w');
% plot(xdata(:),Tdata(:),'b*','MarkerSize',10,'LineWidth',2);hold on;

figure('color','w');
plot(xdata(:),1000* Sdata(:),'b*','MarkerSize',10,'LineWidth',2);hold on;


grid on
% axis off
box on
% axis equal
% axis([-0.81 0.81 -100 80])
% axis([-0.81 0.81 -15 30])
axis([-0.81 0.81 0 90])
fs = 16;%FontSize
xlabel('Slip','FontSize',fs,'FontWeight','bold')
% ylabel('Drawbar-Pull (N)','FontSize',fs,'FontWeight','bold')
% ylabel('Torque (N-m)','FontSize',fs,'FontWeight','bold')
ylabel('Sinkage (mm)','FontSize',fs,'FontWeight','bold')
set(gca,'fontsize',fs,'FontWeight','bold')
