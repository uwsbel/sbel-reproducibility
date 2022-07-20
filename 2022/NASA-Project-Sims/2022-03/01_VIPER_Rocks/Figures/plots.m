
% clc;
% clear all;



body = bodyposrotvel1{:,:};
n = size(body,1);

for i=1:n
    x = body(i,5);
    y = body(i,6);
    z = body(i,7);
    w = body(i,8);
    t0 = +2.0 * (w * x + y * z);
    t1 = +1.0 - 2.0 * (x * x + y * y);
    roll_x(i,1) = atan2(t0, t1) * 180 / pi;

    t2 = +2.0 * (w * y - z * x);
    if t2 > +1.0 
        t2 = +1.0;
    end

    if t2 < -1.0 
        t2 = -1.0;
    end

    pitch_y(i,1) = asin(t2) * 180 / pi;

    t3 = +2.0 * (w * z + x * y);
    t4 = +1.0 - 2.0 * (y * y + z * z);
    yaw_z(i,1) = atan2(t3, t4) * 180 / pi;
    
    tim(i,1) = body(i,1);
end

% I think roll_x is around z of local frame, pitch_y is around y of local frame, yaw_z is around x of local frame;

figure('color','w');
plot(tim(:,1), yaw_z(:,1) - yaw_z(1,1),'r-','MarkerSize',3,'LineWidth',2);hold on;
plot(tim(:,1), -(pitch_y(:,1) - pitch_y(1,1)),'g-','MarkerSize',3,'LineWidth',2);hold on;
plot(tim(:,1), -(roll_x(:,1) - roll_x(1,1)),'b-','MarkerSize',3,'LineWidth',2);hold on;



grid on
% axis off
box on
axis([0 30 -20 10])
fs = 16;%FontSize
xlabel('Time','FontSize',fs,'FontWeight','bold')
ylabel('Rotation Angles','FontSize',fs,'FontWeight','bold')
set(gca,'fontsize',fs,'FontWeight','bold')


