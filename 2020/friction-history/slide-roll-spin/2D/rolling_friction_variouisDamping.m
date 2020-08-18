% disk rolling on flat surface with initial velocity v = 2m/s
% both rolling and sliding friction model enabled
% different damping coefficient is used for the rolling friction model


clc
clear all
close all

%%% need to be fixed dt array not cr array

% car wheel parameters
mass = 5;
radius = 0.2;
inertia = 0.5*mass*radius^2;


gravity = 9.8;
mu_s = 0.25;
mu_k = 0.2;

solver = 'explicit';
scenario = 'disk rolling with initial longitudinal velocity';

Tend = 15; dt = 1e-4;
figure('units','normalized','outerposition',[0 0 1 1]);
hold on

crArray = [0.6 1 1.5];
dt = 1e-4;
t = 0:dt:Tend;

EfArrays = zeros(length(t),length(crArray));
TeArrays = zeros(length(t),length(crArray));
DfArrays = zeros(length(t),length(crArray));
TdArrays = zeros(length(t),length(crArray));
accArrays = zeros(length(t),length(crArray));
alphaArrays = zeros(length(t),length(crArray));
posArrays = zeros(length(t),length(crArray));
thetaArrays = zeros(length(t),length(crArray));


for eta_r = crArray
    
    
    
    % initialize kinematics array
    pos = zeros(length(t),1); velo = zeros(length(t),1); acc = zeros(length(t),1);
    theta = zeros(length(t),1); omic = zeros(length(t),1); alpha = zeros(length(t),1);
    
    % sliding/rolling friction array
    Ef_array = zeros(length(t),1); Df_array = zeros(length(t),1);
    Te_array = zeros(length(t),1); Td_array = zeros(length(t),1);
    
    % initial condition
    velo(1) = 2; omic(1) = 0;
    
    % slide/roling friction parameter
    % rolling friction parameters
    %eta = 0.3;
    %Kr = eta*R^2*Ke;   % rolling friction stiffness
    
    
    % sliding friction model
    Ke = 1e5; % spring stiffness
    sliding_mode = 's'; % sliding friction mode
    Sij = 0; Sij_array = zeros(length(t),1); Sij_array(1) = 0;  % relative displacement
    delta_Sij = 0; dSij_array = zeros(length(t),1); dSij_array(1) = 0; % relative displacement increment
    sliding_friction = 0; % initialzie sliding friction
    sliding_slack_s = mu_s*mass*gravity/Ke; % static slack
    sliding_slack_k = mu_k*mass*gravity/Ke; % kinetic slack
    
    % rolling friction model
%    Kr = 2000;  % value used to tech report
    eta_roll = 0.4;
    Kr = eta_roll * radius ^ 2 * Ke;
    C_cr = 2*sqrt(inertia*Kr);
    
    Cr = eta_r * C_cr;
    
    
    rolling_mode = 's'; % initialze rolling mode
    rolling_slack_s = sliding_slack_s/(4*radius);  % slack for sphere on plane, static
    rolling_slack_k = sliding_slack_k/(4*radius); % slack for sphere on plane, kinetic
    rolling_torque = 0; rolling_torque_array = zeros(length(t),1); % initialize rolling torque
    rolling_history = 0; rolling_history_array = zeros(length(t),1); % relative rolling history
    excursion = 0;
    
    check_excursion = zeros(length(t),1);
    
    
    
    
    for i = 1:length(t)-1
        
        if solver == 'explicit'
            acc(i+1) = -sliding_friction/mass;
            alpha(i+1) = (sliding_friction*radius - rolling_torque)/inertia;
        end
        
        % explicit kinematics update
        velo(i+1) = velo(i) + dt*acc(i+1);
        pos(i+1) = pos(i) + dt*velo(i+1);
        omic(i+1) = omic(i) + dt*alpha(i+1);
        theta(i+1) = theta(i) + dt*omic(i+1);
        
        pj = theta(i+1)*radius - theta(i)*radius;
        pi = pos(i+1) - pos(i);
        
        % relative slide and roll
        delta_Sij = pi-pj;
        Sij = Sij + delta_Sij;
        excursion = (pj)/radius;
        rolling_history = rolling_history + excursion;
        
        slide_slack = abs(Sij);
        Sij_old = Sij;
        rolling_history_old = rolling_history;
        
        % sliding friction mode and magnitude
        if sliding_mode == 's'
            Kd = 2*sqrt(mass*Ke) * eta_r;
            alpha_s = slide_slack/sliding_slack_s;
            if alpha_s > 1
                Sij = Sij/alpha_s;
                sliding_mode = 'k';
                Kd = 0;
            end
        else
            alpha_k = slide_slack/sliding_slack_k;
            if alpha_k > 1
                Sij = sign(Sij)*sliding_slack_k;
                Kd = 0;
            else
                sliding_mode = 's';
                Kd = 2*sqrt(mass*Ke) * eta_r;
            end
        end
        
        % rolling friction mode and magnitude
        if rolling_mode == 's'
            alpha_s = rolling_history/rolling_slack_s;
            torque_damping = Cr * excursion/dt;
            if alpha_s > 1
                rolling_history = rolling_history/alpha_s;
                rolling_mode = 'k';
                torque_damping = 0;
            end
        else
            alpha_k = rolling_history/rolling_slack_k;
            if alpha_k > 1
                rolling_history = sign(rolling_history)*rolling_slack_k;
                torque_damping = 0;
            else
                rolling_mode = 's';
                torque_damping = Cr * excursion/dt;
            end
        end
        
%        fprintf('time=%.5f, rolling_history=%g, rolling_slack_k=%g, rolling_slack_s=%g, Td=%g, omic=%g\n', t(i), rolling_history, rolling_slack_k, rolling_slack_s, torque_damping, omic(i+1));
        
        Ef = Ke * Sij;
        if sliding_mode == 's'
            Df = Kd * delta_Sij/dt;
        else
            Df = 0;
        end
        
        sliding_friction = Ef + Df;
        
        rolling_torque = Kr * rolling_history + torque_damping;
        Ef_array(i+1) = Ef;
        Df_array(i+1) = Df;
        Sij_array(i+1) = Sij;
        dSij_array(i+1) = delta_Sij;
        Te_array(i+1) = Kr * rolling_history;
        Td_array(i+1) = torque_damping;
        rolling_torque_array(i+1) = rolling_torque;
        rolling_history_array(i+1) = rolling_history;
        
        %    fprintf('time=%.5f, slide_mode=%s, excursion=%g\n', t(i), sliding_mode, excursion )
        
        if excursion > 0
            check_excursion(i) = 1;
        else
            check_excursion(i) = -1;
        end
        
    end
    
    
    
    
    numLoop = (1:length(crArray))*(crArray==eta_r)';
    EfArrays(:, numLoop ) = Ef_array;
    TeArrays(:, numLoop ) = Te_array;
    DfArrays(:, numLoop ) = Df_array;
    TdArrays(:, numLoop ) = Td_array;
    accArrays(:, numLoop ) = acc;
    alphaArrays(:, numLoop ) = alpha;
    posArrays(:, numLoop ) = pos;
    thetaArrays(:, numLoop ) = theta;

    
    
end

FontSize = 30;
LineWidth = 3;


%timeRange = [8.1, 8.4];  % range for tech report
timeRange = [10.1 10.5];
subplot(2,3,1);
hold on
grid on

makeSubplot(t, EfArrays,'','$$\mathbf{E}_f$$','', LineWidth, FontSize);
xlim(timeRange)

subplot(2,3,2);
hold on
grid on

makeSubplot(t,TeArrays,'','$$\mathcal{T}_e$$','', LineWidth, FontSize);
xlim(timeRange)

subplot(2,3,4);
hold on
grid on
makeSubplot(t,DfArrays,'time (sec)','$$\mathbf{D}_f$$', '', LineWidth, FontSize);
xlim(timeRange)

subplot(2,3,5);
hold on
grid on
makeSubplot(t,TdArrays,'time (sec)','$$\mathcal{T}_d$$', '', LineWidth, FontSize);
xlim(timeRange)

subplot(2,3,3);
hold on
grid on
makeSubplot(t,accArrays,'','$$\ddot{x}$$', '', LineWidth, FontSize);
xlim(timeRange)

subplot(2,3,6);
hold on
grid on
makeSubplot(t,alphaArrays,'time (sec)','$$\ddot{\theta}$$', '', LineWidth, FontSize);
xlim(timeRange)

subplot(2,3,6)
%legend(sprintf('dt=%g',dtArray(1)), sprintf('dt=%g',dtArray(2)), sprintf('dt=%g',dtArray(3)), sprintf('dt=%g',dtArray(4)), 'Location', 'best')
%legend(sprintf('D_r=%.2g \\surd{IK_r} = %.2g',crArray(1), crArray(1)*C_cr), sprintf('D_r=%.2g',crArray(2)*C_cr), sprintf('D_r=%.2g',crArray(3)*C_cr), 'Location', 'best')
%legend(sprintf('$$D_r=%.2g \\sqrt{IK_r} = %.2g$$',crArray(1), crArray(1)*C_cr), sprintf('D_r=%.2g',crArray(2)*C_cr), sprintf('D_r=%.2g',crArray(3)*C_cr), 'Interpreter', 'latex');
lgd = legend('', '', '', 'Location', 'best');
lgd.Interpreter = 'latex';
lgd.String = {sprintf('$$D_r = %.1f D_{cr}, K_D = %.1f K_{cr}$$', crArray(1),  crArray(1)),...
              sprintf('$$D_r = %.1f D_{cr}, K_D = %.1f K_{cr}$$', crArray(2),  crArray(2)), ...
              sprintf('$$D_r = %.1f D_{cr}, K_D = %.1f K_{cr}$$', crArray(3),  crArray(3))};

lgd.FontSize = FontSize-3;







function makeSubplot(varargin)
switch nargin
    
    case 7
        time = varargin{1};
        data = varargin{2};
        x_str = varargin{3}; y_str = varargin{4}; title_str = varargin{5};
        LW = varargin{6}; FS = varargin{7};
        
        hold on        
        for ii = 1:3
            plot(time, data(:,ii), 'LineWidth', LW);            
        end
        
        if contains(x_str, '$$')
            xlabel(x_str, 'FontSize', FS, 'Interpreter', 'latex');
        else
            xlabel(x_str, 'FontSize', FS);
        end
        
        
        if contains(y_str, '$$')
            ylabel(y_str, 'FontSize', FS, 'Interpreter', 'latex');
        else
            ylabel(y_str, 'FontSize', FS);
        end

        
%         xlabel(x_str, 'FontSize', FS);
%         ylabel(y_str, 'FontSize', FS);
        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        title(title_str);
        
    case 11
        data = varargin{1}; 
        x_str = varargin{2}; y_str = varargin{3}; title_str = varargin{4};
        LW = varargin{5}; FS = varargin{6};
        subplotAx = varargin{7};
        location = varargin{8};
        zoomIndex_1 = varargin{9}; zoomIndex_2 = varargin{10};
        
        lowerLeftPos_X = subplotAx.Position(1);
        lowerLeftPos_Y = subplotAx.Position(2);
        subplotLength = subplotAx.Position(3);
        subplotWidth = subplotAx.Position(4);
        upperrightPos_X = lowerLeftPos_X + subplotLength;
        upperrightPos_Y = lowerLeftPos_Y + subplotWidth;
        
        
        hold on        
        for ii = 1:length(data)
            plot(data{ii}(:,1), data{ii}(:,2), 'LineWidth', LW);
        end
        
        xlabel(x_str, 'FontSize', FS);
        ylabel(y_str, 'FontSize', FS);
        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        title(title_str);
        
        myIndex = zoomIndex_1 < (data{1}(:,1)) & (data{1}(:,1)) < zoomIndex_2;
        box_x1 = min(data{1}(myIndex,1));
        
        box_x2 = max(data{1}(myIndex,1));
        box_y1 = min(data{1}(myIndex,2));
        box_y2 = max(data{1}(myIndex,2));
        
        drawBox(box_x1, box_y1, box_x2, box_y2);
        
        
        switch location
            case 'northeast'
                zoomPlotAx = axes('Position', [lowerLeftPos_X+0.5*subplotLength, lowerLeftPos_Y+0.5*subplotWidth, 0.45*subplotLength, 0.45*subplotWidth]);
                
            case 'southeast'
                zoomPlotAx = axes('Position', [lowerLeftPos_X+0.5*subplotLength, lowerLeftPos_Y+0.01, 0.45*subplotLength, 0.45*subplotWidth]);
                
        end
        box on

                
        for ii = 1:length(data)
            myIndex = zoomIndex_1 < (data{ii}(:,1)) & (data{ii}(:,1)) < zoomIndex_2;
            hold on
            plot(data{ii}(myIndex,1), data{ii}(myIndex,2), 'LineWidth', LW);
            xlim([min(data{ii}(myIndex,1)), max(data{ii}(myIndex,1))]);
            
            
        end

        set(gca,'xtick',[])
        set(gca,'xticklabel',[])        
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])                

end

end

% zoom in plot (plot in a plot)
% ax1 = subplot(2,3,1);
% hold on
% makeSubplot(pos_cell,'time (sec)','position of CM (m)',scenario, LineWidth, FontSize);
% 
% 
% ax2 = subplot(2,3,2);
% hold on
% makeSubplot(velo_cell,'time (sec)','velocity of CM (m/s)',sprintf('\\mu_s=%.2f,\\mu_k=%.2f', mu_s, mu_k), LineWidth, FontSize, ax2,'northeast', 0.375e-3, 1.5e-3);
% 
% ax3 = subplot(2,3,3);
% hold on
% makeSubplot(acc_cell,'time (sec)','acceleration of CM (m/s^2)', sprintf('%s solver', solver), LineWidth, FontSize, ax3, 'southeast', 0, 0.6e-3);
% 
% ax4 = subplot(2,3,4);
% hold on
% makeSubplot(theta_cell,'time (sec)','angular position (rad)', '', LineWidth, FontSize, ax4,'southeast', 7e-3, 7.7e-3);
% 
% ax5 = subplot(2,3,5);
% hold on
% makeSubplot(omic_cell,'time (sec)','angular velocity (rad/s)', '', LineWidth, FontSize, ax5,'northeast',  0.2e-3, 2e-3);
% 
% ax6 = subplot(2,3,6);
% hold on
% makeSubplot(alpha_cell,'time (sec)','angular acceleration (rad/s^2)', '', LineWidth, FontSize, ax6, 'northeast', 0, 2e-3);
% 
% subplot(2,3,1)
% legend(sprintf('dt=%g',dtArray(1)), sprintf('dt=%g',dtArray(2)), sprintf('dt=%g',dtArray(3)), sprintf('dt=%g',dtArray(4)), 'Location', 'best')
% legend(sprintf('$$eta_r=%g$$',crArray(1)), sprintf('dt=%g',crArray(2)), sprintf('dt=%g',crArray(3)), 'Location', 'best')


function drawBox(varargin)
switch nargin
    case 4
        
        x1 = varargin{1}; y1 = varargin{2};
        x2 = varargin{3}; y2 = varargin{4};
        
        length = abs(x2-x1);
        width = abs(y2-y1);
        box_ratio = 0.2;   % ratio for how much bigger the box should be than the boundary
        x1 = x1 - length*box_ratio;
        if x1 < 0
            x1 = 0;
        end
        x2 = x2 + length*box_ratio;
        y1 = y1 - width*box_ratio;
        y2 = y2 + width*box_ratio;
       
        
        x = [x1, x2, x2, x1, x1];
        y = [y1, y1, y2, y2, y1];
        hold on
        plot(x, y, 'black', 'LineWidth', 2);
        
        
    case 5
end

end