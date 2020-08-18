% mass spring model, sliding friction only, various dt comparison

clc
clear all
close all



m = 1;
k =1e3;
g = 9.8;
mu_s = 0.25;
mu_k = 0.2;

%solver = 'implicit';
fr_model_damping = true;
solver = 'explicit';
Tend = 0.3;
figure('units','normalized','outerposition',[0 0 1 1]);
hold on

scenario = 'mass-spring system on ground';
dtArray = [1e-3 1e-4 1e-5];
for dt = dtArray
    t = 0:dt:Tend;
    
    x = zeros(length(t),1);
    v = zeros(length(t),1);
    acc = zeros(length(t),1);
    Ef_array = zeros(length(t),1);
    Df_array = zeros(length(t),1);
    Sij_array = zeros(length(t),1);
    Sij_array(1) = 0;
    delta_Sij = zeros(length(t),1);
    delta_Sij(1) = 0;
    
    %x(1) = 2*mu_k*m*g/k;
    x(1) = 1e-2;
    x_init = x(1);
    v(1) = 0;
    acc(1) = 0;
    Ef_array(1) = 0;
    Df_array(1) = 0;
    
    Ke = 1e5;
    %Ke = 0;
    %Kd = 2*sqrt(m*Ke);
    %Kd = 0;
    
    slide_mode = 's';
    %define relative displacement history
    Sij = 0;
    slide_slack_s = mu_s*m*g/Ke;
    slide_slack_k = mu_k*m*g/Ke;
    friction = 0;
    pj = 0;
    delta_Sij = 0;
    Kd=0;
    for i = 1:length(t)-1
        
        
        
        if solver == 'implicit'
            if fr_model_damping == true
                Kd = 2*sqrt(m*Ke);
            else
                Kd = 0;
            end
            
            if slide_mode == 's'
                acc(i+1) = (-k*x(i) - dt*k*v(i) - Ke*Sij - Ke*dt*v(i) - Kd*v(i))/(m + k*dt^2 + Ke*dt^2 + Kd*dt);
            else
                
                acc(i+1) = (-k*x(i) - k*dt*v(i) - Ke*Sij)/(m + k*dt^2);
            end
            
        else
            
            acc(i+1) = (-k*x(i)-friction)/m;
        end
        
        v(i+1) = v(i) + dt*acc(i+1);
        x(i+1) = x(i) + dt*v(i+1);
        
        pi = x(i+1) - x(i);
        delta_Sij = pi-pj;
        Sij = Sij + delta_Sij;
        
        slide_slack = abs(Sij);
        
        if (i == 2641)
            fprintf('here\n');
        end
        
        if (i == 2741)
            fprintf('here\n');
        end
        
        Sij_old = Sij;
        if slide_mode == 's'
            
            if fr_model_damping == true
                Kd = 2*sqrt(m*Ke);
            end
            alpha_s = slide_slack/slide_slack_s;
            if alpha_s > 1
                Sij = Sij/alpha_s;
                slide_mode = 'k';
                Kd = 0;
            end
        else
            alpha_k = slide_slack/slide_slack_k;
            if alpha_k > 1
                %Sij = Sij/alpha_k;
                
                Sij = sign(Sij)*slide_slack_k;
                Kd = 0;
            else
                slide_mode = 's';
                if fr_model_damping == true
                    Kd = 2*sqrt(m*Ke);
                end
            end
        end
        
        Ef = Ke * Sij;
        
        %     if abs(acc(i+1))*abs(v(i+1)) < 1e-4
        %         Kd = 2*sqrt(m*Ke);
        %
        %     end
        
        
        Df = Kd * delta_Sij/dt;
        
        friction = Ef + Df;
        
        
        Ef_array(i+1) = Ef;
        Df_array(i+1) = Df;
        Sij_array(i+1) = Sij;
        dSij_array(i+1) = delta_Sij;
        
        fprintf('time=%.5f, slide_mode=%s, sij_old=%g, sij=%g, dsij=%g\n', t(i), slide_mode, Sij_old, Sij, delta_Sij)
        
    end
    
    FontSize = 22;
    LineWidth = 2;
    
    subplot(2,3,1)
    hold on
    makePlot(t, x, 'time (sec)', 'position (m)', scenario, LineWidth, FontSize);
    subplot(2,3,4)
    hold on
    makePlot(t, v, 'time (sec)', 'velocity (m/s)', sprintf('\\mu_s=%.2f, \\mu_k=%.2f', mu_s, mu_k), LineWidth, FontSize);
    subplot(2,3,2)
    hold on
    makePlot(t, acc, 'time (sec)', 'acceleration (m/s^2)', sprintf('%s solver', solver),  LineWidth, FontSize);
    subplot(2,3,5)
    hold on
    makePlot(t, Df_array, 'time (sec)', 'F_r damping part (N)', '', LineWidth, FontSize);
    subplot(2,3,3)
    hold on
    makePlot(t, Sij_array*1e5, 'time (sec)', 'S_{ij}(\times 10^{-5})', sprintf('Fr_s = %.3fN, Fr_k = %.3fN', mu_s*m*g, mu_k*g), LineWidth, FontSize);
    subplot(2,3,6)
    hold on
    makePlot(t, dSij_array, 'time (sec)', '\Delta S_{ij}', '', LineWidth, FontSize);
end

legend(sprintf('dt=%g',dtArray(1)), sprintf('dt=%g',dtArray(2)), sprintf('dt=%g',dtArray(3)), 'Location', 'best')
