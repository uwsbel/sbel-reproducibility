subplot(2,3,1);
hold on
makeSubplot(pos_cell,'time (sec)','position of CM (m)',scenario, LineWidth, FontSize,[10,11.5]);

subplot(2,3,2);
hold on
makeSubplot(velo_cell,'time (sec)','velocity of CM (m/s)',sprintf('\\mu_s=%.2f,\\mu_k=%.2f', mu_s, mu_k), LineWidth, FontSize, [10,11.5]);

subplot(2,3,3);
hold on
makeSubplot(acc_cell,'time (sec)','acceleration of CM (m/s^2)', sprintf('%s solver', solver), LineWidth, FontSize, [10,11.5]);

subplot(2,3,4);
hold on
makeSubplot(theta_cell,'time (sec)','angular position (rad)', '', LineWidth, FontSize, [10,11.5]);

subplot(2,3,5);
hold on
makeSubplot(omic_cell,'time (sec)','angular velocity (rad/s)', '', LineWidth, FontSize, [10,11.5]);

subplot(2,3,6);
hold on
makeSubplot(alpha_cell,'time (sec)','angular acceleration (rad/s^2)', '', LineWidth, FontSize, [10,11.5]);

subplot(2,3,1)
legend(sprintf('dt=%g',dtArray(1)), sprintf('dt=%g',dtArray(2)), sprintf('dt=%g',dtArray(3)), sprintf('dt=%g',dtArray(4)), 'Location', 'best')


function makeSubplot(varargin)
switch nargin
    
    case 6  %plot cell data with labels and titles
        
        data = varargin{1};
        x_str = varargin{2}; y_str = varargin{3}; title_str = varargin{4};
        LW = varargin{5}; FS = varargin{6};
        
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
        
    case 7 %plot data with a certain time range
        data = varargin{1};
        x_str = varargin{2}; y_str = varargin{3}; title_str = varargin{4};
        LW = varargin{5}; FS = varargin{6};
        timeRange = varargin{7}; 
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
        
        xlim([timeRange(1), timeRange(2)]);
        
    case 10  % zoom in plot
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
