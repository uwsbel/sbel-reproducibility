function makePlot(varargin)
switch nargin
    
    case 7
        
        x = varargin{1}; y = varargin{2};
        x_str = varargin{3}; y_str = varargin{4}; title_str = varargin{5};
        LW = varargin{6}; FS = varargin{7};
        plot(x, y, 'LineWidth', LW);
        xlabel(x_str, 'FontSize', FS);
        ylabel(y_str, 'FontSize', FS);
        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        xlim([0,max(x)])
        title(title_str);
        
    case 11
        x = varargin{1}; y = varargin{2};
        x_str = varargin{3}; y_str = varargin{4}; title_str = varargin{5};
        LW = varargin{6}; FS = varargin{7};
        innerBox_1 = varargin{8}; innerBox_2 = varargin{9};
        zoomIndex_1 = varargin{10}; zoomIndex_2 = varargin{11};
        plot(x, y, 'LineWidth', LW);
        xlabel(x_str, 'FontSize', FS);
        ylabel(y_str, 'FontSize', FS);
        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        xlim([0,max(x)])
        title(title_str);
        axes('Position', [innerBox_1, innerBox_2, 0.2, 0.2])
        box on
        myIndex = zoomIndex_1 < x & x < zoomIndex_2;
        plot(x(myIndex), y(myIndex));
        
        
end




