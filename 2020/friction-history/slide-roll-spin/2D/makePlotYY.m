function makePlotYY(varargin)
switch nargin
    case 8
        x1 = varargin{1}; y1 = varargin{2}; x2 = varargin{3}; y2 = varargin{4};
        x_str = varargin{5}; y_str1 = varargin{6}; y_str2 = varargin{7}; title_str = varargin{8};
        
        yyaxis left
        plot(x1, y1);
        xlabel(x_str);
        ylabel(y_str1);
        set(gca);
        a = get(gca, 'XTick');
        set(gca)
        xlim([0,max(x1)])
        
        yyaxis right
        plot(x2, y2);
        xlabel(x_str);
        ylabel(y_str2);
        set(gca);
        a = get(gca, 'XTick');
        set(gca)
        xlim([0,max(x1)])
        title(title_str);
        
        
    case 10
        x1 = varargin{1}; y1 = varargin{2}; x2 = varargin{3}; y2 = varargin{4};
        x_str = varargin{5}; y_str1 = varargin{6}; y_str2 = varargin{7}; title_str = varargin{8};
        LW = varargin{9}; FS = varargin{10}; 
        
        
        yyaxis left
        plot(x1, y1, 'LineWidth', LW);
        if contains(x_str, '$$')
            xlabel(x_str, 'FontSize', FS, 'Interpreter', 'latex');
        else
            xlabel(x_str, 'FontSize', FS);
        end
        
        
        if contains(y_str1, '$$')
            ylabel(y_str1, 'FontSize', FS, 'Interpreter', 'latex');
        else
            ylabel(y_str1, 'FontSize', FS);
        end
        
        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        xlim([0,max(x1)])
        
        yyaxis right
        plot(x2, y2, 'LineWidth', LW);
        if contains(x_str, '$$')
            xlabel(x_str, 'FontSize', FS, 'Interpreter', 'latex');
        else
            xlabel(x_str, 'FontSize', FS);
        end
        
        
        if contains(y_str2, '$$')
            ylabel(y_str2, 'FontSize', FS, 'Interpreter', 'latex');
        else
            ylabel(y_str2, 'FontSize', FS);
        end
        
        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        xlim([0,max(x1)])
        title(title_str);
        
end