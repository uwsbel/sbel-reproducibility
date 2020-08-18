% =============================================================================
  % SIMULATION-BASED ENGINEERING LAB (SBEL) - http://sbel.wisc.edu
  %
  % Copyright (c) 2019 SBEL
  % All rights reserved.
  %
  % Use of this source code is governed by a BSD-style license that can be found
  % at https://opensource.org/licenses/BSD-3-Clause
  %
  % =============================================================================
  % Contributors: Luning Fang
  % =============================================================================

function varargout =  makePlot(varargin)
switch nargin
    
    case 7
        % 7 input parameters, no need for plot in a plot
        % inputs are, x y coordinate, xlabel, ylabel, title, linewidth and
        % fonsize
        x = varargin{1}; y = varargin{2};
        x_str = varargin{3}; y_str = varargin{4}; title_str = varargin{5};
        LW = varargin{6}; FS = varargin{7};
        handle = plot(x, y, 'LineWidth', LW);
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

        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        xlim([0,max(x)])
        if contains(title_str, '$$')
            title(title_str, 'FontSize', FS, 'Interpreter', 'latex');
        else
            title(title_str, 'FontSize', FS);
        end
        
    case 11
        x = varargin{1}; y = varargin{2};
        x_str = varargin{3}; y_str = varargin{4}; title_str = varargin{5};
        LW = varargin{6}; FS = varargin{7};
        % x y position of the zoomed in box (normalized)
        innerBox_1 = varargin{8}; innerBox_2 = varargin{9};
        % zoom in range
        zoomIndex_1 = varargin{10}; zoomIndex_2 = varargin{11};
        handle = plot(x, y, 'LineWidth', LW);
        xlabel(x_str, 'FontSize', FS);
        ylabel(y_str, 'FontSize', FS);
        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        xlim([0,max(x)])
        title(title_str);
        a2 = axes();
        a2.Position = [innerBox_1, innerBox_2, 0.2, 0.2];
        box on
        myIndex = zoomIndex_1 < x & x < zoomIndex_2;
        plot(x(myIndex), y(myIndex));
        
        
end

switch nargout
    case 0
    case 1
        varargout{1} = handle;
end