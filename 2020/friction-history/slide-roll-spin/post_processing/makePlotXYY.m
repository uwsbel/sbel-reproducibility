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

function makePlotXYY(varargin)
% make plot y1 vs x and y2 vs x  (two plots one coordinate)
switch nargin
    case 9
        LW = 0.5; FS = 10;
    case 11
        LW = varargin{10}; FS = varargin{11}; 
        
end

        x1 = varargin{1}; y1 = varargin{2}; x2 = varargin{3}; y2 = varargin{4};
        x_label = varargin{5}; y_label = varargin{6}; lgd1 = varargin{7}; lgd2 = varargin{8};
        title_str = varargin{9};

        plot(x1, y1, 'LineWidth', LW);
        
        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        xlim([0,max(x1)])
        
        hold on
        plot(x2, y2, 'LineWidth', LW);
        
        if contains(x_label, '$$')
            xlabel(x_label, 'FontSize', FS, 'Interpreter', 'latex');
        else
            xlabel(x_label, 'FontSize', FS);
        end
        
        
        if contains(y_label, '$$')
            ylabel(y_label, 'FontSize', FS, 'Interpreter', 'latex');
        else
            ylabel(y_label, 'FontSize', FS);
        end
        
        
        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        xlim([0,max(x1)])
        title(title_str);
        lgd = legend(lgd1, lgd2, 'Location', 'best');
        if contains(lgd1, '$$')
            lgd.Interpreter = 'latex';
        end
        lgd.FontSize = FS - 3;
        
end