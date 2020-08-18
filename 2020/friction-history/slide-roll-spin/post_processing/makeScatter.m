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

function makeScatter(varargin)
switch nargin
    
    case 5
        MS = 36; LW = 0.5; FS = 20;
    case 8
        % 6 input parameters
        % inputs are, x y coordinate, xlabel, ylabel, title, marker size
        LW = varargin{6};
        FS = varargin{7};
        MS = varargin{8};
end
        x = varargin{1}; y = varargin{2};
        x_str = varargin{3}; y_str = varargin{4}; title_str = varargin{5};
        scatter(x, y, 'SizeData', MS);
        xlabel(x_str, 'FontSize', FS);
        ylabel(y_str, 'FontSize', FS);
        set(gca, 'linewidth', LW);
        a = get(gca, 'XTick');
        set(gca, 'FontSize', FS-3)
        xlim([0,max(x)])
        title(title_str);
        
        
end