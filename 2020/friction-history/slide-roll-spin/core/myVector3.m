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
  
classdef myVector3 < handle
    properties
        vx
        vy
        vz
        vect3
    end
    methods
        % initialize a vector
        function obj = myVector3(varargin)
            switch nargin
                case 1
                    v = varargin{1};
                    obj.vx = v(1); obj.vy = v(2); obj.vz = v(3);
                    obj.vect3 = [obj.vx;obj.vy;obj.vz];
                case 3
                    obj.vx = varargin{1};
                    obj.vy = varargin{2};
                    obj.vz = varargin{3};
                    obj.vect3 = [obj.vx;obj.vy;obj.vz];
            end
        end
        
        % draw vector at location o with length of l and color c
        function drawVect(obj, o, l, c)
            vectArrow = quiver3(o(1), o(2), o(3), l*obj.vx, l*obj.vy, l*obj.vz, c);
            vectArrow.LineWidth = 3;
        end
        
        % draw vector at location o with length of l, color c and text str at
        % the origin of the arrow
        function drawVectWithTextOrigin(obj, o, l, c, str, FS)
            LW = 4;
            vectArrow = quiver3(o(1), o(2), o(3), l*obj.vx, l*obj.vy, l*obj.vz, c);
            vectArrow.LineWidth = LW;
            text(o(1), o(2), o(3), str, 'FontSize', FS);
        end

        % draw vector at location o with length of l, color c and text str at
        % the end of the arrow
        function drawVectWithTextEnd(obj, o, l, c, str, FS)
            LW = 4;
            vectArrow = quiver3(o(1), o(2), o(3), l*obj.vx, l*obj.vy, l*obj.vz, c);
            vectArrow.LineWidth = LW;
            text(o(1)+l*obj.vx, o(2)+l*obj.vy, o(3)+l*obj.vz, str, 'FontSize', FS);
        end
        
    end
end