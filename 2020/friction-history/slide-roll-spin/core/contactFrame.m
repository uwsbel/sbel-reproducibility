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

classdef contactFrame < handle
    properties
        u
        w
        n
        CF
    end
    methods
        % initialize an identical reference frame
        function obj = contactFrame(varargin)
            switch nargin
                case 0
                    obj.u = [1;0;0];
                    obj.w = [0;1;0];
                    obj.n = [0;0;1];
                    obj.CF = [obj.u obj.w obj.n];
                case 1
                    obj.CF = varargin{1};
                    obj.u = obj.CF(:,1);
                    obj.w = obj.CF(:,2);
                    obj.n = obj.CF(:,3);
                case 3
                    obj.u = varargin{1};
                    obj.w = varargin{2};
                    obj.n = varargin{3};
                    obj.CF = [obj.u obj.w obj.n];
            end
        end

        % given the local reference frame A, express local contact frame globally
        function CF_global = expressContactFrameInGlobalRF(obj, A)
            CF_g = A*obj.CF;
            CF_global = contactFrame(CF_g);
        end

        % given the local reference frame A, express global contact frame locally
        function CF_local = expressContactFrameInLocalRF(obj, A)
            CF_l = A'*obj.CF;
            CF_local = contactFrame(CF_l);
        end

        % plot contact frame at a given point o with given color c and
        % scale to a length of d
        function drawContactFrame(obj, o, c, d)
            hdl = quiver3(o(1), o(2), o(3), d*obj.u(1), d*obj.u(2), d*obj.u(3), c);
            hold on

            hdl.LineWidth = 3;
            hdl = quiver3(o(1), o(2), o(3), d*obj.w(1), d*obj.w(2), d*obj.w(3), c);
            hdl.LineWidth = 3;
            hdl = quiver3(o(1), o(2), o(3), d*obj.n(1), d*obj.n(2), d*obj.n(3), c);
            hdl.LineWidth = 3;
            hold off

        end

        % plot contact frame at a given point o with given color c and
        % scale to a length of d with label t1, t2 and t3
        function drawContactFrameWithLabels(obj, o, c, d, t1, t2, t3)
            hdl = quiver3(o(1), o(2), o(3), d*obj.u(1), d*obj.u(2), d*obj.u(3));
            text(o(1)+d*obj.u(1), o(2)+d*obj.u(2), o(3)+d*obj.u(3), t1, 'FontSize', 20);
            hold on

            hdl.LineWidth = 3;
            hdl.Color = c;
            hdl = quiver3(o(1), o(2), o(3), d*obj.w(1), d*obj.w(2), d*obj.w(3));
            text(o(1)+d*obj.w(1), o(2)+d*obj.w(2), o(3)+d*obj.w(3), t2, 'FontSize', 20);

            hdl.LineWidth = 3;
            hdl.Color = c;

            hdl = quiver3(o(1), o(2), o(3), d*obj.n(1), d*obj.n(2), d*obj.n(3));
            text(o(1)+d*obj.n(1), o(2)+d*obj.n(2), o(3)+d*obj.n(3), t3, 'FontSize', 20);

            hdl.LineWidth = 3;
            hdl.Color = c;

            hold off

        end

        % print out contact frame
        function printOut(obj)
            fprintf('u = [%g, %g, %g]\n w = [%g, %g, %g]\n n = [%g, %g, %g]\n', ...
                obj.u(1), obj.u(2), obj.u(3), ...
                obj.w(1), obj.w(2), obj.w(3), ...
                obj.n(1), obj.n(2), obj.n(3))
        end


    end
end
