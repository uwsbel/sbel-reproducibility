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
  
classdef planeClass
    properties
        normal;  % normal direction 1*3 vector
        offset;  % x0, any point on the plane, 1*3 vector
        unitNormal;  % normal unit direction 1*3 vector
        A; B; C; D;  % Ax + By + Cz + D = 0, plane function
    end
    methods
        function obj = planeClass(n, x0)
            obj.normal = n;
            obj.offset = x0;
            obj.unitNormal = n/norm(n);
            obj.A = n(1); obj.B = n(2); obj.C = n(3);
            obj.D = -n'*x0;
        end
        
        function drawPlane(obj, x_start, x_end)
            space = (x_end - x_start)/4;
            n1 = obj.normal(1); n2 = obj.normal(2); n3 = obj.normal(3);
            x1 = obj.offset(1); y1 = obj.offset(2); z1 = obj.offset(3);
            
            if abs(n3) ~= 0
                [x,y] = meshgrid(x_start:space:x_end);
                z = z1 - n1/n3*(x-x1) - n2/n3*(y-y1);
            else
                
                if abs(n2) ~= 0
                    [x,z] = meshgrid(x_start:space:x_end);
                    y = y1 - n1/n2*(x-x1) - n3/n2*(z-z1);
                else
                    [y, z] = meshgrid(x_start:space:x_end);
                    x = x1 * ones(size(y));
                end
            end
            colorMap = 1.2*ones(size(z));
            myMesh = surf(x, y, z, colorMap);
            myMesh.FaceAlpha = 0.3;
            myMesh.EdgeAlpha = 0;
        end

        function drawPlaneColor(obj, x_start, x_end, colorVal)
            space = (x_end - x_start)/4;
            n1 = obj.normal(1); n2 = obj.normal(2); n3 = obj.normal(3);
            x1 = obj.offset(1); y1 = obj.offset(2); z1 = obj.offset(3);
            
            if abs(n3) ~= 0
                [x,y] = meshgrid(x_start:space:x_end);
                z = z1 - n1/n3*(x-x1) - n2/n3*(y-y1);
            else
                
                if abs(n2) ~= 0
                    [x,z] = meshgrid(x_start:space:x_end);
                    y = y1 - n1/n2*(x-x1) - n3/n2*(z-z1);
                else
                    [y, z] = meshgrid(x_start:space:x_end);
                    x = x1 * ones(size(y));
                end
            end
            colorMap = colorVal*ones(size(z));
            myMesh = surf(x, y, z, colorMap);
            myMesh.FaceAlpha = 0.3;
            myMesh.EdgeAlpha = 0;
        end
        
        % move plane in normal direction with a distance of d
        function newPlane = movePlaneByDist(obj, d)
            newOffset = obj.offset + d * obj.unitNormal;
            newPlane = planeClass(obj.normal, newOffset);
        end
        
        % closest distance of a point to a plane
        function dist = getDistanceFromPt(obj, pt)
            dist = abs(obj.A*pt(1) + obj.B*pt(2) + obj.C*pt(3) + obj.D)/sqrt(obj.A^2 + obj.B^2 + obj.C^2);
        end
        
        % return arbitrary point on plane
        function pt = getArbitraryPoint(obj)
            pt = zeros(3,1);
            pt(1) = rand(1);
            pt(2) = rand(1);
            pt(3) = 1/obj.C * (-obj.A*pt(1) - obj.B*pt(2) - obj.D);
        end
        
        % return tangent plane
        function [u, w] = getTangentPlane(obj)
            pt = getArbitraryPoint(obj);
            u = pt - obj.offset;  % find arbitrary vector on tangent plane
            u = u/norm(u);     % normalize the vector
            w = cross(obj.normal, u);
        end
        
        % return projected point onto the plance given an arbitray point
        function pt_proj = getProjectedPoint(obj, pt)
            if obj.isInPlane(pt)
                error('given point is already in plane!')
            end
                
            pt_proj = pt - (obj.normal' * (pt - obj.offset)) * obj.normal;
        end
        
        % return true if given point is in the plane, false otherwise
        function isInPlane = isInPlane(obj, pt)
            val = obj.A*pt(1) + obj.B*pt(2) + obj.C*pt(3) + obj.D;
            if val < 1e-8
                isInPlane = true;
            else
                isInPlane = false;
            end
        end
    end
end