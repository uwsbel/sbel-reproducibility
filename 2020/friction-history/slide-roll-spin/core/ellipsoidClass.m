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
  
classdef ellipsoidClass < handle
    properties
        a  % radius a, b, c
        b
        c
        
        position
        orientation
        eulerParameter
        velo
        omic
        acc
        omic_dot
        G
        D
        
        mass
        Ix
        Iy
        Iz
        massMatrix
        inertiaMatrix
        type
        inertia  % average inertia for calculating damping
        
    end
    methods
        % constructor for ellipse
        function obj = ellipsoidClass(varargin)
            switch nargin
                case 4    % radius and mass input
                    obj.a = varargin{1};
                    obj.b = varargin{2};
                    obj.c = varargin{3};
                    obj.mass = varargin{4};
                    obj.orientation = eye(3,3);
                    obj.position = zeros(3,1);
                case 6  % radius, mass, orientation and position input
                    obj.a = varargin{1};
                    obj.b = varargin{2};
                    obj.c = varargin{3};
                    obj.mass = varargin{4};
                    obj.orientation = varargin{5};
                    obj.position = varargin{6};
                    if (iscolumn(obj.position) == false)
                        error('Error! Needs a 3*1 vector for ellipsoid origin.')
                    end
            end
            
            obj.eulerParameter = getPfromA(obj.orientation);
            obj.velo = [0;0;0];
            obj.omic = [0;0;0];
            obj.acc = [0;0;0];
            obj.omic_dot = [0;0;0];
            
            obj.Ix = 1/5*obj.mass*(obj.b^2 + obj.c^2);
            obj.Iy = 1/5*obj.mass*(obj.c^2 + obj.a^2);
            obj.Iz = 1/5*obj.mass*(obj.a^2 + obj.b^2);
            
            obj.massMatrix = obj.mass*eye(3,3); % mass matrix
            obj.inertiaMatrix = diag([obj.Ix, obj.Iy, obj.Iz]); % intertia matrix
            
            
            obj.D = diag([1/obj.a^2;1/obj.b^2;1/obj.c^2]);
            obj.G = obj.orientation * obj.D * obj.orientation';
            
            obj.type = 'ellipsoid';
            obj.inertia = (obj.Ix * obj.Iy * obj.Iz)^(1/3);
        end
        
        %%% update the position of the ellipsoid
        function updateRotation(obj, newRot)
            obj.orientation = newRot;
            obj.G = obj.orientation * obj.D * obj.orientation';
        end
        
        %%% update the position of the ellipsoid
        function updatePosition(obj, newPos)
            if (iscolumn(newPos) == false)
                error('Error! Needs a 3*1 vector for ellipsoid updated position.')
            end
            obj.position = newPos;
        end
        
        % plot ellipsoid
        function myMesh = drawEllipsoid(obj)
            [x, y, z] = ellipsoid(0, 0, 0, obj.a, obj.b, obj.c, 10);
            [~, dim] = size(x);
            GRF_x = zeros(size(x));
            GRF_y = zeros(size(y));
            GRF_z = zeros(size(z));
            colorMap = zeros(size(x));
            
            for i = 1:dim
                for j = 1:dim
                    globalCoordinate = obj.orientation * [x(i,j); y(i,j); z(i,j)]...
                        + obj.position;
                    GRF_x(i,j) = globalCoordinate(1);
                    GRF_y(i,j) = globalCoordinate(2);
                    GRF_z(i,j) = globalCoordinate(3);
                    if mod(j,2) == 0 + mod(i,2) == 1
                        colorMap(i,j) = 0.2;
                    else
                        colorMap(i,j) = 1;
                    end
                    
                end
            end
            
            myMesh = surf(GRF_x, GRF_y, GRF_z, colorMap);
            myMesh.FaceAlpha = 0.6;  % transparent
            myMesh.EdgeAlpha = 0;    % no outlines
            
        end
        
        % return intersecting point with a plane defined by normal and
        % offset in the global coordinate
        function [x_star, lambda, costFunc] = findContactPointWithPlane(obj, plane)
            N = plane.normal;
            x0 = plane.offset;
            G_inv = obj.orientation * inv(obj.D) * obj.orientation';
            C = obj.position;
            lambda = N'*(x0-C)/(N'*G_inv*N);
            x_star = C + lambda * G_inv * N;
            costFunc = (x_star-C)' * obj.G * (x_star-C) - lambda*(N'*(x_star-x0));
            
        end
        
        % check if the ellipsoid is in contact with the plane
        % return true if it is in contact
        function isInContact = isInContactWithPlane(obj, plane)
            N = plane.normal;
            x0 = plane.offset;
            G_inv = obj.orientation * inv(obj.D) * obj.orientation';
            C = obj.position;
            x_star = C + (N'*(x0-C)/(N'*G_inv*N)) * G_inv * N;
            if (x_star-obj.position)'*obj.G*(x_star-obj.position) - 1 >= 0
                isInContact = false;
            else
                isInContact = true;
            end
        end
        
        
        % return curvature given any point on the ellipsoid in local
        % coordinate
        function K = getCurvatureAtLocalPt(obj, pt)
            if iscolumn(pt) == false
                error('Error! use 3*1 vector for point coordinate.');
            end
            
            %             if pt'*obj.D*pt - 1 > 1e-5
            %                 error('Error! given point locate outside the ellipsoid.');
            %             end
            %
            %             if pt'*obj.D*pt - 1 < -1e-5
            %                 error('Error! given point locate inside the ellipsoid.');
            %             end
            x = pt(1); y = pt(2); z = pt(3);
            helper = (x^2/obj.a^4 + y^2/obj.b^4 + z^2/obj.c^4)^(-0.5);
            K = helper^4/(obj.a^2 * obj.b^2 + obj.c^2);
        end
        
        % reutrn curvature given any point on ellipsoid in global
        % coordinate
        function K = getCurvatureAtGlobalPt(obj, pt)
            if iscolumn(pt) == false
                error('Error! use 3*1 vector for point coordinate.');
            end
            x_local = obj.orientation' * (pt - obj.position);
            K = getCurvatureAtLocalPt(obj, x_local);
        end
        
        % return curvature given any local point on a 2D ellipse
        function K = evaluateCurv(obj, pt, a, b)
            x = pt(1);
            y = pt(2);
            
            t = acos(x/a);
            if y < 0
                t = -t;
            end
            
            K = a * b / ((sqrt(a^2*sin(t)^2 + b^2*cos(t)^2))^3);            
        end
        
        % return unit surface normal vector in LRF given any point on the
        % ellipsoid in local coordinate
        function N = getLocalSurfaceNormalAtLocalPt(obj, pt)
            N = 2*[pt(1)/obj.a^2; pt(2)/obj.b^2; pt(3)/obj.c^2];
            N = N/norm(N);
        end
        
        % return unit surface normal vector in GRF given any point on the
        % ellipsoid in local coordinate
        function N = getGlobalSurfaceNormalAtLocalPt(obj, pt)
            n_local = getLocalSurfaceNormalAtLocalPt(obj, pt);
            N = obj.orientation * n_local;
        end
        
        % return unit surface normal vector in LRF given any point on the
        % ellipsoid in global coordinate
        function N = getLocalSurfaceNormalAtGlobalPt(obj, pt)
            pt_local = obj.orientation' * (pt - obj.position);
            N = getLocalSurfaceNormalAtLocalPt(obj, pt_local);
        end
        
        % return unit surface normal vector in GRF given any point on the
        % ellipsoid in global coordinate
        function N = getGlobalSurfaceNormalAtGlobalPt(obj, pt)
            pt_local = obj.orientation' * (pt - obj.position);
            N = getGlobalSurfaceNormalAtLocalPt(obj, pt_local);
        end
        
        % random generate an arbitrary point on the surface of the
        % ellipsoid in LRF
        function pt = randomGenPtOnSurface(obj)
            xa = rand(1) * obj.a * 2 - obj.a;
            xb = rand(1) * obj.b * 2 - obj.b;
            xc = sqrt((1 - xa^2/obj.a^2 - xb^2/obj.b^2) * obj.c^2);
            pt = [xa; xb; xc];
            
            if abs(pt'*obj.D*pt - 1) > 1e-10
                error('random point generated is NOT on ellipsoid surface, residual = %g', abs(pt'*obj.D*pt - 1));
            end
        end
        
        % find penetration depth given the plane normal and the minimum pt
        % plane in global reference frame
        function delta = getPenetrationDepth(obj, plane)
            N = plane.normal;
            xo = plane.offset;
            G_inv = obj.orientation * inv(obj.D) * obj.orientation';
            C = obj.position;
            lambda = N'*(xo-C)/(N'*G_inv*N);
            x_star = C + lambda * G_inv * N;
            
            % express plane in local frame (ellipsoid)
            N_local = obj.orientation' * N;
            offset_local = obj.orientation' * (xo - obj.position);
            x_star_local = obj.orientation' * (x_star - obj.position);
            
            % find a point x on surface such that x_star-x // plane normal
            A = N_local' * obj.D * N_local;
            B = 2*x_star_local'*obj.D*N_local;
            C = x_star_local'*obj.D*x_star_local - 1;
            
            % solve quadratic A*k^2 + B*k + C = 0
            val = B^2 - 4*A*C;
            if val < 0
                error('imaginary value');
            end
            
            k1 = (-B + sqrt(val))/(2*A); k2 = (-B - sqrt(val))/(2*A);
            delta = min(norm(k1*N), norm(k2*N));
            
        end
        
        % given any global point pt, find coordinates on the local sphere
        % reference frame
        function pt_local = expressGlobalPtInLocalRF(obj, pt_global)
            pt_local = obj.orientation'*(pt_global - obj.position);
        end
        
        % given any local point on the sphere, find global cooridnates
        function pt_global = expressLocalPtInGlobalRF(obj, pt_local)
            pt_global = obj.position + obj.orientation * pt_local;
        end
        
        function updateKinematics(obj, acc, omic_dot, dt)
            obj.acc = acc;
            obj.omic_dot = omic_dot;
            obj.velo = obj.velo + acc * dt;
            obj.omic = obj.omic + omic_dot * dt;
            obj.position = obj.position + obj.velo * dt;
            obj.eulerParameter = eulerParUpdate(obj.omic, obj.eulerParameter, dt);
            obj.orientation = getAfromP(obj.eulerParameter);
            obj.G = obj.orientation * obj.D * obj.orientation';
        end
        % draw reference frame in color 'c'
        function drawReferenceFrame(obj, c)
            A = obj.orientation;
            u = A(:,1); w = A(:,2); n = A(:,3);
            cm = obj.position;
            r = 1/3*(obj.a + obj.b + obj.c);
            quiver3(cm(1), cm(2), cm(3), r*u(1), r*u(2), r*u(3), c);
            hdl = quiver3(cm(1), cm(2), cm(3), r*u(1), r*u(2), r*u(3), c);
            hdl.LineWidth = 3;
            hold on
            hdl = quiver3(cm(1), cm(2), cm(3), r*w(1), r*w(2), r*w(3), c);
            hdl.LineWidth = 3;
            hdl = quiver3(cm(1), cm(2), cm(3), r*n(1), r*n(2), r*n(3), c);
            hdl.LineWidth = 3;
            hold off
            
        end
        
        % evaluate kinetic energy
        function KE = getKineticEnergy(obj)
            KE = 0.5 * obj.velo'*obj.massMatrix*obj.velo + 0.5 * obj.omic'*obj.inertiaMatrix*obj.omic;
        end
        
        
    end
end
