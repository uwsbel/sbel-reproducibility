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

classdef sphereClass < handle
    properties
        radius
        position
        
        density
        youngsModulus
        poissonRatio
        
        
        
        orientation
        eulerParameter
        velo
        omic
        acc
        omic_dot
        
        mass
        inertia
        massMatrix
        inertiaMatrix
        
        type
        renderTransparency
        
    end
    
    methods
        % constructor for sphere
        function obj = sphereClass(varargin)
            switch nargin
                case 2
                    obj.radius = varargin{1};
                    obj.mass = varargin{2};
                    obj.position = [0;0;0];
                case 3
                    obj.radius = varargin{1};
                    obj.mass = varargin{2};
                    obj.position = varargin{3};
            end
            obj.orientation = eye(3);
            obj.eulerParameter = [1;0;0;0];
            obj.velo = [0;0;0];
            obj.omic = [0;0;0];
            obj.acc = [0;0;0];
            obj.omic_dot = [0;0;0];
            
            obj.inertia = 0.4*obj.mass*obj.radius^2;
            obj.massMatrix = obj.mass*eye(3,3);
            obj.inertiaMatrix = obj.inertia*eye(3,3);
            obj.type = 'sphere';
            obj.renderTransparency = 0.6;
            
        end
        
        % update orientation
        function updateOrientation(obj, newA)
            obj.orientation = newA;
            obj.eulerParameter = getPfromA(newA);
        end
        
        % update euler parameter
        function updateEulerParameter(obj, p)
            obj.eulerParameter = p;
            obj.orientation = getAfromP(p);
        end
        
        function updateKinematics(obj, acc, omic_dot, dt)
            obj.acc = acc;
            obj.omic_dot = omic_dot;
            obj.velo = obj.velo + acc * dt;
            obj.omic = obj.omic + omic_dot * dt;
            obj.position = obj.position + obj.velo * dt;
            obj.eulerParameter = eulerParUpdate(obj.omic, obj.eulerParameter, dt);
            obj.orientation = getAfromP(obj.eulerParameter);
        end
        
        function updateKinematicFromForce(obj, force, moment, dt)
            obj.acc = obj.massMatrix\force;
            obj.omic_dot = obj.inertiaMatrix\moment;
            obj.velo = obj.velo + obj.acc * dt;
            obj.omic = obj.omic + obj.omic_dot * dt;
            obj.position = obj.position + obj.velo * dt;
            obj.eulerParameter = eulerParUpdate(obj.omic, obj.eulerParameter, dt);
            obj.orientation = getAfromP(obj.eulerParameter);
        end
        
        function drawSphereSurface(obj, surfaceType)
            [X, Y, Z] = sphere(10);
            [~, dim] = size(X);
            GRF_x = zeros(size(X));
            GRF_y = zeros(size(Y));
            GRF_z = zeros(size(Z));
            colorMap = zeros(size(X));  % color for surface points
            
            for i = 1:dim
                for j = 1:dim
                    if length(obj.position) ~= 3
                        error('position needs to be a column vector.\n');
                    end
                    globalCoordinate = obj.orientation * obj.radius*[X(i,j); Y(i,j); Z(i,j)]...
                        + obj.position;
                    GRF_x(i,j) = globalCoordinate(1);
                    GRF_y(i,j) = globalCoordinate(2);
                    GRF_z(i,j) = globalCoordinate(3);
                    
                    switch (surfaceType)
                        case 'solid'
                            colorMap(i,j) = 1;
                        case 'checkerboard'
                            if mod(j,2) + mod(i,2) == 1  % checker board style
                                colorMap(i,j) = 0.2;
                            else
                                colorMap(i,j) = 1;
                            end
                        case 'watermelon'
                            if mod(j,2) == 0  % water melon style
                                colorMap(i,j) = 0.2;
                            else
                                colorMap(i,j) = 1;
                            end
                    end
                end
            end
            
            myMesh = surf(GRF_x, GRF_y, GRF_z, colorMap);
            myMesh.FaceAlpha = obj.renderTransparency;
            myMesh.EdgeAlpha = 0;
        end
        
        function drawSpherePoints(obj)
            PI = 3.14115926;
            th = 0:PI/10:2*PI;
            psi = 0:PI/10:PI; % coordinate z
            drawSphere = zeros(length(th)*length(psi), 3);
            
            count = 1;
            for ii = 1:length(psi)
                for jj = 1:length(th)
                    drawSphere(count,1) = obj.position(1) + obj.radius*cos(th(jj))*sin(psi(ii));
                    drawSphere(count,2) = obj.position(2) + obj.radius*sin(th(jj))*sin(psi(ii));
                    drawSphere(count,3) = obj.position(3) + obj.radius*cos(psi(ii));
                    count = count + 1;
                end
            end
            
            plot3(drawSphere(:,1), drawSphere(:,2), drawSphere(:,3),'.');
        end
        
        % draw reference frame in color 'c'
        function drawReferenceFrame(obj, c)
            LW = 4;
            A = obj.orientation;
            u = A(:,1); w = A(:,2); n = A(:,3);
            cm = obj.position;
            r = obj.radius;
            hdl = quiver3(cm(1), cm(2), cm(3), r*u(1), r*u(2), r*u(3), c);
            hdl.LineWidth = LW;
            hold on
            hdl = quiver3(cm(1), cm(2), cm(3), r*w(1), r*w(2), r*w(3), c);
            hdl.LineWidth = LW;
            hdl = quiver3(cm(1), cm(2), cm(3), r*n(1), r*n(2), r*n(3), c);
            hdl.LineWidth = LW;
            hold off
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
        
        % evaluate kinetic energy
        function KE = getKineticEnergy(obj)
            KE = 0.5 * obj.velo'*obj.massMatrix*obj.velo + 0.5 * obj.omic'*obj.inertiaMatrix*obj.omic;
        end
        
        function contactRadius = getContactRadius(obj, normalF)
%            k_hz = 4/3 * obj.youngsModulus / (1 - obj.poissonRatio^2) * sqrt(obj.radius);
            k_hz = 4/3 * obj.youngsModulus / (1 - obj.poissonRatio^2) * sqrt(1/29);

            delta = (normalF / k_hz) ^ (2/3);
%            contactRadius = sqrt(obj.radius * delta);

            contactRadius = sqrt(1/29 * delta);

        end
        
    end
    
    
end

