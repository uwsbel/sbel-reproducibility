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

classdef spherePlaneContactModel < handle
    
    properties
        CF_prev_global         % contact frame in global frame at t0,          type: contactFrame
        CF_prev_local          % contact frame in local frame at t0,           type: contactFrame
        CF_prev_global_curr    % previous contact frame in global frame at t1, type: contactFrame
        CF_curr_global         % current contact frame in global frame at t1,  type: contactFrame
        CF_curr_local          % current contact frame in local frame at t1,   type: contactFrame
        
        CP_prev_global         % contact point in global frame at t0,          type: 3*1 vector
        CP_prev_local          % contact point in local frame at t0,           type: 3*1 vector
        CP_prev_global_curr    % previous contact point in global frame at t1, type: 3*1 vector
        CP_curr_global         % current contact point in global frame at t1,  type: 3*1 vector
        CP_curr_local          % current contact point in local frame at t1,   type: 3*1 vector
        
        
        initiationOfContact    % whether or not current time is the initiation of contact
        
    end
    
    methods
        function obj = spherePlaneContactModel(sphere, plane)
            obj.initiationOfContact = true;
            obj.CP_prev_global = plane.getProjectedPoint(sphere.position);
            obj.CP_prev_local = sphere.expressGlobalPtInLocalRF(obj.CP_prev_global);
            
            %            n0_global = sphere.position - obj.CP_prev_global;
            %            n0_global = n0_global/norm(n0_global);
            n0_global = plane.unitNormal;
            %%%%%%%%%%%%%%%%%%%%%%%%%
            % fix this uo_global not 1; 0; 0  for sure!%
            %%%%%%%%%%%%%%%%%%%%%%%%%
            u0_global = [1; 0; 0];
            w0_global = [0; n0_global(3); -n0_global(2)];
            
            %             [u0_global, w0_global] = plane.getTangentPlane;
            obj.CF_prev_global = contactFrame(u0_global, w0_global, n0_global);
            obj.CF_prev_local = obj.CF_prev_global.expressContactFrameInLocalRF(sphere.orientation);
            
            % change this later, now set previous and current value same
            obj.CP_curr_global = obj.CP_prev_global;
            obj.CP_prev_global_curr = sphere.expressLocalPtInGlobalRF(obj.CP_prev_local);
            obj.CP_curr_local  = obj.CP_prev_local;
            obj.CF_curr_global = obj.CF_prev_global;
            obj.CF_curr_local  = obj.CF_prev_local;
            
        end
        
        function updateContactAtNextTimeStep(obj, sphere, plane, slopeAngle)
            obj.initiationOfContact = false;
            
            obj.CF_prev_global_curr = obj.CF_prev_local.expressContactFrameInGlobalRF(sphere.orientation);
            obj.CP_prev_global_curr = sphere.expressLocalPtInGlobalRF(obj.CP_prev_local);
            
            
            a = obj.CF_prev_global_curr.u;
            b = obj.CF_prev_global_curr.w;
            c = obj.CF_prev_global_curr.n;
            
            
            %            fprintf('before: c = [%g, %g, %g]\n', c(1), c(2), c(3));
            % <=======> this needs to be working, look into this later,
            % Rotation matrix should rotate things back to normal
            % this is some major bug here... figure it out
            
            
            % calculate rotation matrix needed to align c to [0,0,1] configuration
            R_fromCtoZ = getRotationMatrixFromAtoB(c, [0;0;1]);
            %            slope_angle_deg = 1;
            slope_angle_deg = slopeAngle;
            PI = 3.141592653589793238462643383279;
            
            slope_angle = slope_angle_deg/180*PI;
            
            R_fromCtoZ = [ 1,   0                ,   0; ...
                0,   cos(slope_angle),   sin(slope_angle); ...
                0,  -sin(slope_angle),   cos(slope_angle)];
            
            newA = R_fromCtoZ * a;
            newB = R_fromCtoZ * b;
            
            % find minimum point
            a1 = newA(1); a2 = newA(2);
            b1 = newB(1); b2 = newB(2);
            PI = 3.141592653589793238462643383279;
            if abs(a1+b2) < 1e-16
                theta_star = PI/2;
            else
                theta_star = atan((a2-b1)/(a1+b2));
            end
            
            costFunc = (a1+b2)*cos(theta_star) + (a2-b1)*sin(theta_star);
            
            if costFunc < 0
                theta_star = theta_star + PI;
            end
            
            CF_u1_global = [ cos(theta_star); sin(theta_star); 0];
            CF_w1_global = [-sin(theta_star); cos(theta_star); 0];
            CF_n1_global = [0; 0; 1];   %%   <=====  look into this, what to do when this is not the global norm?
            
            % rotate back
            CF_u1_global = R_fromCtoZ' * CF_u1_global;
            CF_w1_global = R_fromCtoZ' * CF_w1_global;
            CF_n1_global = R_fromCtoZ' * CF_n1_global;
            
            %            fprintf('after: n = [%g, %g, %g]\n', CF_n1_global(1), CF_n1_global(2), CF_n1_global(3));
            
            obj.CF_curr_global = contactFrame(CF_u1_global, CF_w1_global, CF_n1_global);
            
            obj.CF_curr_local = obj.CF_curr_global.expressContactFrameInLocalRF(sphere.orientation);
            
            obj.CP_curr_global = plane.getProjectedPoint(sphere.position);
            obj.CP_curr_local = sphere.expressGlobalPtInLocalRF(obj.CP_curr_global);
            
        end
        
        %             % for the gound current contact frame
        %             u1bar = obj.CF_prev_global.u;
        %
        %             % for the sphere current contact frame
        %             u1 = obj.CF_curr_global.u;
        %             n1 = obj.CF_curr_global.n;
        %
        %             psi_cos = u1bar' * u1/(norm(u1bar)*norm(u1));
        %
        %             % determine whether or not psi should be positive
        %             if dot(cross(u1bar, u1), n1) > 0
        %                 psi = acos(psi_cos);
        %             else
        %                 psi = -acos(psi_cos);
        %             end
        %         end
        
        function replacePreviousContactFrame(obj)
            obj.CF_prev_global = obj.CF_curr_global;
            obj.CF_prev_local  = obj.CF_curr_local;
        end
        
        function replacePreviousContactPoint(obj)
            obj.CP_prev_global = obj.CP_curr_global;
            obj.CP_prev_local  = obj.CP_curr_local;
        end
    end
end
