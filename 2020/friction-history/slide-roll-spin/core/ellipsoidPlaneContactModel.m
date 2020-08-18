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
  
classdef ellipsoidPlaneContactModel < handle
    
    properties
        CF_prev_global         % contact frame in global frame at t0,             type: contactFrame
        CF_prev_local          % contact frame in local frame at t0,              type: contactFrame
        CF_prev_global_curr    % previous contact frame in global frame at t1,    type: contactFrame
        CF_curr_global         % current contact frame in global frame at t1,     type: contactFrame
        CF_curr_local          % current contact frame in local frame at t1,      type: contactFrame
        CF_init_global         % contact frame at the inititation of the contact, type: contactFrame

        CP_prev_global         % contact point in global frame at t0,             type: 3*1 vector
        CP_prev_local          % contact point in local frame at t0,              type: 3*1 vector
        CP_prev_global_curr    % previous contact point in global frame at t1,    type: 3*1 vector
        CP_curr_global         % current contact point in global frame at t1,     type: 3*1 vector
        CP_curr_local          % current contact point in local frame at t1,      type: 3*1 vector

        initiationOfContact    % whether or not current time is the initiation of contact
        
    end
    
    methods
        % ===> TODO: what to do when contact normal changes
        % assume contact normal is always the same as the normal of the
        % plane
        function obj = ellipsoidPlaneContactModel(ellipsoid, plane)
            obj.initiationOfContact = true;
            % find contact point
            obj.CP_prev_global = ellipsoid.findContactPointWithPlane(plane);
            obj.CP_prev_local = ellipsoid.expressGlobalPtInLocalRF(obj.CP_prev_global);
            
            [u0_global, w0_global] = plane.getTangentPlane;
            
             u0_global = [1;0;0];
             w0_global = [0;1;0];
            
            n0_global = plane.unitNormal;
            obj.CF_prev_global = contactFrame(u0_global, w0_global, n0_global);

            
            
%            obj.CF_prev_global = obj.CF_init_global;
            obj.CF_prev_local = obj.CF_prev_global.expressContactFrameInLocalRF(ellipsoid.orientation);
            
            % change this later, now set previous and current value same
            obj.CP_curr_global = obj.CP_prev_global;
            obj.CP_prev_global_curr = ellipsoid.expressLocalPtInGlobalRF(obj.CP_prev_local);
            obj.CP_curr_local  = obj.CP_prev_local;
            obj.CF_curr_global = obj.CF_prev_global;
            obj.CF_curr_local  = obj.CF_prev_local;
                          
        end
        
        function updateContactAtNextTimeStep(obj, ellipsoid, plane)
            obj.initiationOfContact = false;
            
            obj.CF_prev_global_curr = obj.CF_prev_local.expressContactFrameInGlobalRF(ellipsoid.orientation);
            obj.CP_prev_global_curr = ellipsoid.expressLocalPtInGlobalRF(obj.CP_prev_local);
            

            a = obj.CF_prev_global_curr.u;
            b = obj.CF_prev_global_curr.w;
            a1 = a(1); a2 = a(2);
            b1 = b(1); b2 = b(2);
            PI = 3.141592653589793238462643383279;
            if abs(a1+b2) < 1e-16
                theta_star = PI/2;
%                fprintf('*******here********\n')
            else
                theta_star = atan((a2-b1)/(a1+b2));
            end
            
            costFunc = (a1+b2)*cos(theta_star) + (a2-b1)*sin(theta_star);
            
            if costFunc < 0
                theta_star = theta_star + PI;
            end
            
            CF_u1_global = [ cos(theta_star); sin(theta_star); 0];
            CF_w1_global = [-sin(theta_star); cos(theta_star); 0];
            CF_n1_global = plane.normal;   %%   <=====  look into this, what to do when this is not the global norm?
            
            obj.CF_curr_global = contactFrame(CF_u1_global, CF_w1_global, CF_n1_global);
            obj.CF_curr_local = obj.CF_curr_global.expressContactFrameInLocalRF(ellipsoid.orientation);
                    
            obj.CP_curr_global = ellipsoid.findContactPointWithPlane(plane);
            obj.CP_curr_local = ellipsoid.expressGlobalPtInLocalRF(obj.CP_curr_global);  
            
        end
        
        % this method not called
        function psi = evaluatePsi(obj)
            % for the gound current contact frame
            u1bar = obj.CF_prev_global.u;
            
            % for the sphere current contact frame
            u1 = obj.CF_curr_global.u;
            n1 = obj.CF_curr_global.n;
            
            psi_cos = u1bar' * u1/(norm(u1bar)*norm(u1));
            
            % determine whether or not psi should be positive
            if dot(cross(u1bar, u1), n1) > 0
                psi = acos(psi_cos);
            else
                psi = -acos(psi_cos);
            end
        end
        
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