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
  
classdef frictionModel < handle
    properties
        mu_s
        mu_k
        sliding_stiffness
        rolling_stiffness
        normalForce
        slidingFr
        rollingTr
        spinningTr
        dt
        eta
        etaSpin
    end
    
    methods
        
        % constructor for sphere  
        % frictionModel(mu_s, mu_k, K_s, eta_roll, eta_spin, N, sphere, dt)
        % contsructor for ellipsoid 
        % frictionModel(mu_s, mu_k, K_s, eta_roll, eta_spin, N, , ellipsoid, dt, curvature)
        function obj = frictionModel(varargin)
            obj.mu_s = varargin{1};
            obj.mu_k = varargin{2};
            obj.sliding_stiffness = varargin{3};
            obj.etaSpin = varargin{5};
            obj.normalForce = varargin{6};
            object = varargin{7};
            obj.dt = varargin{8};
            obj.slidingFr.totalFriction = zeros(3,1);
            obj.rollingTr.totalFriction = zeros(3,1);
            obj.spinningTr.totalFriction = zeros(3,1);
            obj.eta = varargin{4};
            
            switch nargin
                case 8  % object being the sphere
                    if object.type ~= 'sphere'
                        error('sphere object needed.');
                    end
                    curvature = 1/object.radius;
                    
                    
                case 9  % object being the ellipsoid, curvature at contact point is needed
                    if object.type ~= 'ellipsoid'
                        error('ellipsoid object needed');
                    end
                    curvature = varargin{9};

            end
            
            obj.rolling_stiffness = 4*obj.eta/curvature^2 * obj.sliding_stiffness;
            
            slidingD = sqrt(object.mass * obj.sliding_stiffness);
            sliding_S_static = norm(obj.normalForce)*obj.mu_s/obj.sliding_stiffness;
            sliding_S_kinetic = norm(obj.normalForce)*obj.mu_k/obj.sliding_stiffness;
            obj.slidingFr = frictionComponent('sliding', sliding_S_static, sliding_S_kinetic, obj.sliding_stiffness, slidingD);
            
            

%            rollingD = 3*sqrt(object.inertia * obj.rolling_stiffness);
            rollingD = sqrt(object.inertia * obj.rolling_stiffness);
            
            rolling_S_static  = 1*sliding_S_static/2*curvature;
            rolling_S_kinetic = 1*sliding_S_kinetic/2*curvature; 
            obj.rollingTr = frictionComponent('rolling', rolling_S_static, rolling_S_kinetic, obj.rolling_stiffness, rollingD);
            
            
            if obj.etaSpin == 0 % hertzian contact based model
                
                normalForce = object.mass * 9.8;
                contactRadius = object.getContactRadius(normalForce)
                
                spinningK = 0.5 * contactRadius^2 * obj.sliding_stiffness;
                spinningD = 2*sqrt(object.inertia * spinningK);
                spinning_S_static   = sliding_S_static/contactRadius;
                spinning_S_kinetic = sliding_S_kinetic/contactRadius;
                maxFe = spinningK * spinning_S_static
            else
                % nonzero eta_spin, user defined parameter for eta_spin
                spinningK = obj.etaSpin * obj.sliding_stiffness / (curvature^2);
                spinningD = 4*sqrt(object.inertia * spinningK);
                spinning_S_static = curvature*sliding_S_static;
                spinning_S_kinetic = curvature*sliding_S_kinetic;
            end
            
            spinningK
            spinning_S_static
            obj.spinningTr = frictionComponent('spinning', spinning_S_static, spinning_S_kinetic, spinningK, spinningD);
            
        end
        
        function updateFrictionStiffnessAndSlack(obj, curvature, ellipsoid, normalForce)
            
            obj.normalForce = normalForce;
            obj.slidingFr.slackStatic  = norm(normalForce)*obj.mu_s/obj.sliding_stiffness;
            obj.slidingFr.slackKinetic = norm(normalForce)*obj.mu_k/obj.sliding_stiffness;
            
            
            obj.rollingTr.stiffness = 4*obj.eta/curvature^2 * obj.sliding_stiffness;            
%            obj.rollingTr.stiffness = 4*obj.eta/curvature * obj.sliding_stiffness;            

%            obj.rollingTr.stiffness = 2000;

%            obj.rollingTr.dampingCr = 0.005 * obj.rollingTr.stiffness / curvature^2;
%           obj.rollingTr.dampingCr = 2*sqrt(ellipsoid.inertia * obj.rollingTr.stiffness)/curvature;
           obj.rollingTr.dampingCr = 4*sqrt(ellipsoid.inertia * obj.rollingTr.stiffness);   
           %<== this is what used in tech
%           reports and slides

            obj.rollingTr.slackStatic  = obj.slidingFr.slackStatic /2*curvature;
            obj.rollingTr.slackKinetic = obj.slidingFr.slackKinetic/2*curvature; 
            
%            obj.spinningTr.stiffness = obj.etaSpin*obj.rollingTr.stiffness;  % this works really well for ellipsoid rolling
            obj.spinningTr.stiffness = obj.etaSpin * obj.sliding_stiffness/(curvature^2);
            obj.spinningTr.dampingCr = 2*sqrt(ellipsoid.inertia * obj.spinningTr.stiffness); 
            obj.spinningTr.slackStatic  = 2*obj.slidingFr.slackStatic;
            obj.spinningTr.slackKinetic = 2*obj.slidingFr.slackKinetic;
            
            
            
            
        end
        
        % update increment, history and damping coefficients for sliding,
        % rolling and spinning friction model
        function updateFrictionParameters(obj, delta, excursion, psi)
            obj.slidingFr.updateFrictionParameters(delta);
            obj.rollingTr.updateFrictionParameters(excursion);
            obj.spinningTr.updateFrictionParameters(psi);
        end
        
        % evaluate friction force or torque
        function evaluateForces(obj)
            obj.slidingFr.evaluateForces(obj.dt);
            obj.rollingTr.evaluateForces(obj.dt);
            obj.spinningTr.evaluateForces(obj.dt);
        end
        
        function printOutInfo(obj)
            obj.slidingFr.printOutInfo;
            obj.rollingTr.printOutInfo;
            obj.spinningTr.printOutInfo;
        end
    end
end

