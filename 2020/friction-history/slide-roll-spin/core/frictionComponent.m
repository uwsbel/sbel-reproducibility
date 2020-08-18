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
  
classdef frictionComponent < handle
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        type         % type of friction rolling, spinning or sliding
        mode         % type of mode, static or kinetic
        slackStatic
        slackKinetic
        increment    % different type gives different way of evaluating increment and history
        history
        stiffness
        dampingCr    % 2*sqrt(mass * K) value doesn't change
        dampingCo    % actual damping coefficient being calculated
        elasticComponent
        plasticComponent
        totalFriction
%        comesToStop  % only adds damping when sphere is coming to a stop
        
    end
    
    methods
        function obj = frictionComponent(type, ss, sk, K, D)
            obj.type = type;
            obj.mode = 's';
            obj.slackStatic = ss;
            obj.slackKinetic = sk;
            obj.stiffness = K;
            obj.dampingCr = D;
            obj.dampingCo = D;
            obj.elasticComponent = zeros(3,1);
            obj.plasticComponent = zeros(3,1);
            obj.totalFriction = obj.elasticComponent + obj.plasticComponent;
%            obj.comesToStop = false;
            
            switch type
                case 'sliding'
                    obj.increment = zeros(3,1); obj.history = zeros(3,1);                
                case 'rolling'
                    obj.increment = zeros(3,1); obj.history = zeros(3,1);
                case 'spinning'
                    obj.increment = 0; obj.history = 0; obj.mode = 'k';
            end
        end
        
        % update friction forces with new increment
        function updateFrictionParameters(obj, delta)
            obj.increment = delta;
            obj.history = obj.history + delta;
            slack = norm(obj.history);
            if obj.mode == 's'
                alphaStatic = slack/obj.slackStatic;
                dampingC = 1;
                if alphaStatic > 1
                    obj.history = obj.history/alphaStatic;
                    obj.mode = 'k';
                    dampingC = 0;   %% <==== was set to zero for sphere case!!!
                end
            else
                alphaKinetic = slack/obj.slackKinetic;
                if alphaKinetic > 1 %% change here and test?
                    obj.history = obj.history/slack*obj.slackKinetic;
                    dampingC = 0;  %% <=== was set to zero for sphere case!!
                else
                    obj.mode = 's';
                    dampingC = 1; 
                end
            end
            obj.dampingCo = obj.dampingCr * dampingC; 
        end
        
        % return friction force or torque
        function evaluateForces(obj, dt)
            obj.elasticComponent = obj.stiffness * obj.history;
            obj.plasticComponent = obj.dampingCo * obj.increment/dt;
            obj.totalFriction = obj.elasticComponent + obj.plasticComponent;
        end
        
        
        % print out information
        function printOutInfo(obj)
            if strcmp(obj.type, 'spinning')
                fprintf('-----------------------------\n');
                fprintf('%s friction (%s) info:\n', obj.type, obj.mode);
                fprintf('static slack %g, kinetic slack %g \n', obj.slackStatic, obj.slackKinetic);
                fprintf('increment = [%g, %g, %g], history = [%g, %g, %g]\n', ...
                    obj.increment(1), obj.history(1));
                fprintf('stiffness K = %g, damping D = %g\n', obj.stiffness, obj.dampingCo);
                fprintf('|%g| = |%g| + |%g|\n', obj.totalFriction(1), obj.elasticComponent(1), obj.plasticComponent(1));
                
            else
                fprintf('-----------------------------\n');
                fprintf('%s friction (%s) info:\n', obj.type, obj.mode);
                fprintf('static slack %g, kinetic slack %g \n', obj.slackStatic, obj.slackKinetic);
                fprintf('increment = [%g, %g, %g], history = [%g, %g, %g]\n', ...
                    obj.increment(1), obj.increment(2), obj.increment(3), ...
                    obj.history(1), obj.history(2), obj.history(3));
                fprintf('stiffness K = %g, damping D = %g\n', obj.stiffness, obj.dampingCo);
                fprintf('|%g|   |%g| + |%g|\n', obj.totalFriction(1), obj.elasticComponent(1), obj.plasticComponent(1));
                fprintf('|%g| = |%g| + |%g|\n', obj.totalFriction(2), obj.elasticComponent(2), obj.plasticComponent(2));
                fprintf('|%g|   |%g| + |%g|\n', obj.totalFriction(3), obj.elasticComponent(3), obj.plasticComponent(3));
                
            end
           
        end
    end
  
end

