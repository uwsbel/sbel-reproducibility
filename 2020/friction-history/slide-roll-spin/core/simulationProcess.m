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
  
classdef simulationProcess < handle
    properties
        name
        endTime
        dt
        currTime
        numSteps
        generateMovie
        movieLoops
        frames = struct('cdata', [], 'colormap', []);
        currFrame
    end
    
    methods
        function obj = simulationProcess(varargin)
            switch nargin
                case 2
                    obj.name = 'untitled';
                    obj.endTime = varargin{1};
                    obj.dt = varargin{2};
                    obj.currTime = 0;
                case 3
                    obj.endTime = varargin{1};
                    obj.dt = varargin{2};
                    obj.name = varargin{3};
            end
            obj.numSteps = obj.endTime/obj.dt + 1;
            obj.currTime = 0;
            obj.generateMovie = false;
            obj.movieLoops = 0;
%            obj.frames(2) = struct('cdata', [], 'colormap', []);
        end
        
        function updateTime(obj)
            obj.currTime = obj.currTime + obj.dt;
        end
        
        function generateFrameStruct(obj, framesPerSec)
            obj.movieLoops = floor(framesPerSec * obj.endTime);
            obj.frames(obj.movieLoops) = struct('cdata', [], 'colormap', []);
            obj.generateMovie = true;
            obj.currFrame = 1;
        end
        
        function writeCurrentFrame(obj)
            obj.frames(obj.currFrame) = getframe(gcf);
            obj.currFrame = obj.currFrame + 1;
        end
            
        function writeMovies(obj, filename)
            if obj.generateMovie == true
            vidObj = VideoWriter(strcat(filename, '.avi'));
            vidObj.Quality = 100;
            open(vidObj);
            writeVideo(vidObj, obj.frames);
            close(vidObj);
            end
        end
            
    end
    
end

