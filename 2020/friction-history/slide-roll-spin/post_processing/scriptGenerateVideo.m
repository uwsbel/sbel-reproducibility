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

videoDirectory = '/Users/lulu/Documents/Research/code/friction3DSimEngine/movies/';
videoName = 'demo_ellipsoid_spinning_various_rollingK.avi';


vidObj = VideoWriter(strcat(videoDirectory, videoName));
vidObj.Quality = 100;
open(vidObj);
writeVideo(vidObj, Frames);
close(vidObj);