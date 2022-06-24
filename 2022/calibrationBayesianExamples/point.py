# Copyright (c) 2022 Simulation-Based Engineering Lab, University of Wisconsin - Madison
# All rights reserved.
#
# BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.#


class Point:
    def __init__(self,initx,inity):
        self.x = initx
        self.y = inity

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def negx(self):
        return -(self.x)

    def negy(self):
        return -(self.y)

    def __str__(self):
        return 'x=' + str(self.x) + ', y=' + str(self.y)

    def halfway(self,target):
        midx = (self.x + target.x) / 2
        midy = (self.y + target.y) / 2
        return Point(midx, midy)

    def distance(self,target):
        xdiff = target.x - self.x
        ydiff = target.y - self.y
        dist = math.sqrt(xdiff**2 + ydiff**2)
        return dist

    def reflect_x(self):
        return Point(self.negx(),self.y)

    def reflect_y(self):
        return Point(self.x,self.negy())

    def reflect_x_y(self):
        return Point(self.negx(),self.negy())

    def slope_from_origin(self):
        if self.x == 0:
            return None
        else:
            return self.y / self.x

    def slope(self,target):
        if target.x == self.x:
            return None
        else:
            m = (target.y - self.y) / (target.x - self.x)
            return m

    def get_eq(self,target):
        c = -(self.slope(target)*self.x - self.y)
        def fun(t):
            return self.slope(target)*t + c
        return fun