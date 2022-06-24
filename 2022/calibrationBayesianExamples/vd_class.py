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


from turtle import st
import warnings
import numpy as np
import types
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

'''
This file contains a class implementation of 2dof and 8dof vehicle models
'''

class vd_2dof:
    def __init__(self, parameters=None, states=None):
        self.g = 9.8 # Gravitational forces

        # Friction parameters for the tire model
        self.umin = 0.5568
        self.umax = 0.9835
        # Relaxataion lengths - not used
        self.xrel = 1.0
        self.yrel = 1.0

        # The smoothing time - Commented out currently
        self.trans_time = 0.2

        # Debug arrays
        self.t_arr = []
        self.ff = []
        self.fr = []
        self.dt = []
        self.fdt= []
        self.rdt= []
        self.s_arr= []
        self.xtf_ar= []

        self.debug = 0
        if parameters is not None:
            if isinstance(parameters, list):
                raise Exception(
                    "Please provide a dictionary for the parameters")
            try:
                # distance of c.g. from front axle (m)
                self.a = parameters['a']
            except:
                self.a = 1.14
                warnings.warn(
                    f"Set 'a' to default value {self.a}", UserWarning)
            try:
                # distance of c.g. from rear axle  (m)
                self.b = parameters['b']
            except:
                self.b = 1.4
                warnings.warn(
                    f"Set 'b' to default value {self.b}", UserWarning)
            try:
                # height of c.g. from ground (m)
                self.h = parameters['h']
            except:
                self.h = 0.713
                warnings.warn(
                    f"Set 'h' to default value {self.h}", UserWarning)
            try:
                # front axle cornering stiffness (N/rad)
                self.Cf = parameters['Cf']
            except:
                self.Cf = 88000
                warnings.warn(
                    f"Set 'Cf' to default value {self.Cf}", UserWarning)
            try:
                # rear axle cornering stiffness (N/rad)
                self.Cr = parameters['Cr']
            except:
                self.Cr = 88000
                warnings.warn(
                    f"Set 'Cr' to default value {self.Cr}", UserWarning)
            try:
                # front axle longitudinal stiffness (N)
                self.Cxf = parameters['Cxf']
            except:
                self.Cxf = 10000
                warnings.warn(
                    f"Set 'Cxf' to default value {self.Cxf}", UserWarning)
            try:
                # rear axle longitudinal stiffness (N)
                self.Cxr = parameters['Cxr']
            except:
                self.Cxr = 10000
                warnings.warn(
                    f"Set 'Cxr' to default value {self.Cxr}", UserWarning)
            try:
                # tire stiffness - front
                self.ktf = parameters['ktf']
            except:
                self.ktf = 326332
                warnings.warn(
                    f"Set 'ktf' to default value {self.ktf}", UserWarning)
            try:
                # tire stiffness - rear
                self.ktr = parameters['ktr']
            except:
                self.ktr = 326332
                warnings.warn(
                    f"Set 'ktr' to default value {self.ktr}", UserWarning)
            try:
                self.m = parameters['m']  # the mass of the vehicle (kg)
            except:
                self.m = 1720
                warnings.warn(
                    f"Set 'm' to default value {self.m}", UserWarning)
            try:
                self.muf = parameters['muf']  # the front unsprung mass
            except:
                self.muf = 1720
                warnings.warn(
                    f"Set 'muf' to default value {self.muf}", UserWarning)
            try:
                self.mur = parameters['mur']  # the rear unsprung mass
            except:
                self.mur = 1720
                warnings.warn(
                    f"Set 'mur' to default value {self.mur}", UserWarning)

            try:
                self.Jz = parameters['Jz']  # yaw moment of inertia (kg.m^2)
            except:
                self.Jz = 2420
                warnings.warn(
                    f"Set 'Jz' to default value {self.Jz}", UserWarning)
            try:
                self.r0 = parameters['r0']  # wheel radius
            except:
                self.r0 = 0.285
                warnings.warn(
                    f"Set 'r0' to default value {self.r0}", UserWarning)
            try:
                self.Jw = parameters['Jw']  # wheel roll inertia
            except:
                self.Jw = 2
                warnings.warn(
                    f"Set 'Jw' to default value {self.Jw}", UserWarning)
            try:
                self.rr = parameters['rr']  # wheel roll inertia
            except:
                self.rr = 0.0125
                warnings.warn(
                    f"Set 'rr' to default value {self.rr}", UserWarning)

            # A dictionary of parameters
            self.params = {"a": self.a, "b": self.b, "h": self.h, "Cf": self.Cf, "Cr": self.Cr, "Cxf": self.Cxf, "Cxr": self.Cxr,"ktf": self.ktf,
                           "ktr": self.ktr, "m": self.m, "muf": self.muf, "mur": self.mur, "Jz": self.Jz, "r0": self.r0, "Jw": self.Jw, "rr": self.rr}
        else:
            self.a = 1.14  # distance of c.g. from front axle (m)
            self.b = 1.4  # distance of c.g. from rear axle  (m)
            self.h = 0.713 # height of c.g. from the ground (m)
            self.Cf = 88000  # front axle cornering stiffness (N/rad)
            self.Cr = 88000  # rear axle cornering stiffness (N/rad)
            self.Cxf = 10000  # front axle longitudinal stiffness (N)
            self.Cxr = 10000  # rear axle longitudinal stiffness (N)
            self.ktf = 326332 # tire stiffness - front (N/m)
            self.ktr = 326332 # tire stiffmess - rear (N/m)
            self.m = 1720  # the mass of the vehicle (kg)
            self.muf = 120 # the front unsprung mass (kg)
            self.mur = 120 # the rear unsprung mass (kg)
            self.Jz = 2420  # yaw moment of inertia (kg.m^2)
            self.r0 = 0.285  # wheel radius
            self.Jw = 2  # wheel roll inertia
            self.rr = 0.0125 # Rolling resistance
            # A dictionary of parameters
            self.params = {"a": self.a, "b": self.b, "h": self.h, "Cf": self.Cf, "Cr": self.Cr, "Cxf": self.Cxf, "Cxr": self.Cxr,"ktf": self.ktf,
                           "ktr": self.ktr, "m": self.m, "muf": self.muf, "mur": self.mur, "Jz": self.Jz, "r0": self.r0, "Jw": self.Jw, "rr": self.rr}
            warnings.warn("Set parameters to default values" +
                          '\n' + f"{self.params}", UserWarning)

        if states is not None:
            if isinstance(states, list):
                raise Exception("Please provide a dictionary for the states")

            # State of the vehicle
            try:
                self.x = states['x']  # lateral velocity
            except:
                self.x = 0.
                warnings.warn(
                    f"Set 'x' to default value {self.x}", UserWarning)
            try:
                self.y = states['y']  # lateral velocity
            except:
                self.y = 0.
                warnings.warn(
                    f"Set 'y' to default value {self.y}", UserWarning)
            try:
                self.Vy = states['Vy']  # lateral velocity
            except:
                self.Vy = 0.
                warnings.warn(
                    f"Set 'Vy' to default value {self.Vy}", UserWarning)
            try:
                self.Vx = states['Vx']
            except:
                self.Vx = 50./3.6
                warnings.warn(
                    f"Set 'Vx' to default value {self.Vx}", UserWarning)
            try:
                self.psi = states['psi']  # yaw rate
            except:
                self.psi = 0.
                warnings.warn(
                    f"Set 'psi' to default value {self.psi}", UserWarning)
            try:
                self.psi_dot = states['psi_dot']  # yaw rate
            except:
                self.psi_dot = 0.
                warnings.warn(
                    f"Set 'psi_dot' to default value {self.psi_dot}", UserWarning)
            try:
                self.wf = states['wf']  # Front wheel angular velocity
            except:
                self.wf = 50./(3.6 * 0.285)
                warnings.warn(
                    f"Set 'wf' to default value {self.wf}", UserWarning)
            try:
                self.wr = states['wr']  # Rear wheel angular velocity
            except:
                self.wr = 50./(3.6 * 0.285)
                warnings.warn(
                    f"Set 'wr' to default value {self.wr}", UserWarning)
            # A dictionary of states
            self.states = {'x' : self.x,'y':self.y,'Vx': self.Vx, 'Vy': self.Vy,'psi' : self.psi,
                           'psi_dot': self.psi_dot, 'wf': self.wf, 'wr': self.wr}
        else:
            # State of the vehicle
            self.x = 0. # x coordinate
            self.y = 0. # y coordinate
            self.Vy = 0.  # lateral velocity
            self.Vx = 50./3.6  # longitudinal velocity
            self.psi = 0. # Yaw angle
            self.psi_dot = 0.  # yaw rate
            self.wf = 50./(3.6 * 0.285)  # Front wheel angular velocity
            self.wr = 50./(3.6 * 0.285)  # Rear wheel angular velocity
            # A dictionary of states
            self.states = {'x' : self.x,'y':self.y,'Vx': self.Vx, 'Vy': self.Vy,'psi' : self.psi,
                           'psi_dot': self.psi_dot, 'wf': self.wf, 'wr': self.wr}
            warnings.warn("Set States to default values" +
                          '\n' + f"{self.states}", UserWarning)
        
        # Vertical forces initially
        self.Fzgf = ((self.m*self.g*self.b) /
                        (2*(self.a+self.b))+self.muf*self.g) # front
        self.Fzgr = ((self.m*self.g*self.b) /
                        (2*(self.a+self.b))+self.mur*self.g) # rear

        # the initial tire compression xt
        self.xtf = self.Fzgf/self.ktf  # front
        self.xtr = self.Fzgr/self.ktr # rear


    # Simple print to show the parameters and states
    def __str__(self):
        return str(self.__class__) + '\n' + "Vehicle Parameters are" + '\n' + f"{self.params}" + '\n' + "Vehicle state is" + '\n' + f"{self.states}"

    # Sets the steering of the vehicle
    def set_steering(self, steering, max_steering=0.6525249):
        """
        Takes as input a normalised steering function from -1 to 1 and a max steering value in radians. +ve is a left turn
        """
        # Pitman arm
        # self.max_steer = 0.7328647
        self.steering = steering
        self.max_steer = max_steering

    # Sets the throttle controls of the vehicle
    def set_throttle(self, throttle, gr=0.3*0.2, mt=1000., ms=2000.):
        """
        Takes as input a normalised throttle fron -1 to 1, a gear ratio, a max torque(Nm) and a max speed (RPS?)
	    The throttle is then applied in the drive_torque function along with the other paramters
        """
        # The 0.2 is from the conical gears of the chrono vehicle
        self.throttle = throttle
        self.gear_ratio = gr
        self.max_torque = mt
        self.max_speed = ms
	
	#Sets the brake controls of the vehicle
    def set_braking(self,brake,mbt = 4000):
        self.brake = brake
        self.max_brake_torque = mbt


    # Update the states externally after initilisation - Through dict or kwargs
    def update_states(self,state_dict = None, **kwargs):
        if(state_dict == None):
            for key, value in kwargs.items():
                if(key not in self.states):
                    raise Exception(f"{key} is not vehicle state")

                # Set the state to the respective class attribute
                setattr(self, key, value)
                # Update the states dict attribute as well
                self.states[key] = value
        else:
            for key,value in state_dict.items():
                self.states[key] = value


    # Update the parameters
    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if(key not in self.params):
                raise Exception(f"{key} is not vehicle parameter")

            # Set the parameter to the respective class attribute
            setattr(self, key, value)
            # Update the params dict of the vehicle as well
            self.params[key] = value

    # Returns a list of the parameters
    def get_params(self):
        return self.params

    # Returns a list of the current states
    def get_states(self):
        return self.states


    # Just a method to reset the state to default values or values specified when vehicle object was created
    # Needed for running multiple simulations
    def reset_state(self,init_state):
        for key,value in init_state.items():
            self.states[key] = value

        # Vertical forces initially
        self.Fzgf = ((self.m*self.g*self.b) /
                        (2*(self.a+self.b))+self.muf*self.g) # front
        self.Fzgr = ((self.m*self.g*self.b) /
                        (2*(self.a+self.b))+self.mur*self.g) # rear

        # the initial tire compression xt
        self.xtf = self.Fzgf/self.ktf  # front
        self.xtr = self.Fzgr/self.ktr # rear


        if(self.debug):
            self.t_arr = []
            self.ff = []
            self.fr = []
            self.dt = []
            self.fdt= []
            self.rdt= []
            self.s_arr= []
            self.xtf_ar= []

    # Evaluate the drive torque based on the wheel speed - Similar to 4 motors on each of the wheels
    def drive_torque(self, t, w):
        motor_speed = w / self.gear_ratio
        motor_torque = self.max_torque - \
            (motor_speed * (self.max_torque / self.max_speed))
        motor_torque = motor_torque * self.throttle(t)
        return motor_torque / self.gear_ratio
    
    # Evaluate the brake torque at a paticular time step
    def brake_torque(self,t):
        return self.brake(t)*self.max_brake_torque


    # A utility function to stabilsie the initial transients
    def smooth_step(self, t, f1, t1, f2, t2):
        if(t < t1):
            return f1
        elif(t >= t2):
            return f2
        else:
            return f1 + ((f2 - f1)*((t-t1)/(t2-t1))**2 * (3 - 2*((t - t1)/(t2-t1))))

    # returns the differential equations of the model for a linear tire
    def model_linear(self, t, state):
        g = self.g
        a, b, h, Cf, Cr, Cxf, Cxr, ktf, ktr, m, muf, mur, Jz, r0, Jw, rr = list(self.params.values())
        x, y, Vx, Vy, psi, psi_dot, wf, wr = state

        mt=m+2*muf+2*mur # vehicle total mass
        # instataneous tire radius
        Rf=r0-self.xtf
        Rr=r0-self.xtr
        huf = Rf
        hur = Rr

        st_a = self.steering(t) * self.max_steer
        # Longitudinal slip
        sf = (Rf*wf-(Vx*np.cos(st_a)+(Vy+a*psi_dot)*np.sin(st_a))) / \
            np.abs(Vx*np.cos(st_a) +
                   (Vy+a*psi_dot)*np.sin(st_a))
        sr = (Rr*wr-Vx)/np.abs(Vx)

        # Lateral slip
        delta_f = np.arctan2((Vy+a*psi_dot),Vx)-(st_a)
        delta_r = np.arctan2((Vy-b*psi_dot),Vx)

        # Longitufinal force
        Fxtf = Cxf*sf
        Fxtr = Cxr*sr

        # Lateral force
        Fytf = -Cf*delta_f
        Fytr = -Cr*delta_r

        # ODE's
        Vy_dot = -Vx*psi_dot + (Fxtf*np.sin(st_a) + Fytf*np.cos(st_a) + Fytr)/mt
        Vx_dot = Vy*psi_dot + (Fxtf*np.cos(st_a) - Fytf*np.sin(st_a) + Fxtr)/mt
        dpsi_dot = (a*(Fytf*np.cos(st_a) + Fxtf*np.sin(st_a)) - b*Fytr)/Jz
        dx = Vx*np.cos(psi) - Vy*np.sin(psi)
        dy = Vx*np.sin(psi) + Vy*np.cos(psi)
        dpsi = psi_dot

        # rolling resistance
        rolling_res_f = -rr * np.abs(self.Fzgf) * np.sign(wf)
        rolling_res_r = -rr * np.abs(self.Fzgr) * np.sign(wr)

        # Brake Torques of each wheel
        br_t_f = - np.sign(wf) * self.brake_torque(t)
        br_t_r = - np.sign(wr) * self.brake_torque(t)


        # Wheel rotational model - Dividing by 4 still - this is slighly confusing as there are only 2 wheels
        # However the point is to apply a torque that is the same as the chrono vehicle which has 4 wheels
        dwf=(1/Jw)*(self.drive_torque(t,wf)/4 + rolling_res_f + br_t_f - Fxtf*Rf)
        dwr=(1/Jw)*(self.drive_torque(t,wr)/4 + rolling_res_r + br_t_r - Fxtr*Rr)


        # The normal forces at four tires are determined using d'alemberts principle
        Z1 = (m*g*b)/(2*(a+b))+(muf*g)/2
        Z4 = ((m*h+muf*huf+mur*hur)*(Vx_dot-psi_dot*Vy))/(2*(a+b))
        self.Fzgf = Z1-Z4
        Z5 = (m*g*a)/(2*(a+b))+(mur*g)/2
        Z8 = ((m*h+muf*huf+mur*hur)*(Vx_dot-psi_dot*Vy))/(2*(a+b))
        self.Fzgr = Z5+Z8

        # These vertical forces cannot be lesser than 0 as that means that the wheel is off the ground
        if(self.Fzgf < 0):
            self.Fzgf = 0
        if(self.Fzgr < 0):
            self.Fzgr = 0

        # Tire deflection calculated using the tire stiffness and the normal force on the tires
        self.xtf = self.Fzgf/ktf
        self.xtr = self.Fzgr/ktr


        if(self.debug):
            self.t_arr.append(t)
            self.ff.append(self.Fzgf)
            self.fr.append(self.Fzgr)
            self.dt.append(self.drive_torque(t, wr)/4)
            self.fdt.append(Fxtf*Rf)
            self.rdt.append(rolling_res_f)
            self.s_arr.append(sr)
            self.xtf_ar.append(self.xtf)

        return np.stack([dx, dy, Vx_dot, Vy_dot, dpsi, dpsi_dot, dwf, dwr])

    def model_fiala(self, t, state):
        g = self.g
        a, b, h, Cf, Cr, Cxf, Cxr, ktf, ktr, m, muf, mur, Jz, r0, Jw, rr = list(self.params.values())
        x, y, Vx, Vy, psi, psi_dot, wf, wr = state

        mt=m+2*muf+2*mur # vehicle total mass

        # instataneous tire radius
        Rf=r0-self.xtf
        Rr=r0-self.xtr
        huf = Rf
        hur = Rr

        # the steering angle inputted to the wheels
        st_a = self.steering(t) * self.max_steer

        # longitudinal slip
        sf = (Rf*wf-(Vx*np.cos(st_a)+(Vy+a*psi_dot)*np.sin(st_a))) / \
            np.abs(Vx*np.cos(st_a) +
                   (Vy+a*psi_dot)*np.sin(st_a))
        sr = (Rr*wr-Vx)/np.abs(Vx)

        # lateral slip
        delta_f = np.arctan2((Vy+a*psi_dot),Vx)-(st_a)
        delta_r = np.arctan2((Vy-b*psi_dot),Vx)

        # comprehensive slip ratio
        ss_f = min(np.sqrt(sf**2 + np.tan(delta_f)**2), 1.)
        ss_r = min(np.sqrt(sr**2 + np.tan(delta_r)**2), 1.)

        # Coefficient of friction based on the comprehensive slip
        u_f = self.umax - (self.umax - self.umin)*ss_f
        u_r = self.umax - (self.umax - self.umin)*ss_r

        # critical longitudinal slip
        s_crit_f = np.abs((u_f * self.Fzgf) / (2 * Cxf))
        s_crit_r = np.abs((u_r * self.Fzgr) / (2 * Cxr))

        # critical lateral slip
        al_crit_f = np.arctan((3*u_f * np.abs(self.Fzgf))/Cf)
        al_crit_r = np.arctan((3*u_r * np.abs(self.Fzgr))/Cr)


        # longitudinal forces based on whether the tire is in the elastic regime or the sliding regime
        if(np.abs(sf) < s_crit_f):
            Fxtf = Cxf*sf
        else:
            Fxtf_1 = u_f * np.abs(self.Fzgf)
            Fxtf_2 = np.abs((u_f * self.Fzgf)**2 / (4 * sf * Cxf))
            Fxtf = np.sign(sf)*(Fxtf_1 - Fxtf_2)
        
        if(np.abs(sr) < s_crit_r):
            Fxtr = Cxr*sr
        else:
            Fxtr_1 = u_r * np.abs(self.Fzgr)
            Fxtr_2 = np.abs((u_r * self.Fzgr)**2 / (4 * sr * Cxr))
            Fxtr = np.sign(sr)*(Fxtr_1 - Fxtr_2)

        # lateral forces based on whether the tire is in elastic regime or the sliding regime 
        if(np.abs(delta_f) <= al_crit_f):
            h_ = 1 - ((Cf * np.abs(np.tan(delta_f))) / (3 * u_f * np.abs(self.Fzgf)))
            Fytf = -u_f * np.abs(self.Fzgf) * (1-h_**3)*np.sign(delta_f)
        else:
            Fytf = -u_f * np.abs(self.Fzgf) * np.sign(delta_f)

        if(np.abs(delta_r) <= al_crit_r):
            h_ = 1 - ((Cr * np.abs(np.tan(delta_r))) / (3 * u_r * np.abs(self.Fzgr)))
            Fytr = -u_r * np.abs(self.Fzgr) * (1-h_**3)*np.sign(delta_r)
        else:
            Fytr = -u_r * np.abs(self.Fzgr) * np.sign(delta_r)

        # ODE's
        Vy_dot = -Vx*psi_dot + (Fxtf*np.sin(st_a) + Fytf*np.cos(st_a) + Fytr)/mt
        Vx_dot = Vy*psi_dot + (Fxtf*np.cos(st_a) - Fytf*np.sin(st_a) + Fxtr)/mt
        dpsi_dot = (a*(Fytf*np.cos(st_a) + Fxtf*np.sin(st_a)) - b*Fytr)/Jz
        dx = Vx*np.cos(psi) - Vy*np.sin(psi)
        dy = Vx*np.sin(psi) + Vy*np.cos(psi)
        dpsi = psi_dot

        # rolling resistance
        rolling_res_f = -rr * np.abs(self.Fzgf) * np.sign(wf)
        rolling_res_r = -rr * np.abs(self.Fzgr) * np.sign(wr)

        # Brake Torques of each wheel
        br_t_f = - np.sign(wf) * self.brake_torque(t)
        br_t_r = - np.sign(wr) * self.brake_torque(t)


        # Wheel rotational model - Dividing by 4 still - this is slighly confusing as there are only 2 wheels
        # However the point is to apply a torque that is the same as the chrono vehicle which has 4 wheels
        dwf=(1/Jw)*(self.drive_torque(t,wf)/4 + rolling_res_f + br_t_f - Fxtf*Rf)
        dwr=(1/Jw)*(self.drive_torque(t,wr)/4 + rolling_res_r + br_t_r - Fxtr*Rr)


        # The normal forces at four tires are determined using d'alemberts principle
        Z1 = (m*g*b)/(2*(a+b))+(muf*g)/2
        Z4 = ((m*h+muf*huf+mur*hur)*(Vx_dot-psi_dot*Vy))/(2*(a+b))
        self.Fzgf = Z1-Z4
        Z5 = (m*g*a)/(2*(a+b))+(mur*g)/2
        Z8 = ((m*h+muf*huf+mur*hur)*(Vx_dot-psi_dot*Vy))/(2*(a+b))
        self.Fzgr = Z5+Z8

        # These vertical forces cannot be lesser than 0 as that means that the wheel is off the ground
        if(self.Fzgf < 0):
            self.Fzgf = 0
        if(self.Fzgr < 0):
            self.Fzgr = 0

        # Tire deflection calculated using the tire stiffness and the normal force on the tires
        self.xtf = self.Fzgf/ktf
        self.xtr = self.Fzgr/ktr


        if(self.debug):
            self.t_arr.append(t)
            self.ff.append(self.Fzgf)
            self.fr.append(self.Fzgr)
            self.dt.append(self.drive_torque(t, wr)/4)
            self.fdt.append(Fxtf*Rf)
            self.rdt.append(rolling_res_f)
            self.s_arr.append(sr)
            self.xtf_ar.append(self.xtf)

        return np.stack([dx, dy, Vx_dot, Vy_dot, dpsi, dpsi_dot, dwf, dwr])    

    # function to obtain the 2nd level variables - needed for the half-implicit solver
    def lvl_2_fiala(self,t,state):
        g = self.g
        a, b, h, Cf, Cr, Cxf, Cxr, ktf, ktr, m, muf, mur, Jz, r0, Jw, rr = list(self.params.values())
        x, y, Vx, Vy, psi, psi_dot, wf, wr = state

        mt=m+2*muf+2*mur # vehicle total mass

        # instataneous tire radius
        Rf=r0-self.xtf
        Rr=r0-self.xtr
        huf = Rf
        hur = Rr

        # the steering angle inputted to the wheels
        st_a = self.steering(t) * self.max_steer

        # longitudinal slip
        sf = (Rf*wf-(Vx*np.cos(st_a)+(Vy+a*psi_dot)*np.sin(st_a))) / \
            np.abs(Vx*np.cos(st_a) +
                   (Vy+a*psi_dot)*np.sin(st_a))
        sr = (Rr*wr-Vx)/np.abs(Vx)

        # lateral slip
        delta_f = np.arctan2((Vy+a*psi_dot),Vx)-(st_a)
        delta_r = np.arctan2((Vy-b*psi_dot),Vx)

        # comprehensive slip ratio
        ss_f = min(np.sqrt(sf**2 + np.tan(delta_f)**2), 1.)
        ss_r = min(np.sqrt(sr**2 + np.tan(delta_r)**2), 1.)

        # Coefficient of friction based on the comprehensive slip
        u_f = self.umax - (self.umax - self.umin)*ss_f
        u_r = self.umax - (self.umax - self.umin)*ss_r

        # critical longitudinal slip
        s_crit_f = np.abs((u_f * self.Fzgf) / (2 * Cxf))
        s_crit_r = np.abs((u_r * self.Fzgr) / (2 * Cxr))

        # critical lateral slip
        al_crit_f = np.arctan((3*u_f * np.abs(self.Fzgf))/Cf)
        al_crit_r = np.arctan((3*u_r * np.abs(self.Fzgr))/Cr)

        # longitudinal forces based on whether the tire is in the elastic regime or the sliding regime
        if(np.abs(sf) < s_crit_f):
            Fxtf = Cxf*sf
        else:
            Fxtf_1 = u_f * np.abs(self.Fzgf)
            Fxtf_2 = np.abs((u_f * self.Fzgf)**2 / (4 * sf * Cxf))
            Fxtf = np.sign(sf)*(Fxtf_1 - Fxtf_2)
        
        if(np.abs(sr) < s_crit_r):
            Fxtr = Cxr*sr
        else:
            Fxtr_1 = u_r * np.abs(self.Fzgr)
            Fxtr_2 = np.abs((u_r * self.Fzgr)**2 / (4 * sr * Cxr))
            Fxtr = np.sign(sr)*(Fxtr_1 - Fxtr_2)

        # lateral forces based on whether the tire is in elastic regime or the sliding regime 
        if(np.abs(delta_f) <= al_crit_f):
            h_ = 1 - ((Cf * np.abs(np.tan(delta_f))) / (3 * u_f * np.abs(self.Fzgf)))
            Fytf = -u_f * np.abs(self.Fzgf) * (1-h_**3)*np.sign(delta_f)
        else:
            Fytf = -u_f * np.abs(self.Fzgf) * np.sign(delta_f)

        if(np.abs(delta_r) <= al_crit_r):
            h_ = 1 - ((Cr * np.abs(np.tan(delta_r))) / (3 * u_r * np.abs(self.Fzgr)))
            Fytr = -u_r * np.abs(self.Fzgr) * (1-h_**3)*np.sign(delta_r)
        else:
            Fytr = -u_r * np.abs(self.Fzgr) * np.sign(delta_r)

        # ODE's
        Vy_dot = -Vx*psi_dot + (Fxtf*np.sin(st_a) + Fytf*np.cos(st_a) + Fytr)/mt
        Vx_dot = Vy*psi_dot + (Fxtf*np.cos(st_a) - Fytf*np.sin(st_a) + Fxtr)/mt
        dpsi_dot = (a*(Fytf*np.cos(st_a) + Fxtf*np.sin(st_a)) - b*Fytr)/Jz

        # rolling resistance
        rolling_res_f = -rr * np.abs(self.Fzgf) * np.sign(wf)
        rolling_res_r = -rr * np.abs(self.Fzgr) * np.sign(wr)

        # Brake Torques of each wheel
        br_t_f = - np.sign(wf) * self.brake_torque(t)
        br_t_r = - np.sign(wr) * self.brake_torque(t)


        # Wheel rotational model - Dividing by 4 still - this is slighly confusing as there are only 2 wheels
        # However the point is to apply a torque that is the same as the chrono vehicle which has 4 wheels
        dwf=(1/Jw)*(self.drive_torque(t,wf)/4 + rolling_res_f + br_t_f - Fxtf*Rf)
        dwr=(1/Jw)*(self.drive_torque(t,wr)/4 + rolling_res_r + br_t_r - Fxtr*Rr)


        # The normal forces at four tires are determined using d'alemberts principle
        Z1 = (m*g*b)/(2*(a+b))+(muf*g)/2
        Z4 = ((m*h+muf*huf+mur*hur)*(Vx_dot-psi_dot*Vy))/(2*(a+b))
        self.Fzgf = Z1-Z4
        Z5 = (m*g*a)/(2*(a+b))+(mur*g)/2
        Z8 = ((m*h+muf*huf+mur*hur)*(Vx_dot-psi_dot*Vy))/(2*(a+b))
        self.Fzgr = Z5+Z8

        # These vertical forces cannot be lesser than 0 as that means that the wheel is off the ground
        if(self.Fzgf < 0):
            self.Fzgf = 0
        if(self.Fzgr < 0):
            self.Fzgr = 0

        # Tire deflection calculated using the tire stiffness and the normal force on the tires
        self.xtf = self.Fzgf/ktf
        self.xtr = self.Fzgr/ktr


        if(self.debug):
            self.dt.append(self.drive_torque(t, wr)/4)
            self.fdt.append(Fxtf*Rf)
            self.rdt.append(rolling_res_f)
            self.s_arr.append(sr)

        return np.stack([Vx_dot, Vy_dot, dpsi_dot, dwf, dwr])

    # half implicit solver
    def solve_half_impl(self,t_span,t_eval,tbar):
        a, b, h, Cf, Cr, Cxf, Cxr, ktf, ktr, m, muf, mur, Jz, r0, Jw, rr = list(self.params.values())
        g = 9.8
        mt=m+2*muf+2*mur # vehicle total mass

        time_steps = np.arange(t_span[0]+tbar,t_span[1]+0.0000001,tbar)
        self.level_1_vars = np.array([self.states['Vx'],self.states['Vy'],self.states['psi_dot'],self.states['wf'],self.states['wr']])
        self.level_0_vars = np.array([self.states['x'],self.states['y'],self.states['psi']])

        # Save n every 
        n_ = round((t_eval[1]-t_eval[0])/tbar)
        
        # For saving the results at t_eval
        count = 1

        outs = np.empty((len(t_eval),len(self.states)))
        outs[0] = (np.array(list(self.states.values())))

        for t in time_steps:
            # evalaute level 2 variables
            level_2_vars = self.lvl_2_fiala(t,list(self.states.values()))

            # use new level 2 variables to update level 1 variables
            self.level_1_vars = self.level_1_vars + tbar * np.array(level_2_vars)

            # use new level 1 varaibles to update level 0 variables - first update vehicle global coordiantes
            self.level_0_vars[0] = self.level_0_vars[0] + tbar * (self.level_1_vars[0]*np.cos(self.level_0_vars[2]) - 
            self.level_1_vars[1]*np.sin(self.level_0_vars[2]))

            self.level_0_vars[1] = self.level_0_vars[1] + tbar * (self.level_1_vars[0]*np.sin(self.level_0_vars[2]) + 
            self.level_1_vars[1]*np.cos(self.level_0_vars[2]))

            # update the yaw angle
            self.level_0_vars[2] = self.level_0_vars[2] + tbar * (self.level_1_vars[2])

            # update the satate of the vehicle
            x,y,psi = self.level_0_vars
            Vx,Vy,psi_dot,wf,wr = self.level_1_vars
            Vx_dot,Vy_dot,dpsi_dot,dwf,dwr = level_2_vars
            st = [x,y,Vx,Vy,psi,psi_dot,wf,wr]
            self.states.update(zip(self.states,st))

            # use the updated state the evalaute the normal forces, tire deflection 
            # and inst. tire radius for the next time step
            
            # instataneous tire radius
            Rf=r0-self.xtf
            Rr=r0-self.xtr
            huf = Rf
            hur = Rr

            # The normal forces at four tires are determined using d'alemberts principle
            Z1 = (m*g*b)/(2*(a+b))+(muf*g)/2
            Z4 = ((m*h+muf*huf+mur*hur)*(Vx_dot-psi_dot*Vy))/(2*(a+b))
            self.Fzgf = Z1-Z4
            Z5 = (m*g*a)/(2*(a+b))+(mur*g)/2
            Z8 = ((m*h+muf*huf+mur*hur)*(Vx_dot-psi_dot*Vy))/(2*(a+b))
            self.Fzgr = Z5+Z8

            # These vertical forces cannot be lesser than 0 as that means that the wheel is off the ground
            if(self.Fzgf < 0):
                self.Fzgf = 0
            if(self.Fzgr < 0):
                self.Fzgr = 0

            # Tire deflection calculated using the tire stiffness and the normal force on the tires
            self.xtf = self.Fzgf/ktf
            self.xtr = self.Fzgr/ktr

            #DEBUG
            if(self.debug):
                self.t_arr.append(t)
                self.ff.append(self.Fzgf)
                self.fr.append(self.Fzgr)
                self.xtf_ar.append(self.xtf)

            if(count%n_ == 0):
                outs[round(count/n_)] = (np.array(list(self.states.values())))
            count = count + 1
        return outs 


    # A wrapper function for maybe a few packages - starting with solve_ivp, odeint
    def solve(self,package = 'solve_ivp',tire_model = 0,t_eval = None,tbar = 1e-2,**kwargs):
        try:
            self.steering
        except:
            raise Exception("Please provide steering controls for the vehicle with 'set_steering' method")
        
        try:
            self.throttle
        except:
            raise Exception("Please provide throttle controls for the vehicle with 'set_throttle' method")

        if t_eval is None:
            raise Exception("Please provide times steps at which you want the solution to be evaluated")

        try:
            self.brake
        except:
            def zero_brake(t):
                return 0*t
            self.brake = zero_brake

        # Need the start time for the smoothing function
        self.start_time = t_eval[0]

        if(tire_model == 1):
            if(package == 'half_implicit'):
                return self.solve_half_impl(t_span = [t_eval[0],t_eval[-1]],t_eval = t_eval,tbar = 1e-2)
            else:
                return solve_ivp(self.model_fiala,t_span=[t_eval[0],t_eval[-1]],y0 = list(self.states.values()),
                vectorized = False,
                t_eval = t_eval,**kwargs)
        else:
            return solve_ivp(self.model_linear,t_span=[t_eval[0],t_eval[-1]],y0 = list(self.states.values()),vectorized = False,
            t_eval = t_eval,**kwargs)


##################################################################################
### 8 DOF MODEL ###
##################################################################################

class vd_8dof:
    def __init__(self, parameters=None, states=None):
        self.g = 9.8

        # Fiala Tire friction parameters
        self.umin = 0.5568
        self.umax = 0.9835

        # Tire relaxation lengths, not used currently
        self.xrel = 1.0
        self.yrel = 1.0

        # The smoothing time - Commented out currently
        self.trans_time = 0.2

        # Lists used for storing values for debugging
        self.t_arr = []
        self.flf = []
        self.flr = []
        self.frf = []
        self.frr = []
        self.dt = []
        self.fdt = []
        self.rdt = []
        self.s_arr = []
        self.xtrf_ar = []

        self.debug = 0
        if parameters is not None:
            if isinstance(parameters, list):
                raise Exception(
                    "Please provide a dictionary for the parameters")
            try:
                # distance of c.g. from front axle (m)
                self.a = parameters['a']
            except:
                self.a = 1.14
                warnings.warn(
                    f"Set 'a' to default value {self.a}", UserWarning)
            try:
                # distance of c.g. from rear axle  (m)
                self.b = parameters['b']
            except:
                self.b = 1.4
                warnings.warn(
                    f"Set 'b' to default value {self.b}", UserWarning)
            try:
                # front axle cornering stiffness (N/rad)
                self.Cf = parameters['Cf']
            except:
                self.Cf = -44000
                warnings.warn(
                    f"Set 'Cf' to default value {self.Cf}", UserWarning)
            try:
                # rear axle cornering stiffness (N/rad)
                self.Cr = parameters['Cr']
            except:
                self.Cr = -47000
                warnings.warn(
                    f"Set 'Cr' to default value {self.Cr}", UserWarning)
            try:
                # front axle longitudinal stiffness (N)
                self.Cxf = parameters['Cxf']
            except:
                self.Cxf = 5000
                warnings.warn(
                    f"Set 'Cxf' to default value {self.Cxf}", UserWarning)
            try:
                # rear axle longitudinal stiffness (N)
                self.Cxr = parameters['Cxr']
            except:
                self.Cxr = 5000
                warnings.warn(
                    f"Set 'Cxr' to default value {self.Cxr}", UserWarning)
            try:
                self.m = parameters['m']  # the mass of the vehicle (kg)
            except:
                self.m = 1400
                warnings.warn(
                    f"Set 'm' to default value {self.m}", UserWarning)
            try:
                self.Jz = parameters['Jz']  # yaw moment of inertia (kg.m^2)
            except:
                self.Jz = 2420
                warnings.warn(
                    f"Set 'Jz' to default value {self.Jz}", UserWarning)
            try:
                self.r0 = parameters['r0']  # wheel radius
            except:
                self.r0 = 0.285
                warnings.warn(
                    f"Set 'r0' to default value {self.r0}", UserWarning)
            try:
                self.Jw = parameters['Jw']  # wheel roll inertia
            except:
                self.Jw = 1
                warnings.warn(
                    f"Set 'Jw' to default value {self.Jw}", UserWarning)
            try:
                self.Jx = parameters['Jx']  # wheel roll inertia
            except:
                self.Jx = 900
                warnings.warn(
                    f"Set 'Jx' to default value {self.Jx}", UserWarning)
            try:
                self.Jy = parameters['Jy']  # wheel roll inertia
            except:
                self.Jy = 2000
                warnings.warn(
                    f"Set 'Jy' to default value {self.Jy}", UserWarning)
            try:
                self.Jxz = parameters['Jxz']  # wheel roll inertia
            except:
                self.Jxz = 90
                warnings.warn(
                    f"Set 'Jxz' to default value {self.Jxz}", UserWarning)
            try:
                self.h = parameters['h']  # wheel roll inertia
            except:
                self.h = 0.75
                warnings.warn(
                    f"Set 'h' to default value {self.h}", UserWarning)
            try:
                self.cf = parameters['cf']  # wheel roll inertia
            except:
                self.cf = 1.5
                warnings.warn(
                    f"Set 'cf' to default value {self.cf}", UserWarning)
            try:
                self.cr = parameters['cr']  # wheel roll inertia
            except:
                self.cr = 1.5
                warnings.warn(
                    f"Set 'cr' to default value {self.cr}", UserWarning)
            try:
                self.muf = parameters['muf']  # wheel roll inertia
            except:
                self.muf = 80
                warnings.warn(
                    f"Set 'muf' to default value {self.muf}", UserWarning)
            try:
                self.mur = parameters['mur']  # wheel roll inertia
            except:
                self.mur = 80
                warnings.warn(
                    f"Set 'mur' to default value {self.mur}", UserWarning)
            try:
                self.ktf = parameters['ktf']  # wheel roll inertia
            except:
                self.ktf = 200000
                warnings.warn(
                    f"Set 'ktf' to default value {self.ktf}", UserWarning)
            try:
                self.ktr = parameters['ktr']  # wheel roll inertia
            except:
                self.ktr = 200000
                warnings.warn(
                    f"Set 'ktr' to default value {self.ktr}", UserWarning)
            try:
                self.hrcf = parameters['hrcf']  # wheel roll inertia
            except:
                self.hrcf = 0.65
                warnings.warn(
                    f"Set 'hrcf' to default value {self.hrcf}", UserWarning)
            try:
                self.hrcr = parameters['hrcr']  # wheel roll inertia
            except:
                self.hrcr = 0.6
                warnings.warn(
                    f"Set 'hrcr' to default value {self.hrcr}", UserWarning)
            try:
                self.brof = parameters['brof']  # wheel roll inertia
            except:
                self.brof = 3000
                warnings.warn(
                    f"Set 'brof' to default value {self.brof}", UserWarning)
            try:
                self.krof = parameters['krof']  # wheel roll inertia
            except:
                self.krof = 29000
                warnings.warn(
                    f"Set 'krof' to default value {self.krof}", UserWarning)
            try:
                self.kror = parameters['kror']  # wheel roll inertia
            except:
                self.kror = 29000
                warnings.warn(
                    f"Set 'kror' to default value {self.kror}", UserWarning)
            try:
                self.bror = parameters['bror']  # wheel roll inertia
            except:
                self.bror = 3000
                warnings.warn(
                    f"Set 'bror' to default value {self.bror}", UserWarning)
            try:
                self.rr = parameters['rr']  # wheel roll inertia
            except:
                self.rr = 0.0125
                warnings.warn(
                    f"Set 'rr' to default value {self.rr}", UserWarning)

            # A dictionary of parameters - For use in the model
            self.params = {'a': self.a, 'b': self.b, 'Cf': self.Cf, 'Cr': self.Cr, 'Cxf': self.Cxf, 'Cxr': self.Cxr, 'm': self.m, 'Jz': self.Jz, 'r0': self.r0, 'Jw': self.Jw, 'Jx': self.Jx, 'Jy': self.Jy, 'Jxz': self.Jxz, 'h': self.h, 'cf': self.cf, 'cr': self.cr, 'muf': self.muf, 'mur': self.mur, 'ktf': self.ktf, 'ktr': self.ktr, 'hrcf': self.hrcf, 'hrcr': self.hrcr,
                           'krof': self.krof, 'kror': self.kror, 'brof': self.brof, 'bror': self.bror, 'rr': self.rr}
        else:
            self.a = 1.14  # distance of c.g. from front axle (m)
            self.b = 1.4  # distance of c.g. from rear axle  (m)
            self.Cf = 44000  # front axle cornering stiffness (N/rad)
            self.Cr = 47000  # rear axle cornering stiffness (N/rad)
            self.Cxf = 5000  # front axle longitudinal stiffness (N)
            self.Cxr = 5000  # rear axle longitudinal stiffness (N)
            self.m = 1400  # the mass of the vehicle (kg)
            self.Jz = 2420  # yaw moment of inertia (kg.m^2)
            self.r0 = 0.285  # wheel radius
            self.Jw = 2  # wheel roll inertia
            self.Jx = 900  # Sprung mass roll inertia (kg.m^2)
            self.Jy = 2000  # Pitch inertia - NOT USED
            self.Jxz = 90
            self.h = 0.75  # Height of CG
            self.cf = 1.5  # Front track width
            self.cr = 1.5  # Rear track width
            self.muf = 80  # Front unsprung mass
            self.mur = 80  # Rear unsprung mass
            self.ktf = 200000  # Front tire stiffness
            self.ktr = 200000  # Rear tire stiffness
            self.hrcf = 0.65  # Front roll centre height below CG
            self.hrcr = 0.6  # Rear roll centre height below CG
            self.krof = 29000  # Front roll stiffness
            self.kror = 29000  # Rear roll stiffness
            self.brof = 3000  # Front roll damping
            self.bror = 3000  # Rear roll damping
            self.rr = 0.0125  # Roll resistance coefficient

            # A dictionary of parameters
            self.params = {'a': self.a, 'b': self.b, 'Cf': self.Cf, 'Cr': self.Cr, 'Cxf': self.Cxf, 'Cxr': self.Cxr, 'm': self.m, 'Jz': self.Jz, 'r0': self.r0, 'Jw': self.Jw, 'Jx': self.Jx, 'Jy': self.Jy, 'Jxz': self.Jxz, 'h': self.h, 'cf': self.cf, 'cr': self.cr, 'muf': self.muf, 'mur': self.mur, 'ktf': self.ktf, 'ktr': self.ktr, 'hrcf': self.hrcf, 'hrcr': self.hrcr,
                           'krof': self.krof, 'kror': self.kror, 'brof': self.brof, 'bror': self.bror, 'rr': self.rr}

            warnings.warn("Set parameters to default values" +
                          '\n' + f"{self.params}", UserWarning)

        if states is not None:
            if isinstance(states, list):
                raise Exception("Please provide a dictionary for the states")

            # State of the vehicle
            try:
                self.x=states['x'] # lateral velocity 
            except:
                self.x = 0
                warnings.warn(
                    f"Set 'x' to default value {self.x}",UserWarning)
            try:
                self.y=states['y'] # lateral velocity 
            except:
                self.y = 0
                warnings.warn(
                    f"Set 'y' to default value {self.y}",UserWarning)
            try:
                self.u = states['u']  # lateral velocity
            except:
                self.u = 50/3.6
                warnings.warn(
                    f"Set 'u' to default value {self.u}", UserWarning)
            try:
                self.v = states['v']
            except:
                self.v = 0.
                warnings.warn(
                    f"Set 'v' to default value {self.v}", UserWarning)
            try:
                self.psi = states['psi']  # yaw rate
            except:
                self.psi = 0.
                warnings.warn(
                    f"Set 'psi' to default value {self.psi}", UserWarning)
            try:
                self.wlf = states['wlf']  # Front wheel angular velocity
            except:
                self.wlf = self.u/self.r0
                warnings.warn(
                    f"Set 'wlf' to default value {self.wlf}", UserWarning)
            try:
                self.wlr = states['wlr']  # Rear wheel angular velocity
            except:
                self.wlr = self.u/self.r0
                warnings.warn(
                    f"Set 'wlr' to default value {self.wlr}", UserWarning)
            try:
                self.wrf = states['wrf']  # Rear wheel angular velocity
            except:
                self.wrf = self.u/self.r0
                warnings.warn(
                    f"Set 'wrf' to default value {self.wrf}", UserWarning)
            try:
                self.wrr = states['wrr']  # Rear wheel angular velocity
            except:
                self.wrr = self.u/self.r0
                warnings.warn(
                    f"Set 'wrr' to default value {self.wrr}", UserWarning)
            try:
                self.phi = states['phi']  # Rear wheel angular velocity
            except:
                self.phi = 0.
                warnings.warn(
                    f"Set 'phi' to default value {self.phi}", UserWarning)
            try:
                self.wx = states['wx']  # Rear wheel angular velocity
            except:
                self.wx = 0.
                warnings.warn(
                    f"Set 'wx' to default value {self.wx}", UserWarning)
            try:
                self.wz = states['wz']  # Rear wheel angular velocity
            except:
                self.wz = 0.
                warnings.warn(
                    f"Set 'wz' to default value {self.wz}", UserWarning)
                    
            # A dictionary of states
            self.states = {'x':self.x,'y':self.y,'u': self.u, 'v': self.v, 'psi': self.psi, 'phi': self.phi, 'wx': self.wx, 'wz': self.wz, 'wlf': self.wlf, 'wlr': self.wlr, 'wrf': self.wrf,
                           'wrr': self.wrr}
        else:
            self.x = 0
            self.y = 0
            self.u = 50/3.6  # the longitudinal velocity
            self.v = 0.     # the lateral velocity
            self.phi = 0.   # roll angle
            self.psi = 0.  # Yaw angle
            self.wx = 0.  # Roll rate
            self.wz = 0.   # yaw angular velocity
            self.wlf = self.u/self.r0  # angular velocity of left front wheel rotation
            self.wrf = self.u/self.r0  # angular velocity of right front wheel rotation
            self.wlr = self.u/self.r0  # angular velocity of left rear wheel rotation
            self.wrr = self.u/self.r0  # angular velocity of right rear wheel rotation


            # A dictionary of states
            self.states = {'x': self.x, 'y': self.y, 'u': self.u, 'v': self.v, 'psi': self.psi, 'phi': self.phi, 'wx': self.wx, 'wz': self.wz, 'wlf': self.wlf, 'wlr': self.wlr, 'wrf': self.wrf,
                           'wrr': self.wrr}
            warnings.warn("Set States to default values" +
                          '\n' + f"{self.states}", UserWarning)


        # Vertical forces initially
        self.Fzgrf = ((self.m*self.g*self.b) /
                        (2*(self.a+self.b))+self.muf*self.g)
        self.Fzglf = ((self.m*self.g*self.b) /
                        (2*(self.a+self.b))+self.muf*self.g)
        self.Fzglr = ((self.m*self.g*self.b) /
                        (2*(self.a+self.b))+self.mur*self.g)
        self.Fzgrr = ((self.m*self.g*self.b) /
                        (2*(self.a+self.b))+self.mur*self.g)

        # the initial tire compression xt
        self.xtrf = ((self.m*self.g*self.b)/(2*(self.a+self.b)
                                                )+self.muf*self.g)/self.ktf  # Right front
        self.xtlf = ((self.m*self.g*self.b)/(2*(self.a+self.b)
                                                )+self.muf*self.g)/self.ktf  # left front
        self.xtlr = ((self.m*self.g*self.a) /
                        (2*(self.a+self.b))+self.mur*self.g)/self.ktr
        self.xtrr = ((self.m*self.g*self.a) /
                        (2*(self.a+self.b))+self.mur*self.g)/self.ktr

    # Simple print to show the parameters and states
    def __str__(self):
        return str(self.__class__) + '\n' + "Vehicle Parameters are" + '\n' + f"{self.params}" + '\n' + "Vehicle state is" + '\n' + f"{self.states}"

    # Sets the steering of the vehicle
    def set_steering(self, steering, max_steering=0.6525249):
        """
        Takes as input a normalised steering function from -1 to 1 and a max steering value in radians. +ve is a left turn
        """
        # Pitman arm
        # self.max_steer = 0.7328647
        self.steering = steering
        self.max_steer = max_steering

    # Sets the throttle controls of the vehicle
    def set_throttle(self, throttle, gr=0.3*0.2, mt=1000., ms=2000.):
        """
        Takes as input a normalised throttle fron -1 to 1, a gear ratio, a max torque(Nm) and a max speed (RPS?)
	    The throttle is then applied in the drive_torque function along with the other paramters
        """
        # The 0.2 is from the conical gears of the chrono vehicle
        self.throttle = throttle
        self.gear_ratio = gr
        self.max_torque = mt
        self.max_speed = ms
	
	#Sets the brake controls of the vehicle
    def set_braking(self,brake,mbt = 4000):
        self.brake = brake
        self.max_brake_torque = mbt

    # Update the states externally after initilisation - Through dict or kwargs
    def update_states(self,state_dict = None, **kwargs):
        if(state_dict == None):
            for key, value in kwargs.items():
                if(key not in self.states):
                    raise Exception(f"{key} is not vehicle state")

                # Set the state to the respective class attribute
                setattr(self, key, value)
                # Update the states dict attribute as well
                self.states[key] = value
        else:
            for key,value in state_dict.items():
                self.states[key] = value


    # Update the parameters externally after initilisation
    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if(key not in self.params):
                raise Exception(f"{key} is not vehicle parameter")

            # Set the parameter to the respective class attribute
            setattr(self, key, value)
            # Update the params dict of the vehicle as well
            self.params[key] = value

    # Returns a list of the parameters
    def get_params(self):
        return self.params

    # Returns a list of the current states
    def get_states(self):
        return self.states

    # Just a method to reset the state to default values or values specified when vehicle object was created
    # Needed for running multiple simulations
    def reset_state(self,init_state):
        for key,value in init_state.items():
            self.states[key] = value

        #Vertical forces initially
        self.Fzgrf = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)
        self.Fzglf = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)
        self.Fzglr = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.mur*self.g)
        self.Fzgrr = ((self.m*self.g*self.b)/(2*(self.a+self.b))+self.mur*self.g)

        #Vertical forces as an array
        self.Fzg = np.empty((4,))
        self.Fzg[0] = self.Fzglf
        self.Fzg[1] = self.Fzgrf
        self.Fzg[2] = self.Fzglr
        self.Fzg[3] = self.Fzgrr

        ## the initial tire compression xt
        self.xtrf=((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)/self.ktf  # Right front
        self.xtlf=((self.m*self.g*self.b)/(2*(self.a+self.b))+self.muf*self.g)/self.ktf  #left front
        self.xtlr=((self.m*self.g*self.a)/(2*(self.a+self.b))+self.mur*self.g)/self.ktr 
        self.xtrr=((self.m*self.g*self.a)/(2*(self.a+self.b))+self.mur*self.g)/self.ktr 

        #tire compression as an array
        self.xt = np.empty((4,))
        self.xt[0] = self.xtlf
        self.xt[1] = self.xtrf
        self.xt[2] = self.xtlr
        self.xt[3] = self.xtrr

        if(self.debug):
            self.t_arr = []
            self.flf = []
            self.flr = []
            self.frf = []
            self.frr = []
            self.dt = []
            self.fdt= []
            self.rdt= []
            self.s_arr= []
            self.xtrf_ar= []

    # Evaluate the drive torque based on the wheel speed - Similar to 4 motors on each of the wheels
    def drive_torque(self, t, w):
        motor_speed = w / self.gear_ratio
        motor_torque = self.max_torque - \
            (motor_speed * (self.max_torque / self.max_speed))
        motor_torque = motor_torque * self.throttle(t)
        return motor_torque / self.gear_ratio
    
    # Evaluate the brake torque at a paticular time step
    def brake_torque(self,t):
        return self.brake(t)*self.max_brake_torque


    # A utility function to stabilsie the initial transients
    def smooth_step(self, t, f1, t1, f2, t2):
        if(t < t1):
            return f1
        elif(t >= t2):
            return f2
        else:
            return f1 + ((f2 - f1)*((t-t1)/(t2-t1))**2 * (3 - 2*((t - t1)/(t2-t1))))

    # Vehicle model for a very simple linear tire
    def model_linear(self, t, state):
        a, b, Cf, Cr, Cxf, Cxr, m, Jz, r0, Jw, Jx, Jy, Jxz, h, cf, cr, muf, mur, ktf, ktr, hrcf, hrcr, krof, kror, brof, bror, rr = list(
            self.params.values())
        g = 9.8
        x, y, u, v, psi, phi, wx, wz, wlf, wlr, wrf, wrr = state

        # Some calculated parameters
        # the vertical distance from the sprung mass C.M. to the vehicle roll center.
        hrc = (hrcf*b+hrcr*a)/(a+b)
        mt = m+2*muf+2*mur  # vehicle total mass

        # Instantaneous tire radius - Updated using the tire deflection which is calculated using the normal force on tires
        Rrf = r0-self.xtrf
        Rlf = r0-self.xtlf
        Rlr = r0-self.xtlr
        Rrr = r0-self.xtrr

        # position of front and rear unsprung mass
        huf = Rrf
        hur = Rrr

        # the longitudinal and lateral velocities at the tire contact patch in tire coordinate frame
        ugrf = u+(wz*cf)/2
        vgrf = v+wz*a
        uglf = u-(wz*cf)/2
        vglf = v+wz*a
        uglr = u-(wz*cr)/2
        vglr = v-wz*b
        ugrr = u+(wz*cr)/2
        vgrr = v-wz*b

        # tire slip angle of each wheel - if else for 0 velocity condition
        if((abs(ugrf) > 1e-4) or (abs(vgrf) > 1e-4)):
            delta_rf = np.arctan(vgrf/ugrf)-(self.steering(t) * self.max_steer)
            s_rf = max(min((Rrf*wrf-(ugrf*np.cos(self.steering(t) * self.max_steer)+vgrf*np.sin(self.steering(t) * self.max_steer)))/np.abs(ugrf*np.cos(self.steering(t) * self.max_steer)
                                                                                                                                            + vgrf*np.sin(self.steering(t) * self.max_steer)), 1), -1)
        else:
            delta_rf = self.steering(t) * self.max_steer
            s_rf = 0
        if((abs(uglf) > 1e-4) or (abs(vglf) > 1e-4)):
            delta_lf = np.arctan(vglf/uglf)-(self.steering(t) * self.max_steer)
            s_lf = max(min((Rlf*wlf-(uglf*np.cos(self.steering(t) * self.max_steer)+vglf*np.sin(self.steering(t) * self.max_steer)))/np.abs(uglf*np.cos(self.steering(t) * self.max_steer)
                                                                                                                                            + vglf*np.sin(self.steering(t) * self.max_steer)), 1), -1)
        else:
            delta_lf = self.steering(t) * self.max_steer
            s_lf = 0
        if((abs(uglr) > 1e-4) or (abs(vglr) > 1e-4)):
            delta_lr = np.arctan(vglr/uglr)
            s_lr = max(min((Rlr*wlr-uglr)/np.abs(uglr), 1), -1)
        else:
            delta_lr = 0
            s_lr = 0
        if((abs(ugrr) > 1e-4) or (abs(vgrr) > 1e-4)):
            delta_rr = np.arctan(vgrr/ugrr)
            s_rr = max(min((Rrr*wrr-ugrr)/np.abs(ugrr), 1), -1)
        else:
            delta_rr = 0
            s_rr = 0

        # Smoothing function to smooth out initial transients - does not really work well
        smth = self.smooth_step(t, 0, self.start_time,
                                1, self.start_time + self.trans_time)

        # linear tire lateral force
        Fytrf = -Cf*delta_rf*smth
        Fytlf = -Cf*delta_lf*smth
        Fytlr = -Cr*delta_lr*smth
        Fytrr = -Cr*delta_rr*smth

        # linear tire longitudinal force
        Fxtrf = Cxf*s_rf*smth
        Fxtlf = Cxf*s_lf*smth
        Fxtlr = Cxr*s_lr*smth
        Fxtrr = Cxr*s_rr*smth

        # the forces Fxgij obtained by resolving the longitudinal and cornering forces at the tire contact patch
        Fxglf = Fxtlf*np.cos(self.steering(t) * self.max_steer) - \
            Fytlf*np.sin(self.steering(t) * self.max_steer)
        Fxgrf = Fxtrf*np.cos(self.steering(t) * self.max_steer) - \
            Fytrf*np.sin(self.steering(t) * self.max_steer)
        Fxglr = Fxtlr
        Fxgrr = Fxtrr
        Fyglf = Fxtlf*np.sin(self.steering(t) * self.max_steer) + \
            Fytlf*np.cos(self.steering(t) * self.max_steer)
        Fygrf = Fxtrf*np.sin(self.steering(t) * self.max_steer) + \
            Fytrf*np.cos(self.steering(t) * self.max_steer)
        Fyglr = Fytlr
        Fygrr = Fytrr

        # Some other constants used in the differential equations
        dpsi = wz
        dphi = wx
        E1 = -mt*wz*u+(Fyglf+Fygrf+Fyglr+Fygrr)
        E2 = (Fyglf+Fygrf)*a-(Fyglr+Fygrr)*b+(Fxgrf-Fxglf) * \
            cf/2+(Fxgrr-Fxglr)*cr/2+(mur*b-muf*a)*wz*u
        E3 = m*g*hrc*phi-(krof+kror)*phi-(brof+bror)*dphi+hrc*m*wz*u
        A1 = mur*b-muf*a
        A2 = Jx+m*hrc**2
        A3 = hrc*m

        # Chassis Model
        u_dot = wz*v+(1/mt)*((Fxglf+Fxgrf+Fxglr+Fxgrr) +
                             (muf*a-mur*b)*(wz)**2-2*hrc*m*wz*wx)
        v_dot = (E1*Jxz**2-A1*A2*E2+A1*E3*Jxz+A3*E2*Jxz-A2*E1*Jz-A3 *
                 E3*Jz)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
        wx_dot = (A1**2*E3-A1*A3*E2+A1*E1*Jxz-A3*E1*Jz+E2*Jxz*mt-E3 *
                  Jz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
        wz_dot = (A3**2*E2-A1*A2*E1-A1*A3*E3+A3*E1*Jxz-A2*E2*mt+E3 *
                  Jxz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
        dx = u*np.cos(psi) - v*np.sin(psi)
        dy = u*np.sin(psi) + v*np.cos(psi)

        # The normal forces at four tires are determined using d'alemberts principle
        Z1 = (m*g*b)/(2*(a+b))+(muf*g)/2
        Z2 = ((muf*huf)/cf+m*b*(h-hrcf)/(cf*(a+b)))*(v_dot+wz*u)
        Z3 = (krof*phi+brof*dphi)/cf
        Z4 = ((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b))
        Fzglf = Z1-Z2-Z3-Z4
        Fzgrf = Z1+Z2+Z3-Z4
        Z5 = (m*g*a)/(2*(a+b))+(mur*g)/2
        Z6 = ((mur*hur)/cr+m*a*(h-hrcr)/(cr*(a+b)))*(v_dot+wz*u)
        Z7 = (kror*phi+bror*dphi)/cr
        Z8 = ((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b))
        Fzglr = Z5-Z6-Z7+Z8
        Fzgrr = Z5+Z6+Z7+Z8

        # These vertical forces cannot be lesser than 0 as that means that the wheel is off the ground
        if(Fzgrf < 0):
            Fzgrf = 0
        if(Fzglf < 0):
            Fzglf = 0
        if(Fzglr < 0):
            Fzglr = 0
        if(Fzgrr < 0):
            Fzgrr = 0

        # Tire deflection calculated using the tire stiffness and the normal force on the tires
        self.xtlf = Fzglf/ktf
        self.xtrf = Fzgrf/ktf
        self.xtlr = Fzglr/ktr
        self.xtrr = Fzgrr/ktr

        # Rolling resistance on the tire
        rolling_res_lf = -rr * np.abs(Fzglf) * np.sign(wlf)
        rolling_res_rf = -rr * np.abs(Fzgrf) * np.sign(wrf)
        rolling_res_lr = -rr * np.abs(Fzglr) * np.sign(wlr)
        rolling_res_rr = -rr * np.abs(Fzgrr) * np.sign(wrr)

        # Brake Torques of each wheel
        br_t_lf = - np.sign(wlf) * self.brake_torque(t)
        br_t_rf = - np.sign(wrf) * self.brake_torque(t)
        br_t_lr = - np.sign(wlr) * self.brake_torque(t)
        br_t_rr = - np.sign(wrr) * self.brake_torque(t)

        # Tire rotational model
        dwlf = (1/Jw)*(self.drive_torque(t, wlf) /
                       4 + rolling_res_lf + br_t_lf - Fxtlf*Rlf)
        dwrf = (1/Jw)*(self.drive_torque(t, wrf) /
                       4 + rolling_res_rf + br_t_rf - Fxtrf*Rrf)
        dwlr = (1/Jw)*(self.drive_torque(t, wlr) /
                       4 + rolling_res_lr + br_t_lr - Fxtlr*Rlr)
        dwrr = (1/Jw)*(self.drive_torque(t, wrr) /
                       4 + rolling_res_rr + br_t_rr - Fxtrr*Rrr)

        return np.stack([dx, dy, u_dot, v_dot, dpsi, dphi, wx_dot, wz_dot, dwlf, dwlr, dwrf, dwrr])

    # Vehicle model implemented with the Fiala tire model - No transient slip update, only steady state slips
    def model_fiala(self, t, state):
        a, b, Cf, Cr, Cxf, Cxr, m, Jz, r0, Jw, Jx, Jy, Jxz, h, cf, cr, muf, mur, ktf, ktr, hrcf, hrcr, krof, kror, brof, bror, rr = list(
            self.params.values())
        g = 9.8
        x, y, u, v, psi, phi, wx, wz, wlf, wlr, wrf, wrr = state

        # Some calculated parameters
        # the vertical distance from the sprung mass C.M. to the vehicle roll center.
        hrc = (hrcf*b+hrcr*a)/(a+b)
        mt = m+2*muf+2*mur  # vehicle total mass

        # Instantaneous tire radius
        Rrf = r0-self.xtrf
        Rlf = r0-self.xtlf
        Rlr = r0-self.xtlr
        Rrr = r0-self.xtrr

        # position of front and rear unsprung mass
        huf = Rrf
        hur = Rrr
        # the longitudinal and lateral velocities at the tire contact patch in coordinate frame 2
        ugrf = u+(wz*cf)/2
        vgrf = v+wz*a
        uglf = u-(wz*cf)/2
        vglf = v+wz*a
        uglr = u-(wz*cr)/2
        vglr = v-wz*b
        ugrr = u+(wz*cr)/2
        vgrr = v-wz*b

        # tire slip angle of each wheel - if else to handle 0 velocity condition - 8 DOF model cannot start from rest
        if((abs(ugrf) > 1e-4) or (abs(vgrf) > 1e-4)):
            delta_rf = np.arctan2(vgrf, np.abs(ugrf)) - \
                (self.steering(t) * self.max_steer)
            s_rf = (Rrf*wrf-(ugrf*np.cos(self.steering(t) * self.max_steer)+vgrf*np.sin(self.steering(t) * self.max_steer)))/np.abs(ugrf*np.cos(self.steering(t) * self.max_steer)
                                                                                                                                    + vgrf*np.sin(self.steering(t) * self.max_steer))
        else:
            delta_rf = self.steering(t) * self.max_steer
            s_rf = 0
        if((abs(uglf) > 1e-4) or (abs(vglf) > 1e-4)):
            delta_lf = np.arctan2(vglf, np.abs(uglf)) - \
                (self.steering(t) * self.max_steer)
            s_lf = (Rlf*wlf-(uglf*np.cos(self.steering(t) * self.max_steer)+vglf*np.sin(self.steering(t) * self.max_steer)))/np.abs(uglf*np.cos(self.steering(t) * self.max_steer)
                                                                                                                                    + vglf*np.sin(self.steering(t) * self.max_steer))
        else:
            delta_lf = self.steering(t) * self.max_steer
            s_lf = 0
        if((abs(uglr) > 1e-4) or (abs(vglr) > 1e-4)):
            delta_lr = np.arctan2(vglr, np.abs(uglr))
            s_lr = (Rlr*wlr-uglr)/np.abs(uglr)
        else:
            delta_lr = 0
            s_lr = 0
        if((abs(ugrr) > 1e-4) or (abs(vgrr) > 1e-4)):
            delta_rr = np.arctan2(vgrr, np.abs(ugrr))
            s_rr = (Rrr*wrr-ugrr)/np.abs(ugrr)
        else:
            delta_rr = 0
            s_rr = 0

        # Smoothing for the longitduinal and lateral forces
        # smth = self.smooth_step(t,0,self.start_time,1,self.start_time + self.trans_time)
        smth = 1

        ss_rf = min(np.sqrt(s_rf**2 + np.tan(delta_rf)**2), 1.)
        ss_lf = min(np.sqrt(s_lf**2 + np.tan(delta_lf)**2), 1.)
        ss_lr = min(np.sqrt(s_lr**2 + np.tan(delta_lr)**2), 1.)
        ss_rr = min(np.sqrt(s_rr**2 + np.tan(delta_rr)**2), 1.)

        # Coefficient of friction based on the slip
        u_rf = self.umax - (self.umax - self.umin)*ss_rf
        u_lf = self.umax - (self.umax - self.umin)*ss_lf
        u_lr = self.umax - (self.umax - self.umin)*ss_lr
        u_rr = self.umax - (self.umax - self.umin)*ss_rr

        # Critical longitudinal slip of the 4 tires
        s_crit_rf = np.abs((u_rf * self.Fzgrf) / (2 * Cxf))
        s_crit_lf = np.abs((u_lf * self.Fzglf) / (2 * Cxf))
        s_crit_lr = np.abs((u_lr * self.Fzglr) / (2 * Cxr))
        s_crit_rr = np.abs((u_rr * self.Fzgrr) / (2 * Cxr))

        # Longitudinal forces based on whether the tire is in the elastic regime or the sliding regime
        if(np.abs(s_rf) < s_crit_rf):
            Fxtrf = Cxf*s_rf*smth
        else:
            Fxtrf_1 = u_rf * np.abs(self.Fzgrf)
            Fxtrf_2 = np.abs((u_rf * self.Fzgrf)**2 / (4 * s_rf * Cxf))
            Fxtrf = np.sign(s_rf)*(Fxtrf_1 - Fxtrf_2)*smth

        if(np.abs(s_lf) < s_crit_lf):
            Fxtlf = Cxf*s_lf*smth
        else:
            Fxtlf_1 = u_lf * np.abs(self.Fzglf)
            Fxtlf_2 = np.abs((u_lf * self.Fzglf)**2 / (4 * s_lf * Cxf))
            Fxtlf = np.sign(s_lf)*(Fxtlf_1 - Fxtlf_2)*smth
            # print(f"{Fxtlf_1 - Fxtlf_2}")

        if(np.abs(s_lr) < s_crit_lr):
            Fxtlr = Cxr*s_lr*smth
        else:
            Fxtlr_1 = u_lr * np.abs(self.Fzglr)
            Fxtlr_2 = np.abs((u_lr * self.Fzglr)**2 / (4 * s_lr * Cxr))
            Fxtlr = np.sign(s_lr)*(Fxtlr_1 - Fxtlr_2)*smth

        if(np.abs(s_rr) < s_crit_rr):
            Fxtrr = Cxr*s_rr*smth
        else:
            Fxtrr_1 = u_rr * np.abs(self.Fzgrr)
            Fxtrr_2 = np.abs((u_rr * self.Fzgrr)**2 / (4 * s_rr * Cxr))
            Fxtrr = np.sign(s_rr)*(Fxtrr_1 - Fxtrr_2)*smth

        # Critical lateral slip to understand if the tire is in the elastic regime or the sliding regime
        al_crit_rf = np.arctan((3*u_rf * np.abs(self.Fzgrf))/Cf)
        al_crit_lf = np.arctan((3*u_lf * np.abs(self.Fzglf))/Cf)
        al_crit_lr = np.arctan((3*u_lr * np.abs(self.Fzglr))/Cr)
        al_crit_rr = np.arctan((3*u_rr * np.abs(self.Fzgrr))/Cr)

        # The lateral forces based on whether the tire is in elastic regime or the sliding regime
        if(np.abs(delta_rf) <= al_crit_rf):
            h_ = 1 - ((Cf * np.abs(np.tan(delta_rf))) /
                      (3 * u_rf * np.abs(self.Fzgrf)))
            Fytrf = -u_rf * np.abs(self.Fzgrf) * \
                (1-h_**3)*np.sign(delta_rf)*smth
        else:
            Fytrf = -u_rf * np.abs(self.Fzgrf) * np.sign(delta_rf)*smth

        if(np.abs(delta_lf) <= al_crit_lf):
            h_ = 1 - ((Cf * np.abs(np.tan(delta_lf))) /
                      (3 * u_lf * np.abs(self.Fzglf)))
            Fytlf = -u_lf * np.abs(self.Fzglf) * \
                (1-h_**3)*np.sign(delta_lf)*smth
        else:
            Fytlf = -u_lf * np.abs(self.Fzglf) * np.sign(delta_lf)*smth

        if(np.abs(delta_lr) <= al_crit_lr):
            h_ = 1 - ((Cr * np.abs(np.tan(delta_lr))) /
                      (3 * u_lr * np.abs(self.Fzglr)))
            Fytlr = -u_lr * np.abs(self.Fzglr) * \
                (1-h_**3)*np.sign(delta_lr)*smth
        else:
            Fytlr = -u_lr * np.abs(self.Fzglr) * np.sign(delta_lr)*smth

        if(np.abs(delta_rr) <= al_crit_rr):
            h_ = 1 - ((Cr * np.abs(np.tan(delta_rr))) /
                      (3 * u_rr * np.abs(self.Fzgrr)))
            Fytrr = -u_rr * np.abs(self.Fzgrr) * \
                (1-h_**3)*np.sign(delta_rr)*smth
        else:
            Fytrr = -u_rr * np.abs(self.Fzgrr) * np.sign(delta_rr)*smth

        # the forces Fxgij obtained by resolving the longitudinal and cornering forces at the tire contact patch
        Fxglf = Fxtlf*np.cos(self.steering(t) * self.max_steer) - \
            Fytlf*np.sin(self.steering(t) * self.max_steer)
        Fxgrf = Fxtrf*np.cos(self.steering(t) * self.max_steer) - \
            Fytrf*np.sin(self.steering(t) * self.max_steer)
        Fxglr = Fxtlr
        Fxgrr = Fxtrr
        Fyglf = Fxtlf*np.sin(self.steering(t) * self.max_steer) + \
            Fytlf*np.cos(self.steering(t) * self.max_steer)
        Fygrf = Fxtrf*np.sin(self.steering(t) * self.max_steer) + \
            Fytrf*np.cos(self.steering(t) * self.max_steer)
        Fyglr = Fytlr
        Fygrr = Fytrr

        # Some other constants used in the differential equations
        E1 = -mt*wz*u+(Fyglf+Fygrf+Fyglr+Fygrr)
        E2 = (Fyglf+Fygrf)*a-(Fyglr+Fygrr)*b+(Fxgrf-Fxglf) * \
            cf/2+(Fxgrr-Fxglr)*cr/2+(mur*b-muf*a)*wz*u
        E3 = m*g*hrc*phi-(krof+kror)*phi-(brof+bror)*wx+hrc*m*wz*u
        A1 = mur*b-muf*a
        A2 = Jx+m*hrc**2
        A3 = hrc*m

        # Chassis Model
        u_dot = wz*v+(1/mt)*((Fxglf+Fxgrf+Fxglr+Fxgrr) +
                             (muf*a-mur*b)*(wz)**2-2*hrc*m*wz*wx)
        v_dot = (E1*Jxz**2-A1*A2*E2+A1*E3*Jxz+A3*E2*Jxz-A2*E1*Jz-A3 *
                 E3*Jz)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
        wx_dot = (A1**2*E3-A1*A3*E2+A1*E1*Jxz-A3*E1*Jz+E2*Jxz*mt-E3 *
                  Jz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
        wz_dot = (A3**2*E2-A1*A2*E1-A1*A3*E3+A3*E1*Jxz-A2*E2*mt+E3 *
                  Jxz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
        dx = u*np.cos(psi) - v*np.sin(psi)
        dy = u*np.sin(psi) + v*np.cos(psi)
        dpsi = wz
        dphi = wx

        # Smoothing for the 8dof model depends on the velocity - Similar to chrono implementation
        vx_min = 0.125
        vx_max = 0.5

        # Smoothing for rolling resistance based on the velocity - Inspiration from chrono model
        smth_rr = self.smooth_step(np.abs(ugrf), 0, vx_min, 1, vx_max)

        rolling_res_lf = -rr * np.abs(self.Fzglf) * np.sign(wlf)*smth_rr
        rolling_res_rf = -rr * np.abs(self.Fzgrf) * np.sign(wrf)*smth_rr
        rolling_res_lr = -rr * np.abs(self.Fzglr) * np.sign(wlr)*smth_rr
        rolling_res_rr = -rr * np.abs(self.Fzgrr) * np.sign(wrr)*smth_rr

        # Brake Torques of each wheel
        br_t_lf = - np.sign(wlf) * self.brake_torque(t)
        br_t_rf = - np.sign(wrf) * self.brake_torque(t)
        br_t_lr = - np.sign(wlr) * self.brake_torque(t)
        br_t_rr = - np.sign(wrr) * self.brake_torque(t)


        # Wheel rotational model
        dwlf = (1/Jw)*(self.drive_torque(t, wlf) /
                       4 + rolling_res_lf + br_t_lf - Fxtlf*Rlf)
        dwrf = (1/Jw)*(self.drive_torque(t, wrf) /
                       4 + rolling_res_rf + br_t_rf- Fxtrf*Rrf)
        dwlr = (1/Jw)*(self.drive_torque(t, wlr) /
                       4 + rolling_res_lr + br_t_lr - Fxtlr*Rlr)
        dwrr = (1/Jw)*(self.drive_torque(t, wrr) /
                       4 + rolling_res_rr + br_t_rr - Fxtrr*Rrr)

        # The normal forces at four tires are determined as in order to update the tire compression for the next time step
        Z1 = (m*g*b)/(2*(a+b))+(muf*g)/2
        Z2 = ((muf*huf)/cf+m*b*(h-hrcf)/(cf*(a+b)))*(v_dot+wz*u)
        Z3 = (krof*phi+brof*dphi)/cf
        Z4 = ((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b))
        self.Fzglf = (Z1-Z2-Z3-Z4)
        self.Fzgrf = (Z1+Z2+Z3-Z4)
        Z5 = (m*g*a)/(2*(a+b))+(mur*g)/2
        Z6 = ((mur*hur)/cr+m*a*(h-hrcr)/(cr*(a+b)))*(v_dot+wz*u)
        Z7 = (kror*phi+bror*dphi)/cr
        Z8 = ((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b))
        self.Fzglr = (Z5-Z6-Z7+Z8)
        self.Fzgrr = (Z5+Z6+Z7+Z8)

        # Normal force on the tire cannot be negetive
        if(self.Fzgrf < 0):
            self.Fzgrf = 0
        if(self.Fzglf < 0):
            self.Fzglf = 0
        if(self.Fzglr < 0):
            self.Fzglr = 0
        if(self.Fzgrr < 0):
            self.Fzgrr = 0

        if(self.debug):
            self.t_arr.append(t)
            self.flf.append(self.Fzglf)
            self.flr.append(self.Fzglr)
            self.frf.append(self.Fzgrf)
            self.frr.append(self.Fzgrr)
            self.dt.append(self.drive_torque(t, wlr)/4)
            self.fdt.append(Fxtlf*Rlf)
            self.rdt.append(rolling_res_lf)
            self.s_arr.append(s_lr)
            self.xtrf_ar.append(self.xtlf)

        # Tire deflection calculated from the normal force
        self.xtlf = self.Fzglf/ktf
        self.xtrf = self.Fzgrf/ktf
        self.xtlr = self.Fzglr/ktr
        self.xtrr = self.Fzgrr/ktr

        return np.stack([dx, dy, u_dot, v_dot, dpsi, dphi, wx_dot, wz_dot, dwlf, dwlr, dwrf, dwrr])

    def lvl_2_fiala(self,t,state):
        a,b,Cf,Cr,Cxf,Cxr,m,Jz,r0,Jw,Jx,Jy,Jxz,h,cf,cr,muf,mur,ktf,ktr,hrcf,hrcr,krof,kror,brof,bror,rr =list(self.params.values())
        g = 9.8
        x,y,u,v,psi,phi,wx,wz,wlf,wlr,wrf,wrr = state

        ## Some calculated parameters
        hrc=(hrcf*b+hrcr*a)/(a+b) # the vertical distance from the sprung mass C.M. to the vehicle roll center.
        mt=m+2*muf+2*mur # vehicle total mass


        #Instantaneous tire radius
        Rrf=r0-self.xtrf
        Rlf=r0-self.xtlf
        Rlr=r0-self.xtlr
        Rrr=r0-self.xtrr
        
        ##position of front and rear unsprung mass
        huf=Rrf 
        hur=Rrr
        ##the longitudinal and lateral velocities at the tire contact patch in coordinate frame 2
        ugrf=u+(wz*cf)/2
        vgrf=v+wz*a
        uglf=u-(wz*cf)/2
        vglf=v+wz*a
        uglr=u-(wz*cr)/2
        vglr=v-wz*b
        ugrr=u+(wz*cr)/2
        vgrr=v-wz*b
            
        # tire slip angle of each wheel - if else to handle 0 velocity condition - 8 DOF model cannot start from rest
        if((abs(ugrf) > 1e-4) or (abs(vgrf) > 1e-4)):
            delta_rf=np.arctan2(vgrf,np.abs(ugrf))-(self.steering(t)* self.max_steer)
            s_rf = (Rrf*wrf-(ugrf*np.cos(self.steering(t)* self.max_steer)+vgrf*np.sin(self.steering(t)* self.max_steer)))/np.abs(ugrf*np.cos(self.steering(t)* self.max_steer)
            +vgrf*np.sin(self.steering(t)* self.max_steer)) 
        else:
            delta_rf = self.steering(t)* self.max_steer
            s_rf = 0
        if((abs(uglf) > 1e-4) or (abs(vglf) > 1e-4)):
            delta_lf = np.arctan2(vglf,np.abs(uglf))-(self.steering(t)* self.max_steer)
            s_lf=(Rlf*wlf-(uglf*np.cos(self.steering(t)* self.max_steer)+vglf*np.sin(self.steering(t)* self.max_steer)))/np.abs(uglf*np.cos(self.steering(t)* self.max_steer)
            +vglf*np.sin(self.steering(t)* self.max_steer)) 
        else:
            delta_lf = self.steering(t)* self.max_steer
            s_lf = 0
        if((abs(uglr) > 1e-4) or (abs(vglr) > 1e-4)):
            delta_lr=np.arctan2(vglr,np.abs(uglr))
            s_lr=(Rlr*wlr-uglr)/np.abs(uglr)
        else:
            delta_lr = 0
            s_lr = 0
        if((abs(ugrr) > 1e-4) or (abs(vgrr) > 1e-4)):
            delta_rr = np.arctan2(vgrr,np.abs(ugrr))
            s_rr=(Rrr*wrr-ugrr)/np.abs(ugrr)
        else:
            delta_rr = 0
            s_rr = 0

            

        # Smoothing for the longitduinal and lateral forces
        # smth = self.smooth_step(t,0,self.start_time,1,self.start_time + self.trans_time)
        smth = 1


        ss_rf = min(np.sqrt(s_rf**2 + np.tan(delta_rf)**2),1.)
        ss_lf = min(np.sqrt(s_lf**2 + np.tan(delta_lf)**2),1.)
        ss_lr = min(np.sqrt(s_lr**2 + np.tan(delta_lr)**2),1.)
        ss_rr = min(np.sqrt(s_rr**2 + np.tan(delta_rr)**2),1.)

        # Coefficient of friction based on the slip
        u_rf = self.umax - (self.umax - self.umin)*ss_rf
        u_lf = self.umax - (self.umax - self.umin)*ss_lf
        u_lr = self.umax - (self.umax - self.umin)*ss_lr
        u_rr = self.umax - (self.umax - self.umin)*ss_rr

        
        u_rf = u_rf * (0.9/0.8)
        u_lf = u_lf * (0.9/0.8)
        u_lr = u_lr * (0.9/0.8)
        u_rr = u_rr * (0.9/0.8)


        # Critical longitudinal slip of the 4 tires
        s_crit_rf = np.abs((u_rf * self.Fzgrf) / (2 * Cxf))	
        s_crit_lf = np.abs((u_lf * self.Fzglf) / (2 * Cxf))
        s_crit_lr = np.abs((u_lr * self.Fzglr) / (2 * Cxr))
        s_crit_rr = np.abs((u_rr * self.Fzgrr) / (2 * Cxr))
        # print(u_lr, ss_lr,s_crit_lr)



        ## Longitudinal forces based on whether the tire is in the elastic regime or the sliding regime
        if(np.abs(s_rf)  < s_crit_rf):
            Fxtrf=Cxf*s_rf*smth
        else:
            Fxtrf_1 = u_rf * np.abs(self.Fzgrf)
            Fxtrf_2 = np.abs((u_rf * self.Fzgrf)**2 / (4 * s_rf * Cxf))
            Fxtrf = np.sign(s_rf)*(Fxtrf_1 - Fxtrf_2)*smth
            

        if(np.abs(s_lf)  < s_crit_lf):
            Fxtlf=Cxf*s_lf*smth
        else:
            
            Fxtlf_1 = u_lf * np.abs(self.Fzglf)
            Fxtlf_2 = np.abs((u_lf * self.Fzglf)**2 / (4 * s_lf * Cxf))
            Fxtlf = np.sign(s_lf)*(Fxtlf_1 - Fxtlf_2)*smth
            # print(f"{Fxtlf_1 - Fxtlf_2}")


        if(np.abs(s_lr)  < s_crit_lr):
            Fxtlr=Cxr*s_lr*smth
        else:
            # print(t)
            Fxtlr_1 = u_lr * np.abs(self.Fzglr)
            Fxtlr_2 = np.abs((u_lr * self.Fzglr)**2 / (4 * s_lr * Cxr))
            Fxtlr = np.sign(s_lr)*(Fxtlr_1 - Fxtlr_2)*smth

        if(np.abs(s_rr)  < s_crit_rr):
            Fxtrr=Cxr*s_rr*smth
        else:
            Fxtrr_1 = u_rr * np.abs(self.Fzgrr)
            Fxtrr_2 = np.abs((u_rr * self.Fzgrr)**2 / (4 * s_rr * Cxr))
            Fxtrr = np.sign(s_rr)*(Fxtrr_1 - Fxtrr_2)*smth
        


        # Critical lateral slip to understand if the tire is in the elastic regime or the sliding regime
        al_crit_rf = np.arctan((3*u_rf * np.abs(self.Fzgrf))/Cf)
        al_crit_lf = np.arctan((3*u_lf * np.abs(self.Fzglf))/Cf)
        al_crit_lr = np.arctan((3*u_lr * np.abs(self.Fzglr))/Cr)
        al_crit_rr = np.arctan((3*u_rr * np.abs(self.Fzgrr))/Cr)


        ## The lateral forces based on whether the tire is in elastic regime or the sliding regime 
        if(np.abs(delta_rf) <= al_crit_rf):
            h_ = 1 - ((Cf * np.abs(np.tan(delta_rf))) / (3 * u_rf * np.abs(self.Fzgrf)))
            Fytrf = -u_rf * np.abs(self.Fzgrf) * (1-h_**3)*np.sign(delta_rf)*smth
        else:
            Fytrf = -u_rf * np.abs(self.Fzgrf) * np.sign(delta_rf)*smth

        if(np.abs(delta_lf) <= al_crit_lf):
            h_ = 1 - ((Cf * np.abs(np.tan(delta_lf))) / (3 * u_lf * np.abs(self.Fzglf)))
            Fytlf = -u_lf * np.abs(self.Fzglf) * (1-h_**3)*np.sign(delta_lf)*smth
        else:
            Fytlf = -u_lf * np.abs(self.Fzglf) * np.sign(delta_lf)*smth


        if(np.abs(delta_lr) <= al_crit_lr):
            h_ = 1 - ((Cr * np.abs(np.tan(delta_lr))) / (3 * u_lr * np.abs(self.Fzglr)))
            Fytlr = -u_lr * np.abs(self.Fzglr) * (1-h_**3)*np.sign(delta_lr)*smth
        else:
            Fytlr = -u_lr * np.abs(self.Fzglr) * np.sign(delta_lr)*smth

        if(np.abs(delta_rr) <= al_crit_rr):
            h_ = 1 - ((Cr * np.abs(np.tan(delta_rr))) / (3 * u_rr * np.abs(self.Fzgrr)))
            Fytrr = -u_rr * np.abs(self.Fzgrr) * (1-h_**3)*np.sign(delta_rr)*smth
        else:
            Fytrr = -u_rr * np.abs(self.Fzgrr) * np.sign(delta_rr)*smth


        ## the forces Fxgij obtained by resolving the longitudinal and cornering forces at the tire contact patch 
        Fxglf=Fxtlf*np.cos(self.steering(t)* self.max_steer)- \
        Fytlf*np.sin(self.steering(t)* self.max_steer) 
        Fxgrf=Fxtrf*np.cos(self.steering(t)* self.max_steer)- \
        Fytrf*np.sin(self.steering(t)* self.max_steer) 
        Fxglr=Fxtlr 
        Fxgrr=Fxtrr 
        Fyglf=Fxtlf*np.sin(self.steering(t)* self.max_steer)+ \
        Fytlf*np.cos(self.steering(t)* self.max_steer) 
        Fygrf=Fxtrf*np.sin(self.steering(t)* self.max_steer)+ \
        Fytrf*np.cos(self.steering(t)* self.max_steer) 
        Fyglr=Fytlr 
        Fygrr=Fytrr


        # Some other constants used in the differential equations
        E1=-mt*wz*u+(Fyglf+Fygrf+Fyglr+Fygrr) 
        E2=(Fyglf+Fygrf)*a-(Fyglr+Fygrr)*b+(Fxgrf-Fxglf)*cf/2+(Fxgrr-Fxglr)*cr/2+(mur*b-muf*a)*wz*u
        E3=m*g*hrc*phi-(krof+kror)*phi-(brof+bror)*wx+hrc*m*wz*u 
        A1=mur*b-muf*a 
        A2=Jx+m*hrc**2 
        A3=hrc*m

        # Level 2 variables of the chassis model
        u_dot = wz*v+(1/mt)*((Fxglf+Fxgrf+Fxglr+Fxgrr)+(muf*a-mur*b)*(wz)**2-2*hrc*m*wz*wx) 
        v_dot=(E1*Jxz**2-A1*A2*E2+A1*E3*Jxz+A3*E2*Jxz-A2*E1*Jz-A3*E3*Jz)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)
        wx_dot=(A1**2*E3-A1*A3*E2+A1*E1*Jxz-A3*E1*Jz+E2*Jxz*mt-E3*Jz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt) 
        wz_dot=(A3**2*E2-A1*A2*E1-A1*A3*E3+A3*E1*Jxz-A2*E2*mt+E3*Jxz*mt)/(A2*A1**2-2*A1*A3*Jxz+Jz*A3**2+mt*Jxz**2-A2*Jz*mt)

        # Now the tire model
        # Smoothing for the 8dof model depends on the velocity - Similar to chrono implementation
        vx_min = 0.125
        vx_max = 0.5

        # Smoothing for rolling resistance based on the velocity - Inspiration from chrono model
        # smth_rr = self.smooth_step(np.abs(ugrf),0,vx_min,1,vx_max)
        smth_rr = 1

        rolling_res_lf = -rr * np.abs(self.Fzglf) * np.sign(wlf)*smth_rr
        rolling_res_rf = -rr * np.abs(self.Fzgrf) * np.sign(wrf)*smth_rr
        rolling_res_lr = -rr * np.abs(self.Fzglr) * np.sign(wlr)*smth_rr
        rolling_res_rr = -rr * np.abs(self.Fzgrr) * np.sign(wrr)*smth_rr

        # Brake Torques of each wheel
        br_t_lf = - np.sign(wlf) * self.brake_torque(t)
        br_t_rf = - np.sign(wrf) * self.brake_torque(t)
        br_t_lr = - np.sign(wlr) * self.brake_torque(t)
        br_t_rr = - np.sign(wrr) * self.brake_torque(t)

        # Wheel rotational model
        dwlf=(1/Jw)*(self.drive_torque(t,wlf)/4 + rolling_res_lf + br_t_lf - Fxtlf*Rlf)
        dwrf=(1/Jw)*(self.drive_torque(t,wrf)/4 + rolling_res_rf + br_t_rf - Fxtrf*Rrf)
        dwlr=(1/Jw)*(self.drive_torque(t,wlr)/4 + rolling_res_lr + br_t_lr - Fxtlr*Rlr)
        dwrr=(1/Jw)*(self.drive_torque(t,wrr)/4 + rolling_res_rr + br_t_rr - Fxtrr*Rrr)


        if(self.debug):
            self.dt.append(self.drive_torque(t,wlr)/4)
            self.fdt.append(Fxtlf*Rlf)
            self.rdt.append(rolling_res_lf)
            self.s_arr.append(s_lr)


        return [u_dot,v_dot,wx_dot,wz_dot,dwlf,dwrf,dwlr,dwrr]



    def solve_half_impl(self,t_span,t_eval,tbar):
        a,b,Cf,Cr,Cxf,Cxr,m,Jz,r0,Jw,Jx,Jy,Jxz,h,cf,cr,muf,mur,ktf,ktr,hrcf,hrcr,krof,kror,brof,bror,rr =list(self.params.values())
        g = 9.8
        self.start_time = t_eval[0]
        ## Some calculated parameters
        hrc=(hrcf*b+hrcr*a)/(a+b) # the vertical distance from the sprung mass C.M. to the vehicle roll center.
        mt=m+2*muf+2*mur # vehicle total mass

        time_steps = np.arange(t_span[0]+tbar,t_span[1]+0.0000001,tbar)
        self.level_1_vars = np.array([self.states['u'],self.states['v'],self.states['wx'],self.states['wz'],self.states['wlf'],self.states['wrf']
        ,self.states['wlr'],self.states['wrr']])
        self.level_0_vars = np.array([self.states['x'],self.states['y'],self.states['phi'],self.states['psi']])

        # Save n every 
        n_ = round((t_eval[1]-t_eval[0])/tbar)
        
        # For saving the results at t_eval
        count = 1

        outs = np.empty((len(t_eval),len(self.states)))
        outs[0] = (np.array(list(self.states.values())))

        for t in time_steps:
            level_2_vars = self.lvl_2_fiala(t,list(self.states.values()))


            self.level_1_vars = self.level_1_vars + tbar * np.array(level_2_vars)

            self.level_0_vars[0] = self.level_0_vars[0] + tbar * (self.level_1_vars[0]*np.cos(self.level_0_vars[3]) - 
            self.level_1_vars[1]*np.sin(self.level_0_vars[3]))

            self.level_0_vars[1] = self.level_0_vars[1] + tbar * (self.level_1_vars[0]*np.sin(self.level_0_vars[3]) + 
            self.level_1_vars[1]*np.cos(self.level_0_vars[3]))

            self.level_0_vars[2:4] = self.level_0_vars[2:4] + tbar * (self.level_1_vars[2:4])

            x,y,phi,psi = self.level_0_vars
            u,v,wx,wz,wlf,wrf,wlr,wrr = self.level_1_vars
            u_dot,v_dot,wx_dot,wz_dot,dwlf,dwrf,dwlr,dwrr = level_2_vars
            st = [x,y,u,v,psi,phi,wx,wz,wlf,wlr,wrf,wrr]
            self.states.update(zip(self.states,st))

            # Instantaneous tire radius
            Rrf=r0-self.xtrf
            Rlf=r0-self.xtlf
            Rlr=r0-self.xtlr
            Rrr=r0-self.xtrr
        
            # smth_fz = self.smooth_step(t,0,self.start_time,1,self.start_time + self.trans_time)
            smth_fz = 1	   
            ## position of front and rear unsprung mass
            huf=Rrf 
            hur=Rrr
            ## The normal forces at four tires are determined as in order to update the tire compression for the next time step
            Z1=(m*g*b)/(2*(a+b))+(muf*g)/2 
            Z2=((muf*huf)/cf+m*b*(h-hrcf)/(cf*(a+b)))*(v_dot+wz*u) 
            Z3=(krof*phi+brof*wx)/cf 
            Z4=((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b)) 
            self.Fzglf=(Z1-Z2-Z3-Z4)*smth_fz
            self.Fzgrf=(Z1+Z2+Z3-Z4)*smth_fz
            Z5=(m*g*a)/(2*(a+b))+(mur*g)/2 
            Z6=((mur*hur)/cr+m*a*(h-hrcr)/(cr*(a+b)))*(v_dot+wz*u) 
            Z7=(kror*phi+bror*wx)/cr 
            Z8=((m*h+muf*huf+mur*hur)*(u_dot-wz*v))/(2*(a+b)) 
            self.Fzglr=(Z5-Z6-Z7+Z8)*smth_fz 
            self.Fzgrr=(Z5+Z6+Z7+Z8)*smth_fz

            # Normal force on the tire cannot be negetive
            if(self.Fzgrf < 0):
                self.Fzgrf = 0
            if(self.Fzglf < 0):
                self.Fzglf = 0
            if(self.Fzglr < 0):
                self.Fzglr = 0
            if(self.Fzgrr < 0):
                self.Fzgrr = 0

            if(self.debug):
                self.t_arr.append(t)
                self.flf.append(self.Fzglf)
                self.flr.append(self.Fzglr)
                self.frf.append(self.Fzgrf)
                self.frr.append(self.Fzgrr)
                self.xtrf_ar.append(self.xtlf)

            # Tire deflection calculated from the normal force
            self.xtlf=self.Fzglf/ktf 
            self.xtrf=self.Fzgrf/ktf 
            self.xtlr=self.Fzglr/ktr 
            self.xtrr=self.Fzgrr/ktr 

            # Save the required time

            if(count%n_ == 0):
                outs[round(count/n_)] = (np.array(list(self.states.values())))
            count = count + 1
        return outs

    # A wrapper function for maybe a few packages - starting with solve_ivp, odeint
    def solve(self,package = 'half_implicit',tire_model = 1,t_eval = None,tbar = 1e-2,**kwargs):
        try:
            self.steering
        except:
            raise Exception("Please provide steering controls for the vehicle with 'set_steering' method")
        
        try:
            self.throttle
        except:
            raise Exception("Please provide throttle controls for the vehicle with 'set_throttle' method")

        if t_eval is None:
            raise Exception("Please provide times steps at which you want the solution to be evaluated")

        try:
            self.brake
        except:
            def zero_brake(t):
                return 0*t
            self.brake = zero_brake
            self.max_brake_torque = 4000

        # Need the start time for the smoothing function
        self.start_time = t_eval[0]

        if(tire_model == 1):
            if(package == 'half_implicit'):
                return self.solve_half_impl(t_span = [t_eval[0],t_eval[-1]],t_eval = t_eval,tbar = 1e-2)
            else:
                return solve_ivp(self.model_fiala,t_span=[t_eval[0],t_eval[-1]],y0 = list(self.states.values()),
                vectorized = False,
                t_eval = t_eval,**kwargs)
        else:
            return solve_ivp(self.model_linear,t_span=[t_eval[0],t_eval[-1]],y0 = list(self.states.values()),vectorized = False,
            t_eval = t_eval,**kwargs)