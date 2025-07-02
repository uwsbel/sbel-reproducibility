import numpy as np
import pychrono as chrono
import sys,os
# Add the parent directory of 'models' to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)



class simplifiedVehModel():
    def __init__(self,system, state, control ,dt, Visualize = True):
        self.system = system
        # state = [x, y, theta,v]
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.v = state[3]
        self.accel = 0
        # control = [alpha, beta]
        self.alpha = control[0]
        self.beta = control[1]
        self.dt = dt # time step
        self.vis = Visualize

        if Visualize:
            self.veh_chassis = chrono.ChBodyEasyMesh(project_root+'/CCTA-highwayControl/data/audi.obj', 1000, True, True, False,chrono.ChContactMaterialNSC())
            self.veh_chassis.GetVisualShape(0).SetVisible(True)
            self.veh_chassis.SetMass(0)
            self.veh_chassis.SetFixed(False)
            self.veh_chassis.SetPos(chrono.ChVector3d(self.x, self.y, 0.6))
            self.veh_chassis.SetRot(chrono.QuatFromAngleZ(float(self.theta + np.pi)))
            system.Add(self.veh_chassis)
            self.visualize_vehicle()
    
    def visualize_vehicle(self):
        # self.veh_chassis.GetVisualShape(0).SetVisible(True)
        veh_pos = chrono.ChVector3d(self.x, self.y, 0.75)
        veh_rot = chrono.QuatFromAngleZ(float(self.theta + np.pi))
        self.veh_chassis.SetPos(veh_pos)
        self.veh_chassis.SetRot(veh_rot)

        

    def pause_visualization(self):
        self.set_state([0,0,-15,0])
        self.visualize_vehicle()


    def update(self, control):
        # car parameters
        delta = 0.667
        l = 2.5
        tau0 = 100
        omega0 = 1200
        gamma = 1/3
        r_wheel = 0.3
        i_wheel = 0.6
        c0,c1 = 0.01, 0.02

        self.alpha = control[0]
        self.beta = control[1]

        omega_m = self.v/(r_wheel*gamma)
        helpfun1 = - tau0 * omega_m / omega0 + tau0
        helpfunT = self.alpha * helpfun1 - c1*omega_m - c0

        self.x += self.v*np.cos(self.theta)*self.dt
        self.y += self.v*np.sin(self.theta)*self.dt
        self.theta += (self.v*np.tan(self.beta * delta)/l)*self.dt
        self.accel = helpfunT * gamma / i_wheel * r_wheel
        self.v += self.accel * self.dt

        # update the visualization
        if self.vis:
            self.visualize_vehicle()
    
    def set_state(self, state):
        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]
        self.v = state[3]
    
    def get_state(self):
        return [self.x, self.y, self.theta, self.v, self.accel] 

# vehicle = simplifiedVehModel([0,0,0,0],[0,0],0.05)
# vehicle.set_state([1,1,0,0])
# for i in range(100):
#     vehicle.update([0.1,0.2])
#     print(vehicle.x, vehicle.y, vehicle.theta, vehicle.v)        
            
