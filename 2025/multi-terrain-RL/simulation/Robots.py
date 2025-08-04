import pychrono as chrono
import pychrono.irrlicht as irr
try:
    import pychrono.vsg as vsg
except:
    pass
import pychrono.parsers as parsers

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import time
import numpy as np

class A1Robot:
    def __init__(self, chsystem: chrono.ChSystemSMC, 
                       initial_state: chrono.ChFramed = chrono.ChFramed(chrono.ChVector3d(0, 0, 0.5), chrono.QuatFromAngleZ(0.0)),
                       actuation_type = parsers.ChParserURDF.ActuationType_POSITION):
        self.chsystem = chsystem
        urdf_filename = project_root + "/data/robot/unitree_a1/urdf/a1sub.urdf"
        self.foot_bodies_list = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.calf_bodies_list = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
        self.thigh_bodies_list = ["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]

        self.robot = parsers.ChParserURDF(urdf_filename)
        self._set_initial_pose(initial_state)
        self._set_all_joints_actuation_type(actuation_type)
        #self._set_collision_type()
        self.robot.PopulateSystem(self.chsystem)
        self.robot.GetRootChBody().SetFixed(False)
        
        self._set_collision()
        # motor name list
        self.motor_name_list = ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"]
        self._setup_motors()

    def get_contact_force(self)->np.ndarray:
        """
        Get the contact force on each foot in sequence of FR_foot, FL_foot, RR_foot, RL_foot
        """
        contact_force_list = []
        for foot_name in self.foot_bodies_list:
            foot_body = self.robot.GetChBody(foot_name)
            force = foot_body.GetContactForce()
            contact_force_list.append(force.z)
        return np.array(contact_force_list)

    def get_joint_pos(self)->np.ndarray:
        joint_pos_list = []
        for motor_name in self.motor_name_list:
            motor = self.robot.GetChMotor(motor_name)
            # Cast to ChLinkMotorRotation to access GetMotorAngle() method
            rotation_motor = chrono.CastToChLinkMotorRotation(motor)
            joint_pos_list.append(rotation_motor.GetMotorAngle())
        return np.array(joint_pos_list)

    def get_base_pos(self)->chrono.ChVector3d:
        return self.robot.GetChBody("trunk").GetPos()
    
    def get_base_body(self)->chrono.ChBody:
        """Get the base body of the robot"""
        return self.robot.GetChBody("trunk")
    
    def get_base_vel_global(self)->chrono.ChVector3d:
        return self.robot.GetChBody("trunk").GetLinVel()

    def get_base_vel_local(self)->chrono.ChVector3d:
        base_body = self.robot.GetChBody("trunk")
        global_vel = base_body.GetLinVel()
        local_vel = base_body.TransformDirectionParentToLocal(global_vel)
        return local_vel
    
    def get_base_heading(self)->float:
        return self.robot.GetChBody("trunk").GetRot().GetCardanAnglesZYX().z
    
    def get_base_quat(self)->chrono.ChQuaterniond:
        return self.robot.GetChBody("trunk").GetRot()
    
    def get_base_angvel_local(self)->chrono.ChVector3d:
        return self.robot.GetChBody("trunk").GetAngVelLocal()
    
    
    def actuate(self, motor_angles: np.ndarray):
        # Apply actions to motors
        for i, motor in enumerate(self.motor_list):
            cons_func = chrono.ChFunctionConst(float(motor_angles[i]))
            motor.SetMotorFunction(cons_func)
    
    def _setup_motors(self):
        self.motor_list = []
        for motor_name in self.motor_name_list:
            self.motor_list.append(self.robot.GetChMotor(motor_name))

    def print_motor_list(self):
        print(f"There are {len(self.motor_name_list)} motors with names in order: {self.motor_name_list}")

    def _set_initial_pose(self, initial_state: chrono.ChFramed):
        self.robot.SetRootInitPose(initial_state)
    
    def _set_all_joints_actuation_type(self, actuation_type):
        self.robot.SetAllJointsActuationType(actuation_type)

    def _set_collision_type(self):
        for bodyname in self.foot_bodies_list:
            self.robot.SetBodyMeshCollisionType(bodyname,parsers.ChParserURDF.MeshCollisionType_CONVEX_HULL)
            # self.robot.SetBodyMeshCollisionType(bodyname,parsers.ChParserURDF.MeshCollisionType_TRIANGLE_MESH)


    
    def _set_collision(self):
        mat = chrono.ChContactMaterialSMC()
        mat.SetRestitution(0.01)
        mat.SetFriction(0.9)
        mat.SetGn(60.0)
        # mat.SetGn(60.0)
        # mat.SetKn(2e5)
        # set collision for only foot bodies
        for bodyname in self.foot_bodies_list:
            footbody = self.robot.GetChBody(bodyname)
            footbody.EnableCollision(True)
            footbody.GetCollisionModel().SetAllShapesMaterial(mat)
        # disable collision for other bodies
        for bodyname in self.calf_bodies_list + self.thigh_bodies_list:
            body = self.robot.GetChBody(bodyname)
            body.EnableCollision(False)

class Go2Robot:
    def __init__(self, chsystem: chrono.ChSystemSMC, 
                       initial_state: chrono.ChFramed = chrono.ChFramed(chrono.ChVector3d(0, 0, 0.5), chrono.QuatFromAngleZ(np.pi/2)),
                       actuation_type = parsers.ChParserURDF.ActuationType_POSITION,
                       vis_engine: str = 'irrlicht'):
        self.chsystem = chsystem
        if vis_engine == 'irrlicht':
            urdf_filename = project_root + "/data/robot/go2_irrvis/urdf/go2_description.urdf"
        elif vis_engine == 'vsg':
            urdf_filename = project_root + "/data/robot/go2_vsgvis/urdf/go2_description.urdf"
        self.foot_bodies_list = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.calf_bodies_list = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
        self.thigh_bodies_list = ["FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]

        self.robot = parsers.ChParserURDF(urdf_filename)
        self._set_initial_pose(initial_state)
        self._set_all_joints_actuation_type(actuation_type)
        self._set_collision_type()
        self.robot.PopulateSystem(self.chsystem)
        self.robot.GetRootChBody().SetFixed(False)
        self._set_collision()
        # motor name list
        self.motor_name_list = ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"]
        self._setup_motors()

    def get_joint_speed(self)->np.ndarray:
        joint_speed_list = []
        for motor_name in self.motor_name_list:
            motor = self.robot.GetChMotor(motor_name)
            # Cast to ChLinkMotorRotation to access GetMotorAngle() method
            rotation_motor = chrono.CastToChLinkMotorRotation(motor)
            joint_speed_list.append(rotation_motor.GetMotorAngleDt())
        return np.array(joint_speed_list,dtype=np.float32)

    def get_joint_pos(self)->np.ndarray:
        joint_pos_list = []
        for motor_name in self.motor_name_list:
            motor = self.robot.GetChMotor(motor_name)
            # Cast to ChLinkMotorRotation to access GetMotorAngle() method
            rotation_motor = chrono.CastToChLinkMotorRotation(motor)
            joint_pos_list.append(rotation_motor.GetMotorAngle())
        return np.array(joint_pos_list,dtype=np.float32)

    def get_contact_force(self)->np.ndarray:
        """
        Get the contact force on each foot in sequence of FR_foot, FL_foot, RR_foot, RL_foot
        """
        contact_force_list = []
        for foot_name in self.foot_bodies_list:
            foot_body = self.robot.GetChBody(foot_name)
            force = foot_body.GetContactForce()
            contact_force_list.append(force.z)
        return np.array(contact_force_list)

    def get_base_pos(self)->chrono.ChVector3d:
        return self.robot.GetChBody("base").GetPos()
    
    def get_base_body(self)->chrono.ChBody:
        """Get the base body of the robot"""
        return self.robot.GetChBody("base")
    
    def get_base_vel_global(self)->chrono.ChVector3d:
        return self.robot.GetChBody("base").GetLinVel()

    def get_base_vel_local(self)->chrono.ChVector3d:
        base_body = self.robot.GetChBody("base")
        global_vel = base_body.GetLinVel()
        local_vel = base_body.TransformDirectionParentToLocal(global_vel)
        return local_vel
    
    def get_base_heading(self)->float:
        return self.robot.GetChBody("base").GetRot().GetCardanAnglesZYX().z
    
    def get_base_quat(self)->chrono.ChQuaterniond:
        return self.robot.GetChBody("base").GetRot()
    
    def get_base_angvel_local(self)->chrono.ChVector3d:
        return self.robot.GetChBody("base").GetAngVelLocal()

    def actuate(self, motor_angles: np.ndarray):
        # Apply actions to motors
        for i, motor in enumerate(self.motor_list):
            cons_func = chrono.ChFunctionConst(float(motor_angles[i]))
            motor.SetMotorFunction(cons_func)
    
    def _setup_motors(self):
        self.motor_list = []
        for motor_name in self.motor_name_list:
            self.motor_list.append(self.robot.GetChMotor(motor_name))

    def print_motor_list(self):
        print(f"There are {len(self.motor_name_list)} motors with names in order: {self.motor_name_list}")

    def _set_initial_pose(self, initial_state: chrono.ChFramed):
        self.robot.SetRootInitPose(initial_state)
    
    def _set_all_joints_actuation_type(self, actuation_type):
        self.robot.SetAllJointsActuationType(actuation_type)

    def _set_collision_type(self):
        for bodyname in self.foot_bodies_list:
            self.robot.SetBodyMeshCollisionType(bodyname,parsers.ChParserURDF.MeshCollisionType_CONVEX_HULL)

    def _set_collision(self):
        mat = chrono.ChContactMaterialSMC()
        mat.SetRestitution(0.01)
        mat.SetFriction(0.9)
        mat.SetGn(60.0)
        # mat.SetGn(60.0)
        # mat.SetKn(2e5)

        # set collision for only foot bodies
        for bodyname in self.foot_bodies_list:
            footbody = self.robot.GetChBody(bodyname)
            footbody.EnableCollision(True)
            footbody.GetCollisionModel().SetAllShapesMaterial(mat)
        # disable collision for other bodies
        for bodyname in self.calf_bodies_list + self.thigh_bodies_list:
            body = self.robot.GetChBody(bodyname)
            body.EnableCollision(False)
        