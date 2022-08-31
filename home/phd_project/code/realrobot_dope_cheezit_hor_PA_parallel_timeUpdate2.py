# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:57:49 2021

@author: 12106
"""
#ROS
import itertools
import os.path
from ssl import ALERT_DESCRIPTION_ILLEGAL_PARAMETER

import rospy
import threading
import rospkg
from std_msgs.msg import String
from std_msgs.msg import Float32
from std_msgs.msg import Int8
from std_msgs.msg import ColorRGBA, Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point,PointStamped,PoseStamped,Quaternion,TransformStamped, Vector3
import tf
import tf.transformations as transformations
from visualization_msgs.msg import Marker

#pybullet
from pyquaternion import Quaternion
import pybullet as p
import time
import pybullet_data
from pybullet_utils import bullet_client as bc
import numpy as np
import math
import random
import copy
import os
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing

from quaternion_averaging import weightedAverageQuaternions
'''
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
p.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])
planeId = p.loadURDF("plane.urdf")
'''


#visualisation_model
p_visualisation = bc.BulletClient(connection_mode=p.DIRECT)#DIRECT,GUI_SERVER
p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
p_visualisation.setGravity(0,0,-9.81)
p_visualisation.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])
plane_id = p_visualisation.loadURDF("plane.urdf")

boss_obse_err_sum_df = pd.DataFrame()
boss_PFPE_err_sum_df = pd.DataFrame()
boss_PFPM_err_sum_df = pd.DataFrame()

boss_obse_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
boss_PFPE_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
boss_PFPM_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
boss_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])

boss_obse_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
boss_PFPE_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
boss_PFPM_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
boss_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])

boss_obs_pose_PFPM = []
boss_est_pose_PFPM = []
boss_update_flag_obse = 0
boss_update_flag_PFPE = 0
boss_update_flag_PFPM = 0

'''
#load and set franka robot
plane_id = p_visualisation.loadURDF("plane.urdf")
real_franka_robot_start_pos = [0.127, -0.04, 0.03]
real_franka_robot_start_orientation = p_visualisation.getQuaternionFromEuler([0,0,0])
real_robot_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                         real_franka_robot_start_pos,
                                         real_franka_robot_start_orientation,
                                         useFixedBase=1)


#load and set object
cylinder_real_object_start_pos = [0.567, -0.3642, 0.057]
cylinder_real_object_start_orientation = p_visualisation.getQuaternionFromEuler([0,0,0])
real_object_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_object_small.urdf"),
                                          cylinder_real_object_start_pos,
                                          cylinder_real_object_start_orientation)
                                          
'''                                         
                                          
#Class of franka robot move
class Franka_robot():
    def __init__(self,franka_robot_id):
        self.franka_robot_id = franka_robot_id
    def fanka_robot_move(self,targetPositionsJoints):
        self.setJoinVisual(self.franka_robot_id,targetPositionsJoints)
        
    def setJoinVisual(self,robot, position):
        #position[7] = 0.039916139
        #position[8] = 0.039916139
        num_joints = 9
        for joint_index in range(num_joints): 
            if joint_index == 7 or joint_index == 8:
                p_visualisation.resetJointState(robot,
                                                joint_index+2,
                                                targetValue=position[joint_index])                
            else:
                p_visualisation.resetJointState(robot,
                                                joint_index,
                                                targetValue=position[joint_index])
    
    def setJointPosition(self,robot, position):
        #position[7] = 0.039916139
        #position[8] = 0.039916139
        num_joints = 9
        for joint_index in range(num_joints): 
            if joint_index == 7 or joint_index == 8:
                p_visualisation.setJointMotorControl2(robot,
                                                      joint_index+2,
                                                      p_visualisation.POSITION_CONTROL,
                                                      targetPosition=position[joint_index])                
            else:
                p_visualisation.setJointMotorControl2(robot,
                                                      joint_index,
                                                      p_visualisation.POSITION_CONTROL,
                                                      targetPosition=position[joint_index])

#Class of franka robot listen to info from ROS
class Ros_listener():
    def __init__(self):
        self.joint_subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback,queue_size=1)
        self.robot_pose = rospy.Subscriber('/mocap/rigid_bodies/pandaRobot/pose',PoseStamped, self.robot_pose_callback,queue_size=10)
        self.object_pose = rospy.Subscriber('/mocap/rigid_bodies/cheezit/pose',PoseStamped, self.object_pose_callback,queue_size=10)
        self.base_pose = rospy.Subscriber('/mocap/rigid_bodies/baseofcheezit/pose', PoseStamped,
                                          self.base_of_cheezit_callback, queue_size=10)
        self.current_joint_values = [-1.57,0.0,0.0,-2.8,1.7,1.57,1.1]
        self.robot_pos = [0.139080286026, -0.581342339516, 0.0238141193986]
        #x,y,z,w
        self.robot_ori = [0.707254290581, 0.0115503482521, -0.0140119809657, -0.706726074219]
        self.object_pos = [0.504023790359, -0.214561194181, 0.0601389780641]
        #x,y,z,w
        self.object_ori = [-0.51964700222, -0.476704657078, 0.490200251342, 0.512272834778]
        self.base_pos = [0, 0, 0]
        self.base_ori = [0, 0, 0, 1]
        #self.object_ori = [0,0,0,1]        
        rospy.spin
    def joint_values_callback(self, msg):
        self.current_joint_values = list(msg.position)
    def robot_pose_callback(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.robot_pos = [x_pos,y_pos,z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.robot_ori = [x_ori,y_ori,z_ori,w_ori]
    def object_pose_callback(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos,y_pos,z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori,y_ori,z_ori,w_ori]
    def base_of_cheezit_callback(self,data):
        # pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.base_pos = [x_pos, y_pos, z_pos]
        # ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.base_ori = [x_ori, y_ori, z_ori, w_ori]

#Class of particle's structure
class Particle(object):
    def __init__(self,x=0.0,y=0.0,z=0.0,x_angle=0.0,y_angle=0.0,z_angle=0.0,w=1.0,index = 0):
        self.x = x
        self.y = y
        self.z = z
        self.x_angle = x_angle
        self.y_angle = y_angle
        self.z_angle = z_angle
        self.w = w
        self.index = index
    def as_pose(self):
        return True

#Class of initialize the real world model
class InitialRealworldModel():
    def __init__(self,joint_pos=0):
        self.flag = 0
        self.joint_pos = joint_pos
    def initial_robot(self,robot_pos,robot_orientation = [0,0,0,1]):
        #robot_orientation = p_visualisation.getQuaternionFromEuler(robot_euler)
        real_robot_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                                 robot_pos,
                                                 robot_orientation,
                                                 useFixedBase=1)
        
        self.set_real_robot_JointPosition(p_visualisation,real_robot_id,self.joint_pos)
        for i in range(240):
            p_visualisation.stepSimulation()
            #time.sleep(1./240.)
        
        return real_robot_id
    def initial_target_object(self,object_pos,object_orientation = [0,0,0,1]):
        #object_orientation = p_visualisation.getQuaternionFromEuler(object_euler)
        real_object_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cube/cheezit_obj_small_hor.urdf"),
                                                  object_pos,
                                                  object_orientation)
        p_visualisation.changeDynamics(real_object_id,-1,mass=0.380,lateralFriction = 0.5)
        return real_object_id
    def set_real_robot_JointPosition(self,pybullet_simulation_env,robot, position):
        # print("Preparing the joint pose of the panda robot!")
        #position[7] = 0.039916139
        #position[8] = 0.039916139
        #num_joints = 7
        num_joints = 9
        for joint_index in range(num_joints): 
            if joint_index == 7 or joint_index == 8:
                pybullet_simulation_env.setJointMotorControl2(robot,
                                                              joint_index+2,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])                
            else:
                pybullet_simulation_env.setJointMotorControl2(robot,
                                                              joint_index,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])                                 
#Class of initialize the simulation model
class InitialSimulationModel():
    def __init__(self,particle_num,real_robot_start_pos,real_robot_start_ori,noise_obj_pos,noise_obj_ang,pw_T_object_ori_dope):
        self.particle_num = particle_num
        self.noise_obj_pos = noise_obj_pos
        self.noise_obj_ang = noise_obj_ang
        self.real_robot_start_pos = real_robot_start_pos
        self.real_robot_start_ori = real_robot_start_ori
        self.pw_T_object_ori_dope = pw_T_object_ori_dope
        self.particle_cloud = []
        self.pybullet_particle_env_collection = []
        self.fake_robot_id_collection = []
        self.particle_no_visual_id_collection = []
        self.particle_with_visual_id_collection =[]
        self.noise_object_pose = []
        
        self.particle_cloud_PM = []
        self.pybullet_particle_env_collection_PM = []
        self.particle_no_visual_id_collection_PM = []
        self.particle_with_visual_id_collection_PM =[]
        
    def initial_particle(self):
        noise_obj_x = copy.deepcopy(self.noise_obj_pos[0])
        noise_obj_y = copy.deepcopy(self.noise_obj_pos[1])
        noise_obj_z = copy.deepcopy(self.noise_obj_pos[2])
        noise_obj_pos = [noise_obj_x,noise_obj_y,noise_obj_z]
        noise_obj_x_ang = copy.deepcopy(self.noise_obj_ang[0])
        noise_obj_y_ang = copy.deepcopy(self.noise_obj_ang[1])
        noise_obj_z_ang = copy.deepcopy(self.noise_obj_ang[2])
        noise_obj_ang = [noise_obj_x_ang,noise_obj_y_ang,noise_obj_z_ang]
        
        self.noise_object_pose = [noise_obj_x,noise_obj_y,noise_obj_z,noise_obj_x_ang,noise_obj_y_ang,noise_obj_z_ang]

        for i in range(self.particle_num):
            x,y,z,x_angle,y_angle,z_angle,new_quat = self.generate_random_pose(self.noise_object_pose,self.pw_T_object_ori_dope)
            w = 1/self.particle_num
            particle = Particle(x,y,z,x_angle,y_angle,z_angle,w,index=i)
            self.particle_cloud.append(particle)

    def generate_random_pose(self,noise_object_pose, pw_T_object_ori_dope):
        #angle = copy.deepcopy([noise_object_pose[3],noise_object_pose[4],noise_object_pose[5]])
        quat = copy.deepcopy(pw_T_object_ori_dope)#x,y,z,w
        quat_QuatStyle = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3])#w,x,y,z
        x = self.add_noise_to_init_par(noise_object_pose[0],boss_sigma_obs_x)
        y = self.add_noise_to_init_par(noise_object_pose[1],boss_sigma_obs_y)
        z = self.add_noise_to_init_par(noise_object_pose[2],boss_sigma_obs_z)
        random_dir = random.uniform(0, 2*math.pi)
        z_axis = random.uniform(-1,1)
        x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
        y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
        angle_noise = self.add_noise_to_init_par(0,boss_sigma_obs_ang)
        w_quat = math.cos(angle_noise/2.0)
        x_quat = math.sin(angle_noise/2.0) * x_axis
        y_quat = math.sin(angle_noise/2.0) * y_axis
        z_quat = math.sin(angle_noise/2.0) * z_axis
        ###nois_quat(w,x,y,z); new_quat(w,x,y,z)
        nois_quat = Quaternion(x=x_quat,y=y_quat,z=z_quat,w=w_quat)
        new_quat = nois_quat * quat_QuatStyle
        ###pb_quat(x,y,z,w)
        pb_quat = [new_quat[1],new_quat[2],new_quat[3],new_quat[0]]
        new_angle = p_visualisation.getEulerFromQuaternion(pb_quat)
        x_angle = new_angle[0]
        y_angle = new_angle[1]
        z_angle = new_angle[2]
        return x,y,z,x_angle,y_angle,z_angle,pb_quat
    def compute_estimate_pos_of_object(self, particle_cloud):
        x_set = 0
        y_set = 0
        z_set = 0
        w_set = 0
        quaternions = []
        qws = []
        for index, particle in enumerate(particle_cloud):
            x_set = x_set + particle.x * particle.w
            y_set = y_set + particle.y * particle.w
            z_set = z_set + particle.z * particle.w
            q = p_visualisation.getQuaternionFromEuler([particle.x_angle, particle.y_angle, particle.z_angle])
            qws.append(particle.w)
            quaternions.append([q[0], q[1], q[2], q[3]])
            w_set = w_set + particle.w
        # q = average_quaternions(np.array(quaternions))
        q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
        x_angle, y_angle, z_angle = p_visualisation.getEulerFromQuaternion([q[0], q[1], q[2], q[3]])
        return x_set / w_set, y_set / w_set, z_set / w_set, x_angle, y_angle, z_angle
        
    
    def display_particle(self):
        for index, particle in enumerate(self.particle_cloud):
            visualize_particle_pos = [particle.x, particle.y, particle.z]
            visualize_particle_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            visualize_particle_orientation = p_visualisation.getQuaternionFromEuler(visualize_particle_angle)
            visualize_particle_Id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cube/cheezit_par_with_visual_small_PE_hor.urdf"),
                                                             visualize_particle_pos,
                                                             visualize_particle_orientation)
            self.particle_with_visual_id_collection.append(visualize_particle_Id)
    def display_particle_PM(self):
        for index, particle in enumerate(self.particle_cloud_PM):
            visualize_particle_pos = [particle.x, particle.y, particle.z]
            visualize_particle_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            visualize_particle_orientation = p_visualisation.getQuaternionFromEuler(visualize_particle_angle)
            visualize_particle_Id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cube/cheezit_par_with_visual_small_PM_hor.urdf"),
                                                             visualize_particle_pos,
                                                             visualize_particle_orientation)
            self.particle_with_visual_id_collection_PM.append(visualize_particle_Id)
        
    def initial_and_set_simulation_env(self,joint_of_robot):
        for index, particle in enumerate(self.particle_cloud):
            pybullet_simulation_env = bc.BulletClient(connection_mode=p.DIRECT)#DIRECT,GUI_SERVER
            self.pybullet_particle_env_collection.append(pybullet_simulation_env)
            pybullet_simulation_env.setTimeStep(change_sim_time)
            pybullet_simulation_env.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_simulation_env.setGravity(0,0,-9.81)
            fake_plane_id = pybullet_simulation_env.loadURDF("plane.urdf")
            sim_base_id = pybullet_simulation_env.loadURDF(
                os.path.expanduser("~/phd_project/object/cube/base_of_cheezit.urdf"),
                pw_T_base_pos,
                pw_T_base_ori,
                useFixedBase=1)
            fake_robot_start_pos = self.real_robot_start_pos
            fake_robot_start_orientation = self.real_robot_start_ori
            fake_robot_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/phd_project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                                             fake_robot_start_pos,
                                                             fake_robot_start_orientation,
                                                             useFixedBase=1)
            self.fake_robot_id_collection.append(fake_robot_id)
            
            #set joint of fake robot
            self.set_sim_robot_JointPosition(pybullet_simulation_env,fake_robot_id,joint_of_robot)
            
            particle_no_visual_start_pos = [particle.x, particle.y, particle.z]
            particle_no_visual_start_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            particle_no_visual_start_orientation = pybullet_simulation_env.getQuaternionFromEuler(particle_no_visual_start_angle)
            particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/phd_project/object/cube/cheezit_par_no_visual_small_hor.urdf"),
                                                                     particle_no_visual_start_pos,
                                                                     particle_no_visual_start_orientation)
                
            while True:
                pybullet_simulation_env.stepSimulation()
                flag = 0
                contacts = pybullet_simulation_env.getContactPoints(bodyA=fake_robot_id, bodyB=particle_no_visual_id)
                # pmin,pmax = pybullet_simulation_env.getAABB(particle_no_visual_id)
                # collide_ids = pybullet_simulation_env.getOverlappingObjects(pmin,pmax)
                # length = len(collide_ids)
                for contact in contacts:
                    contact_dis = contact[8]
                    if contact_dis < -0.001:
                        print("collision")
                        Px,Py,Pz,Px_angle,Py_angle,Pz_angle,P_quat = self.generate_random_pose(self.noise_object_pose,self.pw_T_object_ori_dope)
                        pybullet_simulation_env.resetBasePositionAndOrientation(particle_no_visual_id,
                                                                                [Px,Py,Pz],
                                                                                P_quat)
                        flag = 1
                        particle.x = Px
                        particle.y = Py
                        particle.z = Pz
                        particle.x_angle = Px_angle
                        particle.y_angle = Py_angle
                        particle.z_angle = Pz_angle
                        break
                if flag == 0:
                    break
            #pybullet_simulation_env.changeDynamics(particle_no_visual_id,-1,mass=3,lateralFriction = 0.75)
            self.particle_no_visual_id_collection.append(particle_no_visual_id)  
        obj_est_set = self.compute_estimate_pos_of_object(self.particle_cloud)
        return obj_est_set[0],obj_est_set[1],obj_est_set[2],obj_est_set[3],obj_est_set[4],obj_est_set[5] 
    def initial_and_set_simulation_env_PM(self,joint_of_robot):
        self.particle_cloud_PM = copy.deepcopy(self.particle_cloud)
        for index, particle in enumerate(self.particle_cloud_PM):
            pybullet_simulation_env = bc.BulletClient(connection_mode=p.DIRECT)
            self.pybullet_particle_env_collection_PM.append(pybullet_simulation_env)
            pybullet_simulation_env.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_simulation_env.setGravity(0,0,-9.81)
            fake_plane_id = pybullet_simulation_env.loadURDF("plane.urdf")

            particle_no_visual_start_pos = [particle.x, particle.y, particle.z]
            particle_no_visual_start_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            particle_no_visual_start_orientation = pybullet_simulation_env.getQuaternionFromEuler(particle_no_visual_start_angle)
            particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/phd_project/object/cube/cheezit_par_no_visual_small_hor.urdf"),
                                                                     particle_no_visual_start_pos,
                                                                     particle_no_visual_start_orientation)
            #pybullet_simulation_env.changeDynamics(particle_no_visual_id,-1,mass=3,lateralFriction = 0.7)
            self.particle_no_visual_id_collection_PM.append(particle_no_visual_id)
        obj_est_set_PM = self.compute_estimate_pos_of_object(self.particle_cloud_PM)
        return obj_est_set_PM[0],obj_est_set_PM[1],obj_est_set_PM[2],obj_est_set_PM[3],obj_est_set_PM[4],obj_est_set_PM[5] 
        
        
    def set_sim_robot_JointPosition(self,pybullet_simulation_env,robot, position):
        #position[7] = 0.039916139
        #position[8] = 0.039916139
        #num_joints = 7
        num_joints = 9
        for joint_index in range(num_joints): 
            if joint_index == 7 or joint_index == 8:
                pybullet_simulation_env.setJointMotorControl2(robot,
                                                              joint_index+2,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])                
            else:
                pybullet_simulation_env.setJointMotorControl2(robot,
                                                              joint_index,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])
        for i in range(240):                        
            pybullet_simulation_env.stepSimulation()
            #time.sleep(1./240.)
    def add_noise_to_init_par(self,current_pos,sigma_init):
        mean = current_pos
        sigma = sigma_init
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    def take_easy_gaussian_value(self,mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal

#Class of Particle Filter move
class PFMove():
    def __init__(self,robot_id=None,real_robot_id=None,object_id=None):
        # init internals
        self.particle_cloud = copy.deepcopy(initial_parameter.particle_cloud)
        self.particle_no_visual_id_collection = copy.deepcopy(initial_parameter.particle_no_visual_id_collection)
        self.pybullet_env_id_collection = copy.deepcopy(initial_parameter.pybullet_particle_env_collection)
        self.pybullet_sim_fake_robot_id_collection = copy.deepcopy(initial_parameter.fake_robot_id_collection)
        self.particle_with_visual_id_collection = copy.deepcopy(initial_parameter.particle_with_visual_id_collection)

        self.joint_num = 7

        self.object_estimate_pose_x = []
        self.object_estimate_pose_y = []
        self.object_real_____pose_x = []
        self.object_real_____pose_y = []
        
        self.noise_object_pos = []
        self.noise_object_ang = []
        self.noise_object_pose = []

    #new structure
    def real_robot_control_PE(self,opti_obj_pos_cur,opti_obj_ori_cur,real_robot_joint_pos,nois_obj_pos_cur,nois_obj_ang_cur):        
        #Cheat
        self.update_particle_filter_PE(self.pybullet_env_id_collection,  # simulation environment per particle
                                       self.pybullet_sim_fake_robot_id_collection,  # fake robot id per sim_env
                                       real_robot_joint_pos,  # execution actions of the fake robot
                                       opti_obj_pos_cur,
                                       opti_obj_ori_cur,
                                       nois_obj_pos_cur,
                                       nois_obj_ang_cur)
        # if Flag is False:
        #     return False
                
    def get_real_robot_joint(self, pybullet_env_id, real_robot_id):
        real_robot_joint_list = []
        for index in range(self.joint_num):
            real_robot_info = pybullet_env_id.getJointState(real_robot_id,index)[0]
            real_robot_joint_list.append(real_robot_info)
        return real_robot_joint_list
    
    def get_real_object_pos(self, object_id):
        object_info = p_visualisation.getBasePositionAndOrientation(object_id)
        return object_info[0]
    
    def get_observation(self, object_id):
        object_info = self.get_real_object_pos(object_id)
        return object_info
    
    #already changed need to del
    def real_robot_move(self, real_robot_id, u_i, step_size):
        for i in range(int(step_size*240)):
            p_visualisation.resetBaseVelocity(real_robot_id,linearVelocity = u_i)
            p_visualisation.stepSimulation()
            #time.sleep(1.0/240)
          
    def set_real_robot_JointPosition(self,pybullet_env,robot, position):
        #position[7] = 0.039916139
        #position[8] = 0.039916139
        #num_joints = 7
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                pybullet_env.setJointMotorControl2(robot,
                                                   joint_index+2,
                                                   pybullet_env.POSITION_CONTROL,
                                                   targetPosition=position[joint_index])                
            else:
                pybullet_env.setJointMotorControl2(robot,
                                                   joint_index,
                                                   pybullet_env.POSITION_CONTROL,
                                                   targetPosition=position[joint_index])  
    
    def compute_pos_err_bt_2_points(self,pos1,pos2):
        x_d = pos1[0]-pos2[0]
        y_d = pos1[1]-pos2[1]
        z_d = pos1[2]-pos2[2]
        distance = math.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)
        return distance           
    #executed_control 
    def update_particle_filter_PE(self, pybullet_sim_env, fake_robot_id, real_robot_joint_pos, opti_obj_pos_cur, opti_obj_ori_cur,nois_obj_pos_cur,nois_obj_ang_cur):
        global flag_record_dope
        global flag_record_PFPE
        global flag_record
        
        self.times = []
        t1 = time.time()
        self.motion_update_PE_parallelised(pybullet_sim_env, fake_robot_id, real_robot_joint_pos)
        t2 = time.time()
        self.times.append(t2-t1)
        # print("Motion model1 time consuming:",t2-t1)
        #self.display_particle_in_visual_model_PE(self.particle_cloud)
        #time.sleep(1)

        estimated_object_pos,estimated_object_ang = self.observation_update_PE(opti_obj_pos_cur,opti_obj_ori_cur,nois_obj_pos_cur,nois_obj_ang_cur)
        estimated_object_ori = p_visualisation.getQuaternionFromEuler(estimated_object_ang)
        nois_obj_ori_cur = p_visualisation.getQuaternionFromEuler(nois_obj_ang_cur)

        # print("observ model time consuming:", t3-t2)

        #if Flag is False:
        #    return False
        
        # print("Display particle")
        if visualisation_particle_flag == True:
            self.display_particle_in_visual_model_PE(self.particle_cloud)
        #self.draw_contrast_figure(estimated_object_pos,observation)
        #neettochange
        err_opti_dope_pos = compute_pos_err_bt_2_points(nois_obj_pos_cur,opti_obj_pos_cur)
        err_opti_dope_ang = compute_ang_err_bt_2_points(nois_obj_ori_cur,opti_obj_ori_cur)
        err_opti_dope_ang = angle_correction(err_opti_dope_ang)
        err_opti_PFPE_pos = compute_pos_err_bt_2_points(estimated_object_pos,opti_obj_pos_cur)
        err_opti_PFPE_ang = compute_ang_err_bt_2_points(estimated_object_ori,opti_obj_ori_cur)
        err_opti_PFPE_ang = angle_correction(err_opti_PFPE_ang)

        t_err_generate = time.time()

        t_before_record = time.time()
        boss_obse_err_pos_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
        boss_obse_err_ang_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
        boss_err_pos_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
        boss_err_ang_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
        flag_record = flag_record + 1
        flag_record_dope = flag_record_dope + 1
        boss_PFPE_err_pos_df.loc[flag_record_PFPE] = [flag_record_PFPE, t_before_record - t_begin, err_opti_PFPE_pos, 'PFPE']
        boss_PFPE_err_ang_df.loc[flag_record_PFPE] = [flag_record_PFPE, t_before_record - t_begin, err_opti_PFPE_ang, 'PFPE']
        boss_err_pos_df.loc[flag_record] = [flag_record_PFPE, t_before_record - t_begin, err_opti_PFPE_pos, 'PFPE']
        boss_err_ang_df.loc[flag_record] = [flag_record_PFPE, t_before_record - t_begin, err_opti_PFPE_ang, 'PFPE']
        flag_record = flag_record + 1
        flag_record_PFPE = flag_record_PFPE + 1
        
        
        # print debug info of all particles here
        #input('hit enter to continue')
        return
    
    def update_partcile_cloud_pose_PE(self, index, x, y, z, x_angle, y_angle, z_angle):
        self.particle_cloud[index].x = x
        self.particle_cloud[index].y = y
        self.particle_cloud[index].z = z
        self.particle_cloud[index].x_angle = x_angle
        self.particle_cloud[index].y_angle = y_angle
        self.particle_cloud[index].z_angle = z_angle     
              
    def motion_update_PE(self, pybullet_sim_env, fake_robot_id, real_robot_joint_pos):
        start = time.time()
        for index, pybullet_env in enumerate(pybullet_sim_env):
            pipe_parent, pipe_child = multiprocessing.Pipe()
            self.function_to_parallelise(index, pybullet_env,fake_robot_id, real_robot_joint_pos, pipe_child)
            x, y, z, x_angle, y_angle, z_angle = pipe_parent.recv()
            self.update_partcile_cloud_pose_PE(index, x, y, z, x_angle, y_angle, z_angle)
        end = time.time()
        # print(end - start)

        
    def motion_update_PE_parallelised(self,pybullet_sim_env, fake_robot_id, real_robot_joint_pos):
        threads = []
        
        start = time.time()
        for index, pybullet_env in enumerate(pybullet_sim_env):
            thread = threading.Thread(target=self.function_to_parallelise, args=(index, pybullet_env, fake_robot_id, real_robot_joint_pos))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
             
        end = time.time()
        # print(end - start)
    
        
    def function_to_parallelise(self, index, pybullet_env,fake_robot_id, real_robot_joint_pos):
        self.change_obj_parameters(pybullet_env,initial_parameter.particle_no_visual_id_collection[index])
        #execute the control

        pf_update_interval_in_sim = boss_pf_update_interval_in_real / change_sim_time
        #boss_pf_update_interval_in_real
        for time_index in range(int(pf_update_interval_in_sim)):
            self.set_real_robot_JointPosition(pybullet_env,fake_robot_id[index],real_robot_joint_pos)
            pybullet_env.stepSimulation()

        ### ori: x,y,z,w
        sim_par_cur_pos,sim_par_cur_ori = self.get_item_pos(pybullet_env,initial_parameter.particle_no_visual_id_collection[index])
        sim_par_cur_ang = p_visualisation.getEulerFromQuaternion(sim_par_cur_ori)
        #add noise on pos of each particle
        normal_x = self.add_noise_2_par(sim_par_cur_pos[0])
        normal_y = self.add_noise_2_par(sim_par_cur_pos[1])
        normal_z = self.add_noise_2_par(sim_par_cur_pos[2])
        #add noise on ang of each particle
        quat = copy.deepcopy(sim_par_cur_ori)#x,y,z,w
        quat_QuatStyle = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3])#w,x,y,z
        random_dir = random.uniform(0, 2*math.pi)
        z_axis = random.uniform(-1,1)
        x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
        y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
        angle_noise = self.add_noise_2_ang(0)
        w_quat = math.cos(angle_noise/2.0)
        x_quat = math.sin(angle_noise/2.0) * x_axis
        y_quat = math.sin(angle_noise/2.0) * y_axis
        z_quat = math.sin(angle_noise/2.0) * z_axis
        ###nois_quat(w,x,y,z); new_quat(w,x,y,z)
        nois_quat = Quaternion(x=x_quat,y=y_quat,z=z_quat,w=w_quat)
        new_quat = nois_quat * quat_QuatStyle
        ###pb_quat(x,y,z,w)
        pb_quat = [new_quat[1],new_quat[2],new_quat[3],new_quat[0]]
        new_angle = p_visualisation.getEulerFromQuaternion(pb_quat)
        x_angle = new_angle[0]
        y_angle = new_angle[1]
        z_angle = new_angle[2]
        #x_angle = sim_par_cur_ang[0]
        #y_angle = sim_par_cur_ang[1]
        #z_angle = sim_par_cur_ang[2]
            
        #self.particle_cloud[index].x = sim_par_cur_pos[0]
        #self.particle_cloud[index].y = sim_par_cur_pos[1]
        #self.particle_cloud[index].z = sim_par_cur_pos[2]
        #self.particle_cloud[index].x_angle = sim_par_cur_angle[0]
        #self.particle_cloud[index].y_angle = sim_par_cur_angle[1]
        #self.particle_cloud[index].z_angle = sim_par_cur_angle[2]
        self.update_partcile_cloud_pose_PE(index, normal_x, normal_y, normal_z, x_angle, y_angle, z_angle)  
        #self.update_poses[index] = (normal_x, normal_y, normal_z, x_angle, y_angle, z_angle)
        # pipe.send()

        
    def observation_update_PE(self, opti_obj_pos_cur,opti_obj_ori_cur,nois_obj_pos_cur,nois_obj_ang_cur):
        nois_obj_x = nois_obj_pos_cur[0]
        nois_obj_y = nois_obj_pos_cur[1]
        nois_obj_z = nois_obj_pos_cur[2]
        nois_obj_x_ang = nois_obj_ang_cur[0]
        nois_obj_y_ang = nois_obj_ang_cur[1]
        nois_obj_z_ang = nois_obj_ang_cur[2]
        nois_obj_pose = [nois_obj_x,nois_obj_y,nois_obj_z,nois_obj_x_ang,nois_obj_y_ang,nois_obj_z_ang]

        self.noise_object_pos = [nois_obj_x,nois_obj_y,nois_obj_z]
        self.noise_object_ang = [nois_obj_x_ang,nois_obj_y_ang,nois_obj_z_ang]
        self.noise_object_pose = [nois_obj_x,nois_obj_y,nois_obj_z,nois_obj_x_ang,nois_obj_y_ang,nois_obj_z_ang]
        
        for index,particle in enumerate(self.particle_cloud):
            
            particle_x = particle.x
            particle_y = particle.y
            particle_z = particle.z
            particle_x_ang = particle.x_angle
            particle_y_ang = particle.y_angle
            particle_z_ang = particle.z_angle
            
            nois_obj_pos = [nois_obj_pose[0],nois_obj_pose[1],nois_obj_pose[2]]
            nois_obj_pos_x = nois_obj_pos[0]
            nois_obj_pos_y = nois_obj_pos[1]
            nois_obj_pos_z = nois_obj_pos[2]
            mean = 0
            dis_x = abs(particle_x-nois_obj_pos_x)
            dis_y = abs(particle_y-nois_obj_pos_y)
            dis_z = abs(particle_z-nois_obj_pos_z)
            #sigma_x = boss_sigma_obs_x
            #sigma_y = boss_sigma_obs_y
            #sigma_z = boss_sigma_obs_z
            #weight_x = self.normal_distribution(dis_x, mean, sigma_x)
            #weight_y = self.normal_distribution(dis_y, mean, sigma_y)
            #weight_z = self.normal_distribution(dis_z, mean, sigma_z)
            #weight_pos = math.sqrt(weight_x ** 2 + weight_y ** 2 + weight_z ** 2)
            #weight_pos = weight_x + weight_y + weight_z
            dis_xyz = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
            weight_xyz = self.normal_distribution(dis_xyz, mean, boss_sigma_obs_pos)
            
            #pybullet x,y,z,w
            nois_obj_ang = [nois_obj_pose[3],nois_obj_pose[4],nois_obj_pose[5]]
            par_ang = [particle_x_ang,particle_y_ang,particle_z_ang]
            nois_obj_ori = p_visualisation.getQuaternionFromEuler(nois_obj_ang)
            par_ori = p_visualisation.getQuaternionFromEuler(par_ang)
            #w,x,y,z
            nois_obj_quat = Quaternion(x=nois_obj_ori[0],y=nois_obj_ori[1],z=nois_obj_ori[2],w=nois_obj_ori[3])
            par_quat = Quaternion(x=par_ori[0],y=par_ori[1],z=par_ori[2],w=par_ori[3])
            err_bt_par_dope = par_quat * nois_obj_quat.inverse
            cos_theta_over_2 = err_bt_par_dope.w
            sin_theta_over_2 = math.sqrt(err_bt_par_dope.x ** 2 + err_bt_par_dope.y ** 2 + err_bt_par_dope.z ** 2)
            theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
            theta = theta_over_2 * 2
            weight_ang = self.normal_distribution(theta, mean, boss_sigma_obs_ang)
            weight = weight_xyz * weight_ang
            particle.w = weight

        Flag = self.normalize_particles()
        #if Flag is False:
        #    return False
        self.resample_particles()
        self.set_paticle_in_each_sim_env()
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection):
            part_pos = pybullet_env.getBasePositionAndOrientation(self.particle_no_visual_id_collection[index])
            #print("particle:",part_pos[0][0],part_pos[0][1],part_pos[0][2])
        object_estimate_pose = self.compute_estimate_pos_of_object(self.particle_cloud)
        estimated_object_pos = [object_estimate_pose[0],object_estimate_pose[1],object_estimate_pose[2]]
        estimated_object_ang = [object_estimate_pose[3],object_estimate_pose[4],object_estimate_pose[5]]
        if visualisation_flag == True:
            self.display_estimated_robot_in_visual_model(estimated_object_pos,estimated_object_ang)
        return estimated_object_pos,estimated_object_ang

    def compare_rob_joint(self,real_rob_joint_list_cur,real_robot_joint_pos):
        for i in range(self.joint_num):
            diff = 10
            diff = abs(real_rob_joint_list_cur[i] - real_robot_joint_pos[i])
            if diff > 0.005:
                return 1
        return 0
    
    def change_obj_parameters(self,pybullet_env,par_id):
        mean_mass = 0.380
        mean_friction = 0.5
        mass_a = random.uniform(0.330,0.430)
        friction_b = random.uniform(0.4,0.6)
        # mass_a = self.take_easy_gaussian_value(mean_mass, 0.05)
        # friction_b = self.take_easy_gaussian_value(mean_friction, 0.1)
        #mass_a = 0.351
        #fricton_b = 0.30
        pybullet_env.changeDynamics(par_id, -1, mass = mass_a, lateralFriction = friction_b)
    
    def get_item_pos(self,pybullet_env,item_id):
        item_info = pybullet_env.getBasePositionAndOrientation(item_id)
        return item_info[0],item_info[1]
    
    def add_noise_2_par(self,current_pos):
        mean = current_pos
        sigma = boss_sigma_obs_x
        sigma = 0.01
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def add_noise_2_ang(self,cur_angle):
        mean = cur_angle
        sigma = boss_sigma_obs_ang
        sigma = 0.1
        new_angle_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_angle_is_added_noise
    
    def take_easy_gaussian_value(self,mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal
    
    def normal_distribution(self, x, mean, sigma):
        return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)
    
    def normalize_particles(self):
        flag_1 = 0
        tot_weight = sum([particle.w for particle in self.particle_cloud])
        if tot_weight == 0:
            # print("Error!,PFPE particles total weight is 0")
            tot_weight = 1
            flag_1 = 1
        for particle in self.particle_cloud:
            if flag_1 == 0:
                particle_w = particle.w/tot_weight
                particle.w = particle_w
            else:
                particle.w = 1/particle_num

        #tot_weight_test = sum([particle.w for particle in self.particle_cloud])
        #print("tot_weight_test:",tot_weight_test)

    def resample_particles(self):
        particles_w = []
        newParticles = [] 
        n_particle = len(self.particle_cloud)
        for particle in self.particle_cloud:
            particles_w.append(particle.w)
        particle_array= np.random.choice(a = n_particle, size = n_particle, replace=True, p= particles_w)
        particle_array_list = list(particle_array)
        for index,i in enumerate(particle_array_list):
            particle = Particle(self.particle_cloud[i].x,
                                self.particle_cloud[i].y,
                                self.particle_cloud[i].z,
                                self.particle_cloud[i].x_angle,
                                self.particle_cloud[i].y_angle,
                                self.particle_cloud[i].z_angle,
                                self.particle_cloud[i].w,index)
            newParticles.append(particle)
        self.particle_cloud = copy.deepcopy(newParticles)
        
    def set_paticle_in_each_sim_env(self):
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection):
            visual_particle_pos = [self.particle_cloud[index].x, self.particle_cloud[index].y, self.particle_cloud[index].z]
            visual_particle_angle = [self.particle_cloud[index].x_angle, self.particle_cloud[index].y_angle, self.particle_cloud[index].z_angle]
            visual_particle_orientation = pybullet_env.getQuaternionFromEuler(visual_particle_angle)
            pybullet_env.resetBasePositionAndOrientation(self.particle_no_visual_id_collection[index],
                                                         visual_particle_pos,
                                                         visual_particle_orientation)
        return
        
    def display_particle_in_visual_model_PE(self, particle_cloud):
        for index, particle in enumerate(particle_cloud):
            visual_particle_pos = [particle.x, particle.y, particle.z]
            visual_particle_ang = [particle.x_angle, particle.y_angle, particle.z_angle]
            visual_particle_orientation = p_visualisation.getQuaternionFromEuler(visual_particle_ang)
            p_visualisation.resetBasePositionAndOrientation(self.particle_with_visual_id_collection[index],
                                                            visual_particle_pos,
                                                            visual_particle_orientation)
            #print("visual_particle_pos:",visual_particle_pos)
            #particle_pos = self.get_item_pos(pybullet_env[index],initial_parameter.particle_no_visual_id_collection[index])


    def display_estimated_robot_in_visual_model(self, observation,estimated_angle):
        esti_obj_pos = observation
        esti_obj_ori = p_visualisation.getQuaternionFromEuler(estimated_angle)
        p_visualisation.resetBasePositionAndOrientation(estimated_object_id,
                                                        esti_obj_pos,
                                                        esti_obj_ori)    

    def draw_contrast_figure(self,estimated_object_pos,observation):
        # print("Begin to draw contrast figure!")
        self.object_estimate_pose_x.append(estimated_object_pos[0])
        self.object_estimate_pose_y.append(estimated_object_pos[1])
        self.object_real_____pose_x.append(observation[0])
        self.object_real_____pose_y.append(observation[1])
        plt.plot(self.object_estimate_pose_x,self.object_estimate_pose_y,"x-",label="Estimated Object Pose")
        plt.plot(self.object_real_____pose_x,self.object_real_____pose_y,"*-",label="Real Object Pose")
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.legend()
        plt.savefig('chart.png')
        plt.close()
        return

    def compute_estimate_pos_of_object(self, particle_cloud):
        x_set = 0
        y_set = 0
        z_set = 0
        w_set = 0
        quaternions = []
        qws = []
        for index, particle in enumerate(particle_cloud):
            x_set = x_set + particle.x * particle.w
            y_set = y_set + particle.y * particle.w
            z_set = z_set + particle.z * particle.w
            q = p_visualisation.getQuaternionFromEuler([particle.x_angle, particle.y_angle, particle.z_angle])
            qws.append(particle.w)
            quaternions.append([q[0], q[1], q[2], q[3]])
            w_set = w_set + particle.w
        # q = average_quaternions(np.array(quaternions))
        q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
        x_angle, y_angle, z_angle = p_visualisation.getEulerFromQuaternion([q[0], q[1], q[2], q[3]])
        return x_set / w_set, y_set / w_set, z_set / w_set, x_angle, y_angle, z_angle
    
    def compute_transformation_matrix(self, init_robot_pos,init_robot_ori,init_object_pos,init_object_ori):
        robot_transformation_matrix = transformations.quaternion_matrix(init_robot_ori)
        ow_T_robot = self.rotation_4_4_to_transformation_4_4(robot_transformation_matrix,init_robot_pos)
        # print("ow_T_robot:",ow_T_robot)
        object_transformation_matrix = transformations.quaternion_matrix(init_object_ori)
        ow_T_object = self.rotation_4_4_to_transformation_4_4(object_transformation_matrix,init_object_pos)
        # print("ow_T_object:",ow_T_object)
        robot_T_ow = np.linalg.inv(ow_T_robot)
        robot_T_object = np.dot(robot_T_ow,ow_T_object)
        # print("robot_T_object:")
        # print(robot_T_object)
        return robot_T_object
    
    def rotation_4_4_to_transformation_4_4(self, rotation_4_4,pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4



class PFMovePM():
    def __init__(self,robot_id=None,real_robot_id=None,object_id=None):
        # init internals   
        self.particle_cloud_PM = copy.deepcopy(initial_parameter.particle_cloud_PM)
        self.particle_no_visual_id_collection_PM = copy.deepcopy(initial_parameter.particle_no_visual_id_collection_PM)
        self.pybullet_env_id_collection_PM = copy.deepcopy(initial_parameter.pybullet_particle_env_collection_PM)
        self.particle_with_visual_id_collection_PM = copy.deepcopy(initial_parameter.particle_with_visual_id_collection_PM)

        self.object_estimate_pose_x = []
        self.object_estimate_pose_y = []
        self.object_real_____pose_x = []
        self.object_real_____pose_y = []
        
        self.noise_object_pos = []
        self.noise_object_ang = []
        self.noise_object_pose = []
        
        

    #new structure
    def real_robot_control_PM(self,opti_obj_pos_cur,opti_obj_ori_cur,nois_obj_pos_cur,nois_obj_ang_cur):      
        #Cheat
        self.update_particle_filter_PM(opti_obj_pos_cur,
                                       opti_obj_ori_cur,
                                       nois_obj_pos_cur,
                                       nois_obj_ang_cur)
        # if Flag is False:
        #     return False
        
    def compute_pos_err_bt_2_points(self,pos1,pos2):
        x_d = pos1[0]-pos2[0]
        y_d = pos1[1]-pos2[1]
        z_d = pos1[2]-pos2[2]
        distance = math.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)
        return distance      
      
    #executed_control 
    def update_particle_filter_PM(self, opti_obj_pos_cur, opti_obj_ori_cur,nois_obj_pos_cur,nois_obj_ang_cur):
        global flag_record_PFPM
        global flag_record
        t1 = time.time()
        self.motion_update_PM(nois_obj_ang_cur)
        t2 = time.time()
        estimated_object_pose_PM= self.observation_update_PM(opti_obj_pos_cur,opti_obj_ori_cur,nois_obj_pos_cur,nois_obj_ang_cur)
        estimated_object_pos_PM = [estimated_object_pose_PM[0],estimated_object_pose_PM[1],estimated_object_pose_PM[2]]
        estimated_object_ang_PM = [estimated_object_pose_PM[3],estimated_object_pose_PM[4],estimated_object_pose_PM[5]]
        boss_est_pose_PFPM.append(estimated_object_pose_PM)
        t3 = time.time()
        # print("motion model2 time consuming:",t2-t1)
        # print("observ model time consuming:",t3-t2)
        #self.draw_contrast_figure(estimated_object_pos,observation)
        estimated_object_ori_PM = p_visualisation.getQuaternionFromEuler(estimated_object_ang_PM)
        if visualisation_particle_flag == True:
            self.display_particle_in_visual_model_PM(self.particle_cloud_PM)
        err_opti_PFPM_pos = compute_pos_err_bt_2_points(estimated_object_pos_PM,opti_obj_pos_cur)
        err_opti_PFPM_ang = compute_ang_err_bt_2_points(estimated_object_ori_PM,opti_obj_ori_cur)
        err_opti_PFPM_ang = angle_correction(err_opti_PFPM_ang)
        t_err_generate = time.time()

        t_before_record = time.time()
        boss_PFPM_err_pos_df.loc[flag_record_PFPM] = [flag_record_PFPM, t_before_record - t_begin, err_opti_PFPM_pos, 'PFPM']
        boss_PFPM_err_ang_df.loc[flag_record_PFPM] = [flag_record_PFPM, t_before_record - t_begin, err_opti_PFPM_ang, 'PFPM']
        boss_err_pos_df.loc[flag_record] = [flag_record_PFPM, t_before_record - t_begin, err_opti_PFPM_pos, 'PFPM']
        boss_err_ang_df.loc[flag_record] = [flag_record_PFPM, t_before_record - t_begin, err_opti_PFPM_ang, 'PFPM']
        flag_record = flag_record + 1
        flag_record_PFPM = flag_record_PFPM + 1
        # print debug info of all particles here
        #input('hit enter to continue')
        return

    def motion_update_PM(self, nois_obj_ang_cur):
        if flag_update_num_PM < 2:
            length = len(boss_obs_pose_PFPM)
            obs_curr_pose = copy.deepcopy(boss_obs_pose_PFPM[length-1])
            obs_last_pose = copy.deepcopy(boss_obs_pose_PFPM[length-2])
            obs_curr_pos = [obs_curr_pose[0],obs_curr_pose[1],obs_curr_pose[2]]
            obs_curr_ang = [obs_curr_pose[3],obs_curr_pose[4],obs_curr_pose[5]]
            obs_curr_ori = p_visualisation.getQuaternionFromEuler(obs_curr_ang)
            obs_last_pos = [obs_last_pose[0],obs_last_pose[1],obs_last_pose[2]]
            obs_last_ang = [obs_last_pose[3],obs_last_pose[4],obs_last_pose[5]]
            obs_last_ori = p_visualisation.getQuaternionFromEuler(obs_last_ang)
            obsO_T_obsN = self.compute_transformation_matrix(obs_last_pos, obs_last_ori, obs_curr_pos, obs_curr_ori)
            parO_T_parN = copy.deepcopy(obsO_T_obsN)
            self.update_particle_in_motion_model_PM(parO_T_parN, nois_obj_ang_cur)
        else:
            length = len(boss_est_pose_PFPM)
            est_curr_pose = copy.deepcopy(boss_est_pose_PFPM[length-1])
            est_last_pose = copy.deepcopy(boss_est_pose_PFPM[length-2])
            est_curr_pos = [est_curr_pose[0],est_curr_pose[1],est_curr_pose[2]]
            est_curr_ang = [est_curr_pose[3],est_curr_pose[4],est_curr_pose[5]]
            est_curr_ori = p_visualisation.getQuaternionFromEuler(est_curr_ang)
            est_last_pos = [est_last_pose[0],est_last_pose[1],est_last_pose[2]]
            est_last_ang = [est_last_pose[3],est_last_pose[4],est_last_pose[5]]
            est_last_ori = p_visualisation.getQuaternionFromEuler(est_last_ang)
            estO_T_estN = self.compute_transformation_matrix(est_last_pos, est_last_ori, est_curr_pos, est_curr_ori)
            parO_T_parN = copy.deepcopy(estO_T_estN)
            self.update_particle_in_motion_model_PM(parO_T_parN, nois_obj_ang_cur)
            
        return

    def observation_update_PM(self,opti_obj_pos_cur,opti_obj_ori_cur,nois_obj_pos_cur,nois_obj_ang_cur):
        nois_obj_x = nois_obj_pos_cur[0]
        nois_obj_y = nois_obj_pos_cur[1]
        nois_obj_z = nois_obj_pos_cur[2]
        nois_obj_x_ang = nois_obj_ang_cur[0]
        nois_obj_y_ang = nois_obj_ang_cur[1]
        nois_obj_z_ang = nois_obj_ang_cur[2]
        nois_obj_pose = [nois_obj_x,nois_obj_y,nois_obj_z,nois_obj_x_ang,nois_obj_y_ang,nois_obj_z_ang]

        self.noise_object_pos = [nois_obj_x,nois_obj_y,nois_obj_z]
        self.noise_object_ang = [nois_obj_x_ang,nois_obj_y_ang,nois_obj_z_ang]
        self.noise_object_pose = [nois_obj_x,nois_obj_y,nois_obj_z,nois_obj_x_ang,nois_obj_y_ang,nois_obj_z_ang]
        
        for index,particle in enumerate(self.particle_cloud_PM):
            particle_x = particle.x
            particle_y = particle.y
            particle_z = particle.z
            particle_x_ang = particle.x_angle
            particle_y_ang = particle.y_angle
            particle_z_ang = particle.z_angle
            
            nois_obj_pos = [nois_obj_pose[0],nois_obj_pose[1],nois_obj_pose[2]]
            nois_obj_pos_x = nois_obj_pos[0]
            nois_obj_pos_y = nois_obj_pos[1]
            nois_obj_pos_z = nois_obj_pos[2]
            mean = 0
            dis_x = abs(particle_x-nois_obj_pos_x)
            dis_y = abs(particle_y-nois_obj_pos_y)
            dis_z = abs(particle_z-nois_obj_pos_z)
            sigma_x = boss_sigma_obs_x
            sigma_y = boss_sigma_obs_y
            sigma_z = boss_sigma_obs_z
            weight_x = self.normal_distribution(dis_x, mean, sigma_x)
            weight_y = self.normal_distribution(dis_y, mean, sigma_y)
            weight_z = self.normal_distribution(dis_z, mean, sigma_z)
            weight_pos = weight_x + weight_y + weight_z
            dis_xyz = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
            weight_xyz = self.normal_distribution(dis_xyz, mean, boss_sigma_obs_pos)
            
            #pybullet x,y,z,w
            nois_obj_ang = [nois_obj_pose[3],nois_obj_pose[4],nois_obj_pose[5]]
            par_ang = [particle_x_ang,particle_y_ang,particle_z_ang]
            nois_obj_ori = p_visualisation.getQuaternionFromEuler(nois_obj_ang)
            par_ori = p_visualisation.getQuaternionFromEuler(par_ang)
            #w,x,y,z
            nois_obj_quat = Quaternion(x=nois_obj_ori[0],y=nois_obj_ori[1],z=nois_obj_ori[2],w=nois_obj_ori[3])
            par_quat = Quaternion(x=par_ori[0],y=par_ori[1],z=par_ori[2],w=par_ori[3])
            err_bt_par_dope = par_quat * nois_obj_quat.inverse
            cos_theta_over_2 = err_bt_par_dope.w
            sin_theta_over_2 = math.sqrt(err_bt_par_dope.x ** 2 + err_bt_par_dope.y ** 2 + err_bt_par_dope.z ** 2)
            theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
            theta = theta_over_2 * 2
            weight_ang = self.normal_distribution(theta, mean, boss_sigma_obs_ang)
            weight = weight_xyz * weight_ang *10000
            particle.w = weight
            
        Flag = self.normalize_particles_PM()
        #if Flag is False:
        #    return False
        
        self.resample_particles_PM()
        self.set_paticle_in_each_sim_env_PM()

        object_estimate_pose = self.compute_estimate_pos_of_object(self.particle_cloud_PM)
        estimated_object_pos = [object_estimate_pose[0],object_estimate_pose[1],object_estimate_pose[2]]
        estimated_object_ang = [object_estimate_pose[3],object_estimate_pose[4],object_estimate_pose[5]]
        if visualisation_flag == True:
            self.display_estimated_robot_in_visual_model(estimated_object_pos,estimated_object_ang)
        return object_estimate_pose

    def update_particle_in_motion_model_PM(self, parO_T_parN, nois_obj_ang_cur):
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection_PM):
            # print("=================================================")
            pw_T_parO_x = self.particle_cloud_PM[index].x
            pw_T_parO_y = self.particle_cloud_PM[index].y
            pw_T_parO_z = self.particle_cloud_PM[index].z
            pw_T_parO_pos = [pw_T_parO_x, pw_T_parO_y, pw_T_parO_z]
            pw_T_parO_x_ang = self.particle_cloud_PM[index].x_angle
            pw_T_parO_y_ang = self.particle_cloud_PM[index].y_angle
            pw_T_parO_z_ang = self.particle_cloud_PM[index].z_angle
            pw_T_parO_ang = [pw_T_parO_x_ang, pw_T_parO_y_ang, pw_T_parO_z_ang]
            pw_T_parO_ori = pybullet_env.getQuaternionFromEuler(pw_T_parO_ang)

            pw_T_parO_3_3 = transformations.quaternion_matrix(pw_T_parO_ori)
            pw_T_parO = self.rotation_4_4_to_transformation_4_4(pw_T_parO_3_3, pw_T_parO_pos)
            pw_T_parN = np.dot(pw_T_parO, parO_T_parN)
            pw_T_parN_pos = [pw_T_parN[0][3], pw_T_parN[1][3], pw_T_parN[2][3]]
            pw_T_parN_ori = transformations.quaternion_from_matrix(pw_T_parN)
            pw_T_parN_ang = pybullet_env.getEulerFromQuaternion(pw_T_parN_ori)
            # test
            par_curr_pos = copy.deepcopy(pw_T_parN_pos)
            par_curr_ori = copy.deepcopy(pw_T_parN_ori)
            par_last_pos = copy.deepcopy(pw_T_parO_pos)
            par_last_ori = copy.deepcopy(pw_T_parO_ori)
            test_parO_T_parN = self.compute_transformation_matrix(par_last_pos, par_last_ori, par_curr_pos,
                                                                  par_curr_ori)
            # print("test_parO_T_parN"+str(index+1)+":")
            # print(test_parO_T_parN)

            # add noise on particle filter
            normal_x = self.add_noise_2_par(pw_T_parN_pos[0])
            normal_y = self.add_noise_2_par(pw_T_parN_pos[1])
            normal_z = self.add_noise_2_par(pw_T_parN_pos[2])
            # normal_x = pw_T_parN_pos[0]
            # normal_y = pw_T_parN_pos[1]
            # normal_z = pw_T_parN_pos[2]
            
            nois_obj_ori_cur = pybullet_env.getQuaternionFromEuler(nois_obj_ang_cur)
            quat = copy.deepcopy(pw_T_parN_ori)  # x,y,z,w
            quat_QuatStyle = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])  # w,x,y,z
            random_dir = random.uniform(0, 2 * math.pi)
            z_axis = random.uniform(-1, 1)
            x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
            y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
            angle_noise = self.add_noise_2_ang(0)
            w_quat = math.cos(angle_noise / 2.0)
            x_quat = math.sin(angle_noise / 2.0) * x_axis
            y_quat = math.sin(angle_noise / 2.0) * y_axis
            z_quat = math.sin(angle_noise / 2.0) * z_axis
            ###nois_quat(w,x,y,z); new_quat(w,x,y,z)
            nois_quat = Quaternion(x=x_quat, y=y_quat, z=z_quat, w=w_quat)
            new_quat = nois_quat * quat_QuatStyle
            ###pb_quat(x,y,z,w)
            pb_quat = [new_quat[1], new_quat[2], new_quat[3], new_quat[0]]
            new_angle = p_visualisation.getEulerFromQuaternion(pb_quat)

            x_angle = new_angle[0]
            y_angle = new_angle[1]
            z_angle = new_angle[2]
            # x_angle = pw_T_parN_ang[0]
            # y_angle = pw_T_parN_ang[1]
            # z_angle = pw_T_parN_ang[2]

            # self.particle_cloud_PM[index].x = pw_T_parN_pos[0]
            # self.particle_cloud_PM[index].y = pw_T_parN_pos[1]
            # self.particle_cloud_PM[index].z = pw_T_parN_pos[2]
            # self.particle_cloud_PM[index].x_angle = pw_T_parN_ang[0]
            # self.particle_cloud_PM[index].y_angle = pw_T_parN_ang[1]
            # self.particle_cloud_PM[index].z_angle = pw_T_parN_ang[2]
            self.particle_cloud_PM[index].x = normal_x
            self.particle_cloud_PM[index].y = normal_y
            self.particle_cloud_PM[index].z = normal_z
            self.particle_cloud_PM[index].x_angle = x_angle
            self.particle_cloud_PM[index].y_angle = y_angle
            self.particle_cloud_PM[index].z_angle = z_angle
                 
    
    def get_item_pos(self,pybullet_env,item_id):
        item_info = pybullet_env.getBasePositionAndOrientation(item_id)
        return item_info[0],item_info[1]
    
    def add_noise_2_par(self,current_pos):
        mean = current_pos
        sigma = boss_sigma_obs_x
        sigma = 0.01
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def add_noise_2_ang(self,cur_angle):
        mean = cur_angle
        sigma = boss_sigma_obs_ang
        sigma = 0.1
        new_angle_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_angle_is_added_noise
    
    def take_easy_gaussian_value(self,mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal
    
    def normal_distribution(self, x, mean, sigma):
        return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)

    def normalize_particles_PM(self):
        flag_1 = 0
        tot_weight = sum([particle.w for particle in self.particle_cloud_PM])
        if tot_weight == 0:
            # print("Error!,PFPM particles total weight is 0")
            tot_weight = 1
            flag_1 = 1
        for particle in self.particle_cloud_PM:
            if flag_1 == 0:
                particle_w = particle.w/tot_weight
                particle.w = particle_w
            else:
                particle.w = 1/particle_num

    def resample_particles_PM(self):
        particles_w = []
        newParticles = [] 
        n_particle = len(self.particle_cloud_PM)
        for particle in self.particle_cloud_PM:
            particles_w.append(particle.w)
        particle_array= np.random.choice(a = n_particle, size = n_particle, replace=True, p= particles_w)
        particle_array_list = list(particle_array)
        for index,i in enumerate(particle_array_list):
            particle = Particle(self.particle_cloud_PM[i].x,
                                self.particle_cloud_PM[i].y,
                                self.particle_cloud_PM[i].z,
                                self.particle_cloud_PM[i].x_angle,
                                self.particle_cloud_PM[i].y_angle,
                                self.particle_cloud_PM[i].z_angle,
                                self.particle_cloud_PM[i].w,index)
            newParticles.append(particle)
        self.particle_cloud_PM = copy.deepcopy(newParticles)
        
    def set_paticle_in_each_sim_env_PM(self):
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection_PM):
            visual_particle_pos = [self.particle_cloud_PM[index].x, self.particle_cloud_PM[index].y, self.particle_cloud_PM[index].z]
            visual_particle_angle = [self.particle_cloud_PM[index].x_angle, self.particle_cloud_PM[index].y_angle, self.particle_cloud_PM[index].z_angle]
            visual_particle_orientation = pybullet_env.getQuaternionFromEuler(visual_particle_angle)
            pybullet_env.resetBasePositionAndOrientation(self.particle_no_visual_id_collection_PM[index],
                                                         visual_particle_pos,
                                                         visual_particle_orientation)
        return        

    def display_particle_in_visual_model_PM(self, particle_cloud):
        for index, particle in enumerate(particle_cloud):
            visual_particle_pos = [particle.x, particle.y, particle.z]
            visual_particle_ang = [particle.x_angle, particle.y_angle, particle.z_angle]
            visual_particle_orientation = p_visualisation.getQuaternionFromEuler(visual_particle_ang)
            p_visualisation.resetBasePositionAndOrientation(self.particle_with_visual_id_collection_PM[index],
                                                            visual_particle_pos,
                                                            visual_particle_orientation)

    def display_estimated_robot_in_visual_model(self, observation,estimated_angle):
        esti_obj_pos = observation
        esti_obj_ori = p_visualisation.getQuaternionFromEuler(estimated_angle)
        p_visualisation.resetBasePositionAndOrientation(estimated_object_id_PM,
                                                        esti_obj_pos,
                                                        esti_obj_ori)    

    def draw_contrast_figure(self,estimated_object_pos,observation):
        # print("Begin to draw contrast figure!")
        self.object_estimate_pose_x.append(estimated_object_pos[0])
        self.object_estimate_pose_y.append(estimated_object_pos[1])
        self.object_real_____pose_x.append(observation[0])
        self.object_real_____pose_y.append(observation[1])
        plt.plot(self.object_estimate_pose_x,self.object_estimate_pose_y,"x-",label="Estimated Object Pose")
        plt.plot(self.object_real_____pose_x,self.object_real_____pose_y,"*-",label="Real Object Pose")
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.legend()
        plt.savefig('chart.png')
        plt.close()
        return

    def compute_estimate_pos_of_object(self, particle_cloud):
        x_set = 0
        y_set = 0
        z_set = 0
        w_set = 0
        quaternions = []
        qws = []
        for index, particle in enumerate(particle_cloud):
            x_set = x_set + particle.x * particle.w
            y_set = y_set + particle.y * particle.w
            z_set = z_set + particle.z * particle.w
            q = p_visualisation.getQuaternionFromEuler([particle.x_angle, particle.y_angle, particle.z_angle])
            qws.append(particle.w)
            quaternions.append([q[0], q[1], q[2], q[3]])
            w_set = w_set + particle.w
        # q = average_quaternions(np.array(quaternions))
        q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
        x_angle, y_angle, z_angle = p_visualisation.getEulerFromQuaternion([q[0], q[1], q[2], q[3]])
        return x_set / w_set, y_set / w_set, z_set / w_set, x_angle, y_angle, z_angle
    
    def compute_transformation_matrix(self, a_pos,a_ori,b_pos,b_ori):
        ow_T_a_3_3 = transformations.quaternion_matrix(a_ori)
        ow_T_a_4_4 = self.rotation_4_4_to_transformation_4_4(ow_T_a_3_3,a_pos)
        ow_T_b_3_3 = transformations.quaternion_matrix(b_ori)
        ow_T_b_4_4 = self.rotation_4_4_to_transformation_4_4(ow_T_b_3_3,b_pos)
        a_T_ow_4_4 = np.linalg.inv(ow_T_a_4_4)
        a_T_b_4_4 = np.dot(a_T_ow_4_4,ow_T_b_4_4)
        return a_T_b_4_4
    
    def rotation_4_4_to_transformation_4_4(self, rotation_4_4,pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4



#function independent of Class        
def get_real_object_pos(object_id):
    object_info = p_visualisation.getBasePositionAndOrientation(object_id)
    return object_info[0]       
def get_observation(object_id):
    object_info = get_real_object_pos(object_id)
    return object_info
def rotation_4_4_to_transformation_4_4(rotation_4_4,pos):
    rotation_4_4[0][3] = pos[0]
    rotation_4_4[1][3] = pos[1]
    rotation_4_4[2][3] = pos[2]
    return rotation_4_4
def compute_pos_err_bt_2_points(pos1,pos2):
    x1=pos1[0]
    y1=pos1[1]
    z1=pos1[2]
    x2=pos2[0]
    y2=pos2[1]
    z2=pos2[2]
    x_d = x1-x2
    y_d = y1-y2
    z_d = z1-z2
    distance = math.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)
    return distance
def compute_ang_err_bt_2_points(object1_ori,object2_ori):
    #x,y,z,w
    obj1_ori = copy.deepcopy(object1_ori)
    obj2_ori = copy.deepcopy(object2_ori)
    #w,x,y,z
    obj1_quat = Quaternion(x=obj1_ori[0],y=obj1_ori[1],z=obj1_ori[2],w=obj1_ori[3])
    obj2_quat = Quaternion(x=obj2_ori[0],y=obj2_ori[1],z=obj2_ori[2],w=obj2_ori[3])
    diff_bt_o1_o2 = obj2_quat * obj1_quat.inverse
    cos_theta_over_2 = diff_bt_o1_o2.w
    sin_theta_over_2 = math.sqrt(diff_bt_o1_o2.x ** 2 + diff_bt_o1_o2.y ** 2 + diff_bt_o1_o2.z ** 2)
    theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
    theta = theta_over_2 * 2
    return theta
def comp_z_ang(ang1,ang2):
    z1 = ang1[2]
    z2 = ang2[2]
    theta = abs(z2-z1)
    return theta
def compute_transformation_matrix(init_robot_pos,init_robot_ori,init_object_pos,init_object_ori):
    robot_transformation_matrix = transformations.quaternion_matrix(init_robot_ori)
    ow_T_robot = rotation_4_4_to_transformation_4_4(robot_transformation_matrix,init_robot_pos)
    object_transformation_matrix = transformations.quaternion_matrix(init_object_ori)
    ow_T_object = rotation_4_4_to_transformation_4_4(object_transformation_matrix,init_object_pos)
    robot_T_ow = np.linalg.inv(ow_T_robot)
    robot_T_object = np.dot(robot_T_ow,ow_T_object)
    return robot_T_object
def add_noise_to_Opti(current_pos,sigma_obs):
    mean = current_pos
    sigma = sigma_obs/(2 ** (1.0/2))
    new_pos_is_added_noise = take_easy_gaussian_value(mean, sigma)
    return new_pos_is_added_noise
def take_easy_gaussian_value(mean,sigma):
    normal = random.normalvariate(mean, sigma)
    return normal

def display_real_object_in_visual_model(ID,opti_obj_pos,opti_obj_ori):
    p_visualisation.resetBasePositionAndOrientation(ID,
                                                    opti_obj_pos,
                                                    opti_obj_ori)
def cheat_dope_obj_ang(angle):
    ang = copy.deepcopy(angle)
    if angle > -math.pi/4 and angle < math.pi/4:
        ang = 0.0
    elif angle > math.pi/4 and angle < 3 * math.pi/4:
        ang = math.pi/2
    elif angle > 3 * math.pi/4 and angle < 5 * math.pi/4:
        ang = math.pi
    elif angle > -3 * math.pi/4 and angle < -math.pi/4:
        ang = -math.pi/2
    elif angle > -5 * math.pi/4 and angle < -3 * math.pi/4:
        ang = -math.pi
    return ang
def angle_correction(angle):
    # print("angle before: ",angle)
    # if angle >= (math.pi*3.0/2.0):
    #     angle = angle - 2 * math.pi
    # elif math.pi/2.0 <= angle and angle < (math.pi*3.0/2.0):
    #     angle = angle - math.pi
    # elif -(math.pi*3.0/2.0) < angle and angle <= -math.pi/2.0:
    #     angle = angle + math.pi
    # elif angle <= -(math.pi*3.0/2.0):
    #     angle = angle + 2 * math.pi
    if math.pi <= angle <= (3.0 * math.pi):
        angle = angle - 2 * math.pi
    elif -(3.0 * math.pi) <= angle <= -math.pi:
        angle = angle + 2 * math.pi
    angle = abs(angle)
    # print("angle _after: ",angle)
    return angle
if __name__ == '__main__':
    t_begin = time.time()
    particle_cloud = []
    particle_num = 80
    visualisation_flag = True
    visualisation_particle_flag = False
    d_thresh = 0.002
    a_thresh = 0.01
    d_thresh_PM = 0.003
    a_thresh_PM = 0.015
    flag_record = 0
    flag_record_dope = 0
    flag_record_PFPE = 0
    flag_record_PFPM = 0
    flag_update_num_PM = 0
    flag_update_num_PE = 0
    flag_record_PM_file = 0
    flag_write_csv_file = 0
    #error in xyz axis DOPE
    boss_pf_update_interval_in_real = 0.13
    change_sim_time = 1.0/90
    boss_sigma_obs_x = 0.03973017808163751
    boss_sigma_obs_y = 0.01167211468503462
    boss_sigma_obs_z = 0.02820930183351492
    # boss_sigma_obs_x = 0.032860982
    # boss_sigma_obs_y = 0.012899399
    # boss_sigma_obs_z = 0.01
    boss_sigma_obs_ang = 0.216773873
    boss_sigma_obs_pos = 0.038226405
    
    rospy.init_node('PF_for_dope')
    
    #build an object of class "Ros_listener"
    ros_listener = Ros_listener()
    #get pose info from DOPE
    listener = tf.TransformListener()
    while True:
        try:
            (trans,rot) = listener.lookupTransform('/panda_link0', '/cracker', rospy.Time(0))
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    rob_T_obj_dope_pos = list(trans)
    rob_T_obj_dope_ori = list(rot)
    rob_T_obj_dope_3_3 = transformations.quaternion_matrix(rob_T_obj_dope_ori)
    rob_T_obj_dope = rotation_4_4_to_transformation_4_4(rob_T_obj_dope_3_3,rob_T_obj_dope_pos)
    #give some time to update the data
    time.sleep(0.5)
    init_robot_pos = ros_listener.robot_pos
    init_robot_ori = ros_listener.robot_ori 
    init_object_pos = ros_listener.object_pos
    init_object_ori = ros_listener.object_ori
    base_of_cheezit_pos = ros_listener.base_pos
    base_of_cheezit_ori = ros_listener.base_ori
    
    pybullet_robot_pos = [0.0, 0.0, 0.026]
    pybullet_robot_ori = [0,0,0,1]
    #compute transformation matrix (OptiTrack)
    #input('Press [ENTER] to compute transformation matrix (OptiTrack)')
    robot_T_object = compute_transformation_matrix(init_robot_pos,init_robot_ori,init_object_pos,init_object_ori)
    #input('Press [ENTER] to compute the pose of object in the pybullet world')
    pw_T_robot_3_3 = transformations.quaternion_matrix(pybullet_robot_ori)
    pw_T_robot = rotation_4_4_to_transformation_4_4(pw_T_robot_3_3,pybullet_robot_pos)
    pw_T_object = np.dot(pw_T_robot,robot_T_object)
    pw_T_object_pos = [pw_T_object[0][3],pw_T_object[1][3],pw_T_object[2][3]]       
    pw_T_object_ori = transformations.quaternion_from_matrix(pw_T_object) 
    pw_T_object_ang = p_visualisation.getEulerFromQuaternion(pw_T_object_ori)
    #load the groud truth object
    if visualisation_flag == True:
        optitrack_object_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cube/cheezit_real_obj_with_visual_small_hor.urdf"),
                                                       pw_T_object_pos,
                                                       pw_T_object_ori)

    robot_T_base = compute_transformation_matrix(init_robot_pos, init_robot_ori, base_of_cheezit_pos,
                                                 base_of_cheezit_ori)
    # input('Press [ENTER] to compute the pose of object in the pybullet world')
    pw_T_robot_3_3 = transformations.quaternion_matrix(pybullet_robot_ori)
    pw_T_robot = rotation_4_4_to_transformation_4_4(pw_T_robot_3_3, pybullet_robot_pos)
    pw_T_base = np.dot(pw_T_robot, robot_T_base)
    pw_T_base_pos = [pw_T_base[0][3], pw_T_base[1][3], pw_T_base[2][3]]
    pw_T_base_ori = transformations.quaternion_from_matrix(pw_T_base)
    pw_T_base_ang = p_visualisation.getEulerFromQuaternion(pw_T_base_ori)
    if visualisation_flag == True:
        optitrack_base_id = p_visualisation.loadURDF(
            os.path.expanduser("~/phd_project/object/cube/base_of_cheezit.urdf"),
            pw_T_base_pos,
            pw_T_base_ori)

    #compute pose of object in DOPE
    pw_T_object_dope = np.dot(pw_T_robot,rob_T_obj_dope)
    pw_T_object_pos_dope = [pw_T_object_dope[0][3],pw_T_object_dope[1][3],pw_T_object_dope[2][3]]       
    pw_T_object_ori_dope = transformations.quaternion_from_matrix(pw_T_object_dope) 
    pw_T_object_ang_dope = p_visualisation.getEulerFromQuaternion(pw_T_object_ori_dope)
    pw_T_object_ang_dope = list(pw_T_object_ang_dope)
    #load the DOPE object
    if visualisation_flag == True:
        dope_object_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cube/cheezit_dope_obj_with_visual_small_PE_hor.urdf"),
                                                  pw_T_object_pos_dope,
                                                  pw_T_object_ori_dope)
    #initialization pose of DOPE
    dope_obj_pos_init = copy.deepcopy(pw_T_object_pos_dope)
    dope_obj_ang_init = copy.deepcopy(pw_T_object_ang_dope)
    dope_obj_ori_init = copy.deepcopy(pw_T_object_ori_dope)
    dope_obj_pose_init = [dope_obj_pos_init[0],
                          dope_obj_pos_init[1],
                          dope_obj_pos_init[2],
                          dope_obj_ang_init[0],
                          dope_obj_ang_init[1],
                          dope_obj_ang_init[2]]
    boss_obs_pose_PFPM.append(dope_obj_pose_init)
    #input('test')
    #compute error
    err_opti_dope_pos = compute_pos_err_bt_2_points(pw_T_object_pos,pw_T_object_pos_dope)
    err_opti_dope_ang = compute_ang_err_bt_2_points(pw_T_object_ori,pw_T_object_ori_dope)
    err_opti_dope_ang = angle_correction(err_opti_dope_ang)
    err_opti_dope_sum = err_opti_dope_pos + err_opti_dope_ang

    
    #input('Press [ENTER] to initial real world model')
    #build an object of class "InitialRealworldModel"
    real_world_object = InitialRealworldModel(ros_listener.current_joint_values)
    #initialize the real robot in the pybullet
    real_robot_id = real_world_object.initial_robot(robot_pos = pybullet_robot_pos,robot_orientation = pybullet_robot_ori)
    #initialize the real object in the pybullet
    #real_object_id = real_world_object.initial_target_object(object_pos = pw_T_object_pos,object_orientation = pw_T_object_ori)
    #build an object of class "Franka_robot"
    franka_robot = Franka_robot(real_robot_id)
    
    #input('Press [ENTER] to initial simulation world model')
    #initialize sim world
    initial_parameter = InitialSimulationModel(particle_num, pybullet_robot_pos, pybullet_robot_ori, dope_obj_pos_init, dope_obj_ang_init, dope_obj_ori_init)
    initial_parameter.initial_particle() #only position of particle
    estimated_object_set = initial_parameter.initial_and_set_simulation_env(ros_listener.current_joint_values)
    estimated_object_pos = [estimated_object_set[0],estimated_object_set[1],estimated_object_set[2]]
    estimated_object_ang = [estimated_object_set[3],estimated_object_set[4],estimated_object_set[5]]
    estimated_object_ori = p_visualisation.getQuaternionFromEuler(estimated_object_ang)
    boss_est_pose_PFPM.append(estimated_object_set)
    if visualisation_particle_flag == True:
        initial_parameter.display_particle()
    initial_parameter.initial_and_set_simulation_env_PM(ros_listener.current_joint_values)
    if visualisation_particle_flag == True:
        initial_parameter.display_particle_PM()
    if visualisation_flag == True:
        estimated_object_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cube/cheezit_est_obj_with_visual_small_PE_hor.urdf"),
                                                       estimated_object_pos,
                                                       estimated_object_ori)
        estimated_object_id_PM = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cube/cheezit_est_obj_with_visual_small_PM_hor.urdf"),
                                                          estimated_object_pos,
                                                          estimated_object_ori)
    #input('test')
    #compute error
    err_opti_esti_pos = compute_pos_err_bt_2_points(estimated_object_pos,pw_T_object_pos)
    err_opti_esti_ang = compute_ang_err_bt_2_points(estimated_object_ori,pw_T_object_ori)
    err_opti_esti_ang = angle_correction(err_opti_esti_ang)
    err_opti_esti_sum = err_opti_esti_pos + err_opti_esti_ang
    
    t_before_record = time.time()
    boss_obse_err_pos_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
    boss_obse_err_ang_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
    boss_err_pos_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
    boss_err_ang_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
    flag_record = flag_record + 1
    flag_record_dope = flag_record_dope + 1
    boss_PFPE_err_pos_df.loc[flag_record_PFPE] = [flag_record_PFPE, t_before_record - t_begin, err_opti_esti_pos, 'PFPE']
    boss_PFPE_err_ang_df.loc[flag_record_PFPE] = [flag_record_PFPE, t_before_record - t_begin, err_opti_esti_ang, 'PFPE']
    boss_err_pos_df.loc[flag_record] = [flag_record_PFPE, t_before_record - t_begin, err_opti_esti_pos, 'PFPE']
    boss_err_ang_df.loc[flag_record] = [flag_record_PFPE, t_before_record - t_begin, err_opti_esti_ang, 'PFPE']
    flag_record = flag_record + 1
    flag_record_PFPE = flag_record_PFPE + 1
    boss_PFPM_err_pos_df.loc[flag_record_PFPM] = [flag_record_PFPM, t_before_record - t_begin, err_opti_esti_pos, 'PFPM']
    boss_PFPM_err_ang_df.loc[flag_record_PFPM] = [flag_record_PFPM, t_before_record - t_begin, err_opti_esti_ang, 'PFPM']
    boss_err_pos_df.loc[flag_record] = [flag_record_PFPM, t_before_record - t_begin, err_opti_esti_pos, 'PFPM']
    boss_err_ang_df.loc[flag_record] = [flag_record_PFPM, t_before_record - t_begin, err_opti_esti_ang, 'PFPM']
    flag_record = flag_record + 1
    flag_record_PFPM = flag_record_PFPM + 1
    
    # initial_parameter.particle_cloud #parameter of particle
    # initial_parameter.pybullet_particle_env_collection #env of simulation
    # initial_parameter.fake_robot_id_collection #id of robot in simulation
    # initial_parameter.particle_no_visual_id_collection #id of particle in simulation
    # print(initial_parameter.pybullet_particle_env_collection)

    #build an object of class "PFMove"
    robot1 = PFMove()
    robot2 = PFMovePM()
    #run the simulation
    Flag = True
    #compute DOPE object old pose
    dope_obj_pos_old = copy.deepcopy(dope_obj_pos_init)
    dope_obj_ang_old = copy.deepcopy(dope_obj_ang_init)
    dope_obj_ori_old = copy.deepcopy(dope_obj_ori_init)
    dope_obj_pos_old_PM = copy.deepcopy(dope_obj_pos_init)
    dope_obj_ang_old_PM = copy.deepcopy(dope_obj_ang_init)
    dope_obj_ori_old_PM = copy.deepcopy(dope_obj_ori_init)
    #input('Press [ENTER] to enter into while loop')
    t_end = time.time()
    agl_start_t = time.time()
    #compute pose of robot arm
    rob_link_9_pose_old_PE = p_visualisation.getLinkState(real_robot_id,9)
    rob_link_9_pose_old_PM = p_visualisation.getLinkState(real_robot_id,9)
    rob_link_9_ang_old_PE = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_old_PE[1])
    rob_link_9_ang_old_PM = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_old_PM[1])
    rob_pose_init = copy.deepcopy(rob_link_9_pose_old_PE)
    
    write_file_flag_obse = 0
    write_file_flag_PFPE = 0
    write_file_flag_PFPM = 0
    pf_update_rate = rospy.Rate(1.0/boss_pf_update_interval_in_real)
    file_time = 25
    run_PFPE_flag = True
    run_PFPM_flag = False
    print("Welcome to Our Approach !")
    t_begin_while = time.time()
    while True:
        #panda robot moves in the visualization window
        #for i_ss in range(240):
        franka_robot.fanka_robot_move(ros_listener.current_joint_values)
        #p_visualisation.stepSimulation()
        time.sleep(1./240.)
        
        #get pose info from DOPE
        while True:
            try:
                (trans,rot) = listener.lookupTransform('/panda_link0', '/cracker', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("can not find tf")
                continue
        rob_T_obj_dope_pos = list(trans)
        rob_T_obj_dope_ori = list(rot)
        rob_T_obj_dope_3_3 = transformations.quaternion_matrix(rob_T_obj_dope_ori)
        rob_T_obj_dope = rotation_4_4_to_transformation_4_4(rob_T_obj_dope_3_3,rob_T_obj_dope_pos)
        pw_T_object_dope = np.dot(pw_T_robot,rob_T_obj_dope)
        pw_T_object_pos_dope = [pw_T_object_dope[0][3],pw_T_object_dope[1][3],pw_T_object_dope[2][3]]       
        pw_T_object_ori_dope = transformations.quaternion_from_matrix(pw_T_object_dope) 
        pw_T_object_ang_dope = p_visualisation.getEulerFromQuaternion(pw_T_object_ori_dope)
        dope_obj_pos_cur = copy.deepcopy(pw_T_object_pos_dope)
        dope_obj_ang_cur = copy.deepcopy(pw_T_object_ang_dope)
        dope_obj_ori_cur = copy.deepcopy(pw_T_object_ori_dope)
        dope_obj_pose_cur = [dope_obj_pos_cur[0],
                             dope_obj_pos_cur[1],
                             dope_obj_pos_cur[2],
                             dope_obj_ang_cur[0],
                             dope_obj_ang_cur[1],
                             dope_obj_ang_cur[2]]
        #display DOPE object in visual model
        if visualisation_flag == True:
            display_real_object_in_visual_model(dope_object_id,dope_obj_pos_cur,dope_obj_ori_cur)
        
        #get ground true pose of robot and object
        robot_T_object = compute_transformation_matrix(ros_listener.robot_pos,
                                                       ros_listener.robot_ori,
                                                       ros_listener.object_pos,
                                                       ros_listener.object_ori)
        pw_T_robot_3_3 = transformations.quaternion_matrix(pybullet_robot_ori)
        pw_T_robot = rotation_4_4_to_transformation_4_4(pw_T_robot_3_3,pybullet_robot_pos)
        pw_T_object = np.dot(pw_T_robot,robot_T_object)
        pw_T_object_pos = [pw_T_object[0][3],pw_T_object[1][3],pw_T_object[2][3]]       
        pw_T_object_ori = transformations.quaternion_from_matrix(pw_T_object) 
        pw_T_object_ang = p_visualisation.getEulerFromQuaternion(pw_T_object_ori)
        pw_T_obj_pos_opti = copy.deepcopy(pw_T_object_pos)
        pw_T_obj_ang_opti = copy.deepcopy(pw_T_object_ang)
        pw_T_obj_ori_opti = copy.deepcopy(pw_T_object_ori)
        
        #compute distance between old DOPE obj and cur DOPE obj (position and angle)
        #dis_betw_cur_and_old = compute_pos_err_bt_2_points(dope_obj_pos_cur,dope_obj_pos_old)
        #ang_betw_cur_and_old = compute_ang_err_bt_2_points(dope_obj_ori_cur,dope_obj_ori_old)
        #dis_betw_cur_and_old_PM = compute_pos_err_bt_2_points(dope_obj_pos_cur,dope_obj_pos_old_PM)
        #ang_betw_cur_and_old_PM = compute_ang_err_bt_2_points(dope_obj_ori_cur,dope_obj_ori_old_PM)

        #compute distance between old robot arm and cur robot arm (position and angle)
        rob_link_9_pose_cur_PE = p_visualisation.getLinkState(real_robot_id,9)
        rob_link_9_pose_cur_PM = p_visualisation.getLinkState(real_robot_id,9)
        rob_link_9_ang_cur_PE = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_cur_PE[1])
        rob_link_9_ang_cur_PM = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_cur_PM[1])
        #dis_robcur_robold_PE = compute_pos_err_bt_2_points(rob_link_9_pose_cur_PE[0],rob_link_9_pose_old_PE[0])
        #dis_robcur_robold_PM = compute_pos_err_bt_2_points(rob_link_9_pose_cur_PM[0],rob_link_9_pose_old_PM[0])
        #ang_robcur_robold_PE = comp_z_ang(rob_link_9_ang_cur_PE,rob_link_9_ang_old_PE)
        #ang_robcur_robold_PM = comp_z_ang(rob_link_9_ang_cur_PM,rob_link_9_ang_old_PM)
        
        #write_file_judgement = compute_pos_err_bt_2_points(rob_link_9_pose_cur_PE[0],rob_pose_init[0])
        
        #Determine if particles need to be updated
        while True:
        #if (dis_betw_cur_and_old > d_thresh) or (ang_betw_cur_and_old > a_thresh) or (dis_robcur_robold_PE > d_thresh):
            if run_PFPE_flag == True:
                t_begin_PFPE = time.time()
                flag_update_num_PE = flag_update_num_PE + 1
                flag_write_csv_file = flag_write_csv_file + 1
                # print("PE: Need to update particles and update frequency is: " + str(flag_update_num_PE))
                #Cheat
                opti_obj_pos_cur = copy.deepcopy(pw_T_object_pos) #get pos of real object
                opti_obj_ori_cur = copy.deepcopy(pw_T_object_ori)
                nois_obj_pos_cur = copy.deepcopy(dope_obj_pos_cur)
                nois_obj_ang_cur = copy.deepcopy(dope_obj_ang_cur)
                #execute sim_robot movement 
                robot1.real_robot_control_PE(opti_obj_pos_cur,
                                             opti_obj_ori_cur,
                                             ros_listener.current_joint_values,
                                             nois_obj_pos_cur,
                                             nois_obj_ang_cur)
                # dope_obj_pos_old = copy.deepcopy(dope_obj_pos_cur)
                # dope_obj_ang_old = copy.deepcopy(dope_obj_ang_cur)
                # dope_obj_ori_old = copy.deepcopy(dope_obj_ori_cur)
                # rob_link_9_pose_old_PE = copy.deepcopy(rob_link_9_pose_cur_PE)
                if visualisation_flag == True:
                    display_real_object_in_visual_model(optitrack_object_id, pw_T_object_pos, pw_T_object_ori)
                # print("Average time of updating: ",np.mean(robot1.times))
                # print("PE: Finished")
                t_finish_PFPE = time.time()
        
            
        #if (dis_betw_cur_and_old_PM > d_thresh_PM) or (ang_betw_cur_and_old_PM > a_thresh_PM) or (dis_robcur_robold_PM > d_thresh_PM):
            if run_PFPM_flag == True:
                flag_update_num_PM = flag_update_num_PM + 1
                boss_obs_pose_PFPM.append(dope_obj_pose_cur)
                opti_obj_pos_cur_PM = copy.deepcopy(pw_T_object_pos) #get pos of real object
                opti_obj_ori_cur_PM = copy.deepcopy(pw_T_object_ori)
                nois_obj_pos_cur_PM = copy.deepcopy(dope_obj_pos_cur)
                nois_obj_ang_cur_PM = copy.deepcopy(dope_obj_ang_cur)
                robot2.real_robot_control_PM(opti_obj_pos_cur_PM,
                                             opti_obj_ori_cur_PM,
                                             nois_obj_pos_cur_PM,
                                             nois_obj_ang_cur_PM)
            
            
            # dope_obj_pos_old_PM = copy.deepcopy(dope_obj_pos_cur)
            # dope_obj_ang_old_PM = copy.deepcopy(dope_obj_ang_cur)
            # dope_obj_ori_old_PM = copy.deepcopy(dope_obj_ori_cur)
            # rob_link_9_pose_old_PM = copy.deepcopy(rob_link_9_pose_cur_PM)
            
            pf_update_rate.sleep()
            break
        t_end_while = time.time() 
        if t_end_while - t_begin > 32:
            file_name_obse_pos = 'time_scene2_obse_err_pos.csv'
            file_name_PFPE_pos = 'time_scene2_PFPE_err_pos.csv'
            file_name_PFPM_pos = 'time_scene2_PFPM_err_pos.csv'
            file_name_obse_ang = 'time_scene2_obse_err_ang.csv'
            file_name_PFPE_ang = 'time_scene2_PFPE_err_ang.csv'
            file_name_PFPM_ang = 'time_scene2_PFPM_err_ang.csv'
            # boss_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_pos,index=0,header=0,mode='a')
            # boss_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_ang,index=0,header=0,mode='a')
            # print("write pos and ang file")
            if run_PFPE_flag == True:
                boss_obse_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_obse_pos,index=0,header=0,mode='a')
                boss_obse_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_obse_ang,index=0,header=0,mode='a')
                print("write obser file")
                write_file_flag_obse = write_file_flag_obse + 1
            # if flag_write_csv_file > 65 and write_file_flag_PFPE == 0:
                boss_PFPE_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_PFPE_pos,index=0,header=0,mode='a')
                boss_PFPE_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_PFPE_ang,index=0,header=0,mode='a')
                print("write PFPE file")
                write_file_flag_PFPE = write_file_flag_PFPE + 1
                print("PE: Update frequency is: " + str(flag_update_num_PE))
        # if flag_write_csv_file > 65 and write_file_flag_PFPM == 0:
            if run_PFPM_flag == True:
                boss_PFPM_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_PFPM_pos,index=0,header=0,mode='a')
                boss_PFPM_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_PFPM_ang,index=0,header=0,mode='a')
                print("write PFPM file")
                write_file_flag_PFPM = write_file_flag_PFPM + 1
                print("PM: Update frequency is: " + str(flag_update_num_PM))
            break
        if Flag is False:
            break
        
    p_visualisation.disconnect()
    

