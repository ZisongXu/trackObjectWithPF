# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:57:49 2021

@author: 12106
"""
#ROS
import itertools
import os.path

import rospy
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
'''
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
p.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])
planeId = p.loadURDF("plane.urdf")
'''


#visualisation_model
p_visualisation = bc.BulletClient(connection_mode=p.GUI_SERVER)
p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
p_visualisation.setGravity(0,0,-9.81)
p_visualisation.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])
plane_id = p_visualisation.loadURDF("plane.urdf")

boss_error_df = pd.DataFrame()
boss_obser_df = pd.DataFrame()
boss_bsln2_df = pd.DataFrame()
boss_estimated_pos = []

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
    	self.setJointPosition(self.franka_robot_id,targetPositionsJoints)
    def setJointPosition(self,robot, position):
        position[7] = 0.039916139
        position[8] = 0.039916139
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
        self.joint_subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback)
        self.robot_pose = rospy.Subscriber('/mocap/rigid_bodies/pandaRobot/pose',PoseStamped, self.robot_pose_callback)
        self.object_pose = rospy.Subscriber('/mocap/rigid_bodies/zisongObject/pose',PoseStamped, self.object_pose_callback)
        self.current_joint_values = [-1.57,0.0,0.0,-2.8,1.7,1.57,1.1]
        self.robot_pos = [ 0.139080286026,
                          -0.581342339516,
                           0.0238141193986]
        #x,y,z,w
        self.robot_ori = [ 0.707254290581,
                           0.0115503482521,
                          -0.0140119809657,
                          -0.706726074219]
                          
        self.object_pos = [ 0.504023790359,
                           -0.214561194181,
                            0.0601389780641]
        #x,y,z,w
        self.object_ori = [-0.51964700222,
                           -0.476704657078,
                            0.490200251342,
                            0.512272834778]
        
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
            time.sleep(1./240.)
        
        return real_robot_id
    def initial_target_object(self,object_pos,object_orientation = [0,0,0,1]):
        #object_orientation = p_visualisation.getQuaternionFromEuler(object_euler)
        real_object_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_object_small.urdf"),
                                                  object_pos,
                                                  object_orientation)
        p_visualisation.changeDynamics(real_object_id,-1,lateralFriction = 0.53)
        return real_object_id
    def set_real_robot_JointPosition(self,pybullet_simulation_env,robot, position):
        print("Preparing the joint pose of the panda robot!")
        position[7] = 0.039916139
        position[8] = 0.039916139
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
    def __init__(self,particle_num,real_robot_start_pos,real_robot_start_ori,real_object_start_pos,real_object_start_ori):
        self.particle_num = particle_num
        self.cylinder_real_object_start_pos = real_object_start_pos
        self.cylinder_real_object_start_ori = real_object_start_ori
        self.cylinder_real_object_start_angle = []
        self.cylinder_real_robot_start_pos = real_robot_start_pos
        self.cylinder_real_robot_start_ori = real_robot_start_ori
        self.particle_cloud = []
        self.pybullet_particle_env_collection = []
        self.fake_robot_id_collection = []
        self.cylinder_particle_no_visual_id_collection = []
        self.cylinder_particle_with_visual_id_collection =[]
        self.sigma_obs = 0.01
        
        self.particle_cloud_copy = []
        self.pybullet_particle_env_collection_copy = []
        self.cylinder_particle_no_visual_id_collection_copy = []
        self.cylinder_particle_with_visual_id_collection_copy =[]
        
    def initial_particle(self):
        self.cylinder_real_object_start_angle = p_visualisation.getEulerFromQuaternion(self.cylinder_real_object_start_ori)
        real_object_x = copy.deepcopy(self.cylinder_real_object_start_pos[0])
        real_object_y = copy.deepcopy(self.cylinder_real_object_start_pos[1])
        real_object_z = copy.deepcopy(self.cylinder_real_object_start_pos[2])
        real_object = [real_object_x,real_object_y,real_object_z]
        self.cylinder_real_object_start_pos[0] = self.add_noise_to_obs_model(self.cylinder_real_object_start_pos[0])
        self.cylinder_real_object_start_pos[1] = self.add_noise_to_obs_model(self.cylinder_real_object_start_pos[1])
        noise_object_x = self.cylinder_real_object_start_pos[0]
        noise_object_y = self.cylinder_real_object_start_pos[1]
        noise_object_z = self.cylinder_real_object_start_pos[2]
        noise_object = [noise_object_x,noise_object_y,noise_object_z]
        error = self.compute_distance(noise_object,real_object)
        boss_obser_df[0]=[error]
        for i in range(self.particle_num):
            x,y,z,x_angle,y_angle,z_angle = self.generate_random_pose(self.cylinder_real_object_start_angle)
            w = 1/self.particle_num
            
            #recover: need to del
            #x = self.cylinder_real_object_start_pos[0]
            #y = self.cylinder_real_object_start_pos[1]
            
            particle = Particle(x,y,z,x_angle,y_angle,z_angle,w,index=i)
            self.particle_cloud.append(particle)
            
        #object_estimate_set = self.compute_estimate_pos_of_object(self.particle_cloud)
        #print("initial_object_estimate_pos:",object_estimate_pos_x,object_estimate_pos_y)
        #return object_estimate_set[0],object_estimate_set[1],object_estimate_set[2],object_estimate_set[3],object_estimate_set[4],object_estimate_set[5]
    def compute_distance(self,object_current_pos,object_last_update_pos):
        x_distance = object_current_pos[0] - object_last_update_pos[0]
        y_distance = object_current_pos[1] - object_last_update_pos[1]
        distance = math.sqrt(x_distance ** 2 + y_distance ** 2)
        return distance
    def generate_random_pose(self,cylinder_real_object_start_angle):
        angle = copy.deepcopy(cylinder_real_object_start_angle)
        x = random.uniform(self.cylinder_real_object_start_pos[0] - 0.07, self.cylinder_real_object_start_pos[0] + 0.07)
        y = random.uniform(self.cylinder_real_object_start_pos[1] - 0.07, self.cylinder_real_object_start_pos[1] + 0.07)
        z = self.cylinder_real_object_start_pos[2]
        x_angle = angle[0]
        y_angle = angle[1]
        z_angle = random.uniform(angle[2] - math.pi/6.0, angle[2] + math.pi/6.0)
        return x,y,z,x_angle,y_angle,z_angle
    def compute_estimate_pos_of_object(self, particle_cloud):
        x_set = 0
        y_set = 0
        z_set = 0
        x_angle_set = 0
        y_angle_set = 0
        z_angle_set = 0
        w_set = 0
        for index,particle in enumerate(particle_cloud):
            x_set = x_set + particle.x * particle.w
            y_set = y_set + particle.y * particle.w
            z_set = z_set + particle.z * particle.w
            x_angle_set = x_angle_set + particle.x_angle * particle.w
            y_angle_set = y_angle_set + particle.y_angle * particle.w
            z_angle_set = z_angle_set + particle.z_angle * particle.w
            w_set = w_set + particle.w
        return x_set/w_set,y_set/w_set,z_set/w_set,x_angle_set/w_set,y_angle_set/w_set,z_angle_set/w_set
        
    
    def display_particle(self):
        for index, particle in enumerate(self.particle_cloud):
            cylinder_visualize_particle_pos = [particle.x, particle.y, 0.057]
            cylinder_visualize_particle_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            cylinder_visualize_particle_orientation = p_visualisation.getQuaternionFromEuler(cylinder_visualize_particle_angle)
            cylinder_visualize_particle_Id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_particle_with_visual_small.urdf"),
                                                                      cylinder_visualize_particle_pos,
                                                                      cylinder_visualize_particle_orientation)
            self.cylinder_particle_with_visual_id_collection.append(cylinder_visualize_particle_Id)
    def display_particle_copy(self):
        for index, particle in enumerate(self.particle_cloud_copy):
            cylinder_visualize_particle_pos = [particle.x, particle.y, 0.057]
            cylinder_visualize_particle_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            cylinder_visualize_particle_orientation = p_visualisation.getQuaternionFromEuler(cylinder_visualize_particle_angle)
            cylinder_visualize_particle_Id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_particle_with_visual_small.urdf"),
                                                                      cylinder_visualize_particle_pos,
                                                                      cylinder_visualize_particle_orientation)
            self.cylinder_particle_with_visual_id_collection_copy.append(cylinder_visualize_particle_Id)
    def initial_and_set_simulation_env(self,pos_of_robot):
        for index, particle in enumerate(self.particle_cloud):
            pybullet_simulation_env = bc.BulletClient(connection_mode=p.DIRECT)
            self.pybullet_particle_env_collection.append(pybullet_simulation_env)
            
            pybullet_simulation_env.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_simulation_env.setGravity(0,0,-9.81)
            fake_plane_id = pybullet_simulation_env.loadURDF("plane.urdf")
            cylinder_fake_robot_start_pos = self.cylinder_real_robot_start_pos
            cylinder_fake_robot_start_orientation = self.cylinder_real_robot_start_ori
            fake_robot_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/phd_project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                                             cylinder_fake_robot_start_pos,
                                                             cylinder_fake_robot_start_orientation,
                                                             useFixedBase=1)
            self.fake_robot_id_collection.append(fake_robot_id)
            
            #set joint of fake robot
            self.set_sim_robot_JointPosition(pybullet_simulation_env,fake_robot_id,pos_of_robot)
            
            z = self.cylinder_real_object_start_pos[2]
            cylinder_particle_no_visual_start_pos = [particle.x, particle.y, z]
            cylinder_particle_no_visual_start_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            cylinder_particle_no_visual_start_orientation = pybullet_simulation_env.getQuaternionFromEuler(cylinder_particle_no_visual_start_angle)
            cylinder_particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_particle_no_visual_small.urdf"),
                                                                              cylinder_particle_no_visual_start_pos,
                                                                              cylinder_particle_no_visual_start_orientation)
            
            while True:
                flag = 0
                pmin,pmax = pybullet_simulation_env.getAABB(cylinder_particle_no_visual_id)
                collide_ids = pybullet_simulation_env.getOverlappingObjects(pmin,pmax)
                length = len(collide_ids)
                for t_i in range(length):
                    if collide_ids[t_i][1] == 8:
                        Px,Py,Pz,Px_angle,Py_angle,Pz_angle = self.generate_random_pose(self.cylinder_real_object_start_angle)
                        cylinder_particle_no_visual_angle = [Px_angle,Py_angle,Pz_angle]
                        cylinder_particle_no_visual_ori = pybullet_simulation_env.getQuaternionFromEuler(cylinder_particle_no_visual_angle)
                        pybullet_simulation_env.resetBasePositionAndOrientation(cylinder_particle_no_visual_id,
                                                                                [Px,Py,Pz],
                                                                                cylinder_particle_no_visual_ori)
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
            pybullet_simulation_env.changeDynamics(cylinder_particle_no_visual_id,-1,lateralFriction = 0.53)
            self.cylinder_particle_no_visual_id_collection.append(cylinder_particle_no_visual_id)  
        object_estimate_set = self.compute_estimate_pos_of_object(self.particle_cloud)
        return object_estimate_set[0],object_estimate_set[1],object_estimate_set[2],object_estimate_set[3],object_estimate_set[4],object_estimate_set[5] 
    def initial_and_set_simulation_env_copy(self,pos_of_robot):
        self.particle_cloud_copy = copy.deepcopy(self.particle_cloud)
        for index, particle in enumerate(self.particle_cloud_copy):
            pybullet_simulation_env = bc.BulletClient(connection_mode=p.DIRECT)
            self.pybullet_particle_env_collection_copy.append(pybullet_simulation_env)
            pybullet_simulation_env.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_simulation_env.setGravity(0,0,-9.81)
            fake_plane_id = pybullet_simulation_env.loadURDF("plane.urdf")
            
            z = self.cylinder_real_object_start_pos[2]
            cylinder_particle_no_visual_start_pos = [particle.x, particle.y, z]
            cylinder_particle_no_visual_start_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            cylinder_particle_no_visual_start_orientation = pybullet_simulation_env.getQuaternionFromEuler(cylinder_particle_no_visual_start_angle)
            cylinder_particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_particle_no_visual_small.urdf"),
                                                                              cylinder_particle_no_visual_start_pos,
                                                                              cylinder_particle_no_visual_start_orientation)
            pybullet_simulation_env.changeDynamics(cylinder_particle_no_visual_id,-1,lateralFriction = 0.53)
            self.cylinder_particle_no_visual_id_collection_copy.append(cylinder_particle_no_visual_id)
        object_estimate_set_copy = self.compute_estimate_pos_of_object(self.particle_cloud_copy)
        return object_estimate_set_copy[0],object_estimate_set_copy[1],object_estimate_set_copy[2],object_estimate_set_copy[3],object_estimate_set_copy[4],object_estimate_set_copy[5] 
        
        
    def set_sim_robot_JointPosition(self,pybullet_simulation_env,robot, position):
        position[7] = 0.039916139
        position[8] = 0.039916139
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
    def add_noise_to_obs_model(self,current_pos):
        mean = current_pos
        sigma = self.sigma_obs
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
        self.particle_no_visual_id_collection = copy.deepcopy(initial_parameter.cylinder_particle_no_visual_id_collection)
        self.pybullet_env_id_collection = copy.deepcopy(initial_parameter.pybullet_particle_env_collection)
        self.pybullet_sim_fake_robot_id_collection = copy.deepcopy(initial_parameter.fake_robot_id_collection)
        self.particle_with_visual_id_collection = copy.deepcopy(initial_parameter.cylinder_particle_with_visual_id_collection)
        
        self.particle_cloud_copy = copy.deepcopy(initial_parameter.particle_cloud_copy)
        self.particle_no_visual_id_collection_copy = copy.deepcopy(initial_parameter.cylinder_particle_no_visual_id_collection_copy)
        self.pybullet_env_id_collection_copy = copy.deepcopy(initial_parameter.pybullet_particle_env_collection_copy)
        self.particle_with_visual_id_collection_copy = copy.deepcopy(initial_parameter.cylinder_particle_with_visual_id_collection_copy)
        
        self.step_size = 1
        self.joint_num = 7
        
        self.judgement_flag = False
        self.d_thresh_limitation = 0.05
        
        self.u_flag = 1
        
        self.sigma_motion_model = 0.01
        self.sigma_observ_model = 0.015
        self.sigma_obs = 0.01

        self.object_estimate_pose_x = []
        self.object_estimate_pose_y = []
        self.object_real_____pose_x = []
        self.object_real_____pose_y = []

    #new structure
    def real_robot_control(self,observation,pw_T_object_ori,real_robot_joint_pos,rob_cur_pos,rob_cur_ori,rob_old_pos,rob_old_ori):   
        #real_robot_joint_pos = self.get_real_robot_joint(real_robot_id)
     
        #Cheat
        Flag = self.update_particle_filter_cheat(self.pybullet_env_id_collection, # simulation environment per particle
                                                 self.pybullet_sim_fake_robot_id_collection, # fake robot id per sim_env
                                                 real_robot_joint_pos, # execution actions of the fake robot
                                                 observation,
                                                 pw_T_object_ori,
                                                 rob_cur_pos,
                                                 rob_cur_ori,
                                                 rob_old_pos,
                                                 rob_old_ori)
        if Flag is False:
            return False    
        #real_robot_joint_pos = self.get_real_robot_joint(real_robot_id)
                
    def get_real_robot_joint(self, real_robot_id):
        real_robot_joint_list = []
        for index in range(self.joint_num):
            real_robot_info = p_visualisation.getJointState(real_robot_id,index)[0]
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
            time.sleep(1.0/240)
          
    def set_real_robot_JointPosition(self,pybullet_env,robot, position):
        position[7] = 0.039916139
        position[8] = 0.039916139
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
    
                                                            
    def compute_distance(self,object_current_pos,object_last_update_pos):
        x_distance = object_current_pos[0] - object_last_update_pos[0]
        y_distance = object_current_pos[1] - object_last_update_pos[1]
        distance = math.sqrt(x_distance ** 2 + y_distance ** 2)
        return distance
        
    #executed_control 
    def update_particle_filter_cheat(self, pybullet_sim_env, fake_robot_id, real_robot_joint_pos, observation, pw_T_object_ori,rob_cur_pos,rob_cur_ori,rob_old_pos,rob_old_ori):
        t1 = time.time()
        self.motion_update(pybullet_sim_env, fake_robot_id, real_robot_joint_pos)
        t2 = time.time()
        self.motion_update_copy(self.pybullet_env_id_collection_copy,rob_cur_pos,rob_cur_ori,rob_old_pos,rob_old_ori)
        t3 = time.time()
        #print("observation:",observation)
        estimated_object_pos = self.observation_update(observation,pw_T_object_ori)
        boss_estimated_pos.append(estimated_object_pos)
        t4 = time.time()
        print("motion model1 time consuming:",t2-t1)
        print("motion model2 time consuming:",t3-t2)
        print("observ model time consuming:",t4-t3)
        #if Flag is False:
        #    return False
        print("display particle")
        self.display_particle_in_visual_model(self.particle_cloud)
        self.display_real_object_in_visual_model(observation)
        self.draw_contrast_figure(estimated_object_pos,observation)
        
        
        error = self.compute_distance(estimated_object_pos,observation)
        boss_error_df[self.u_flag]=[error]
        if self.u_flag >= 7:
            print("write error file")
            boss_error_df.to_csv('error_sum_0_0.csv',index=0,header=0,mode='a')
        self.u_flag = self.u_flag + 1
        # print debug info of all particles here
        #input('hit enter to continue')
        return
    
    def motion_update(self, pybullet_sim_env, fake_robot_id, real_robot_joint_pos):
        for index, pybullet_env in enumerate(pybullet_sim_env):
            
            #execute the control
            for i in range(int(self.step_size*240)):
                self.set_real_robot_JointPosition(pybullet_env,fake_robot_id[index],real_robot_joint_pos)
                pybullet_env.stepSimulation()
                #time.sleep(1./240.)

            #real_robot_joint_list = []
            #for i2 in range(7):
            #    real_robot_info = pybullet_env.getJointState(fake_robot_id[index],i2)[0]
            #    real_robot_joint_list.append(real_robot_info)
            #print("real_robot_joint_list:")
            #print(real_robot_joint_list)
            
            
            sim_particle_old_pos = [self.particle_cloud[index].x,
                                    self.particle_cloud[index].y,
                                    self.particle_cloud[index].z]
            sim_particle_old_angle = [self.particle_cloud[index].x_angle,
                                      self.particle_cloud[index].y_angle,
                                      self.particle_cloud[index].z_angle]
            sim_particle_cur_pos = self.get_item_pos(pybullet_env,
                                                     initial_parameter.cylinder_particle_no_visual_id_collection[index])
            
            #add noise on particle filter
            normal_x = self.add_noise(sim_particle_cur_pos[0],sim_particle_old_pos[0])
            normal_y = self.add_noise(sim_particle_cur_pos[1],sim_particle_old_pos[1])
            
            self.particle_cloud[index].x = normal_x
            self.particle_cloud[index].y = normal_y
            
            #print("particle_x_before:",sim_particle_cur_pos[0]," ","particle_y_before:",sim_particle_cur_pos[1])
            #print("particle_x__after:",self.particle_cloud[index].x," ","particle_y__after:",self.particle_cloud[index].y)

    def observation_update(self, observation,pw_T_object_ori):
        pos_of_real_object = copy.deepcopy(observation) #pos of real object [1,2,3]
        pos_real_obj_x = pos_of_real_object[0]
        pos_real_obj_y = pos_of_real_object[1]
        pos_of_real_object[0] = self.add_noise_to_obs_model(pos_real_obj_x)
        pos_of_real_object[1] = self.add_noise_to_obs_model(pos_real_obj_y)
        
        real_object_x = copy.deepcopy(observation[0])
        real_object_y = copy.deepcopy(observation[1])
        real_object_z = copy.deepcopy(observation[2])
        real_object = [real_object_x,real_object_y,real_object_z]
        noise_object_x = pos_of_real_object[0]
        noise_object_y = pos_of_real_object[1]
        noise_object_z = pos_of_real_object[2]
        noise_object = [noise_object_x,noise_object_y,noise_object_z]
        error = self.compute_distance(noise_object,real_object)
        print("error:",error)
        boss_obser_df[self.u_flag]=[error]        
        if self.u_flag >= 7:
            print("write obser file")
            boss_obser_df.to_csv('obser_sum_0_0.csv',index=0,header=0,mode='a')        

        for index,particle in enumerate(self.particle_cloud):
            
            particle_x = particle.x
            particle_y = particle.y
            
            real_object_pos = pos_of_real_object
            
            real_object_pos_x = real_object_pos[0]
            real_object_pos_y = real_object_pos[1]
            
            distance = math.sqrt((particle_x - real_object_pos_x) ** 2 + (particle_y - real_object_pos_y) ** 2)
            
            x = distance
            mean = 0
            sigma = self.sigma_observ_model
            #weight = self.normal_distribution(x, mean, sigma) * sigma
            weight = self.normal_distribution(x, mean, sigma)
            
            particle.w = weight
            
        Flag = self.normalize_particles()
        #if Flag is False:
        #    return False
        
        self.resample_particles()
        self.set_paticle_in_each_sim_env()
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection):
            part_pos = pybullet_env.getBasePositionAndOrientation(self.particle_no_visual_id_collection[index])
            #print("particle:",part_pos[0][0],part_pos[0][1],part_pos[0][2])
        object_estimate_pos_x,object_estimate_pos_y,object_estimate_pos_z = self.compute_estimate_pos_of_object(self.particle_cloud)
        #print("object_estimate_pos:",object_estimate_pos_x,object_estimate_pos_y)
        #print("object_real_____pos:",pos_of_real_object[0],pos_of_real_object[1])
        estimated_object_pos = [object_estimate_pos_x,object_estimate_pos_y,object_estimate_pos_z]
        self.display_estimated_robot_in_visual_model(estimated_object_pos)    
        return estimated_object_pos

    def motion_update_copy(self, pybullet_sim_env_copy, rob_cur_pos, rob_cur_ori, rob_old_pos, rob_old_ori):
        robO_T_robN = self.compute_transformation_matrix(rob_old_pos, rob_old_ori, rob_cur_pos, rob_cur_ori)
        parO_T_parN = robO_T_robN
        for index, pybullet_env in enumerate(pybullet_sim_env_copy):
            sim_particle_old_pos = [self.particle_cloud_copy[index].x,
                                    self.particle_cloud_copy[index].y,
                                    self.particle_cloud_copy[index].z]
            sim_particle_old_angle = [self.particle_cloud_copy[index].x_angle,
                                      self.particle_cloud_copy[index].y_angle,
                                      self.particle_cloud_copy[index].z_angle]
            
            sim_particle_old_ori = pybullet_env.getQuaternionFromEuler(sim_particle_old_angle)
            
            pybullet_particle_transformation_matrix = transformations.quaternion_matrix(sim_particle_old_ori)
            pw_T_parO = rotation_4_4_to_transformation_4_4(pybullet_particle_transformation_matrix,sim_particle_old_pos)
            pw_T_parN = np.dot(pw_T_parO,parO_T_parN)
            pw_T_parN_pos = [pw_T_parN[0][3],
                             pw_T_parN[1][3],
                             pw_T_parN[2][3]]               
            pw_T_parN_ori = transformations.quaternion_from_matrix(pw_T_parN) 
            
            #add noise on particle filter
            normal_x = self.add_noise(pw_T_parN_pos[0],sim_particle_old_pos[0])
            normal_y = self.add_noise(pw_T_parN_pos[1],sim_particle_old_pos[1])
            
          
            self.particle_cloud_copy[index].x = normal_x
            self.particle_cloud_copy[index].y = normal_y
        return
    
    def get_item_pos(self,pybullet_env,item_id):
        item_info = pybullet_env.getBasePositionAndOrientation(item_id)
        return item_info[0]
    
    def add_noise_to_obs_model(self,current_pos):
        mean = current_pos
        sigma = self.sigma_obs
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def add_noise(self,current_pos,old_pos):
        distance = math.fabs(current_pos - old_pos)
        mean = current_pos
        sigma = self.sigma_motion_model
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def take_easy_gaussian_value(self,mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal
    
    def normal_distribution(self, x, mean, sigma):
        return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)
    
    def normalize_particles(self):
        tot_weight = sum([particle.w for particle in self.particle_cloud])
        if tot_weight == 0:
            print("Error!,total weight is 0")
            return False
        for particle in self.particle_cloud:
            particle_w = particle.w/tot_weight
            particle.w = particle_w
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
            particle = Particle(self.particle_cloud[i].x,self.particle_cloud[i].y,self.particle_cloud[i].z,self.particle_cloud[i].w,index)
            newParticles.append(particle)
        self.particle_cloud = copy.deepcopy(newParticles)
        
    def set_paticle_in_each_sim_env(self):
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection):
            visual_particle_pos = [self.particle_cloud[index].x, self.particle_cloud[index].y, 0.057]
            visual_particle_orientation = pybullet_env.getQuaternionFromEuler([0,0,0])
            
            pybullet_env.resetBasePositionAndOrientation(self.particle_no_visual_id_collection[index],
                                                         visual_particle_pos,
                                                         visual_particle_orientation)
        return
        
        
    def display_particle_in_visual_model(self, particle_cloud):
        for index, particle in enumerate(particle_cloud):
            visual_particle_pos = [particle.x, particle.y, 0.057]
            visual_particle_orientation = p_visualisation.getQuaternionFromEuler([0,0,0])
            p_visualisation.resetBasePositionAndOrientation(self.particle_with_visual_id_collection[index],
                                                            visual_particle_pos,
                                                            visual_particle_orientation)
            #print("visual_particle_pos:",visual_particle_pos)
            #particle_pos = self.get_item_pos(pybullet_env[index],initial_parameter.cylinder_particle_no_visual_id_collection[index])
    
    def display_real_object_in_visual_model(self, observation):
        #print("observation",observation)
        optitrack_obj_pos = observation
        optitrack_obj_ori = p_visualisation.getQuaternionFromEuler([0,0,0])
        p_visualisation.resetBasePositionAndOrientation(optitrack_object_id,
                                                        optitrack_obj_pos,
                                                        optitrack_obj_ori)
    def display_estimated_robot_in_visual_model(self, observation):
        optitrack_obj_pos = observation
        optitrack_obj_ori = p_visualisation.getQuaternionFromEuler([0,0,0])
        p_visualisation.resetBasePositionAndOrientation(estimated_object_id,
                                                        optitrack_obj_pos,
                                                        optitrack_obj_ori)    

    def draw_contrast_figure(self,estimated_object_pos,observation):
        print("Begin to draw contrast figure!")
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
        for index,particle in enumerate(particle_cloud):
            x_set = x_set + particle.x * particle.w
            y_set = y_set + particle.y * particle.w
            z_set = z_set + particle.z * particle.w
            w_set = w_set + particle.w
        return x_set/w_set,y_set/w_set,z_set/w_set
    
    def compute_transformation_matrix(self, init_robot_pos,init_robot_ori,init_object_pos,init_object_ori):
        robot_transformation_matrix = transformations.quaternion_matrix(init_robot_ori)
        ow_T_robot = rotation_4_4_to_transformation_4_4(robot_transformation_matrix,init_robot_pos)
        object_transformation_matrix = transformations.quaternion_matrix(init_object_ori)
        ow_T_object = rotation_4_4_to_transformation_4_4(object_transformation_matrix,init_object_pos)
        robot_T_ow = np.linalg.inv(ow_T_robot)
        robot_T_object = np.dot(robot_T_ow,ow_T_object)
        return robot_T_object
    '''
    def rotation_4_4_to_transformation_4_4(self, rotation_4_4,pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4
    '''
#function independent of Class        
def get_real_object_pos(object_id):
    object_info = p_visualisation.getBasePositionAndOrientation(object_id)
    return object_info[0] 
def compute_distance(object_current_pos,object_last_update_pos):
    x_distance = object_current_pos[0] - object_last_update_pos[0]
    y_distance = object_current_pos[1] - object_last_update_pos[1]
    distance = math.sqrt(x_distance ** 2 + y_distance ** 2)
    return distance        
def get_observation(object_id):
    object_info = get_real_object_pos(object_id)
    return object_info
def rotation_4_4_to_transformation_4_4(rotation_4_4,pos):
    rotation_4_4[0][3] = pos[0]
    rotation_4_4[1][3] = pos[1]
    rotation_4_4[2][3] = pos[2]
    return rotation_4_4
def compute_distance_between_2_points_3D(pos1,pos2):
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
def compute_transformation_matrix(init_robot_pos,init_robot_ori,init_object_pos,init_object_ori):
    robot_transformation_matrix = transformations.quaternion_matrix(init_robot_ori)
    #print("robot_transformation_matrix:")
    #print(robot_transformation_matrix)
    ow_T_robot = rotation_4_4_to_transformation_4_4(robot_transformation_matrix,init_robot_pos)
    #print("ow_T_robot:")
    #print(ow_T_robot)
    object_transformation_matrix = transformations.quaternion_matrix(init_object_ori)
    #print("object_transformation_matrix:")
    #print(object_transformation_matrix)
    ow_T_object = rotation_4_4_to_transformation_4_4(object_transformation_matrix,init_object_pos)
    #print("ow_T_object:")
    #print(ow_T_object)
    robot_T_ow = np.linalg.inv(ow_T_robot)
    robot_T_object = np.dot(robot_T_ow,ow_T_object)
    #print("robot_T_object:")
    #print(robot_T_object)
    return robot_T_object
if __name__ == '__main__':
    rospy.init_node('PF_for_optitrack')
    
    #build an object of class "Ros_listener"
    ros_listener = Ros_listener()
    
    #give some time to update the data
    time.sleep(0.5)
    init_robot_pos = ros_listener.robot_pos
    init_robot_ori = ros_listener.robot_ori 
    init_object_pos = ros_listener.object_pos
    init_object_ori = ros_listener.object_ori
    
    print("init_robot_pos:")
    print(init_robot_pos)
    print("init_robot_ori:")
    print(init_robot_ori)  
    print("init_object_pos:")
    print(init_object_pos)
    print("init_object_ori:")
    print(init_object_ori)

    #compute transformation matrix
    #input('Press [ENTER] to compute transformation matrix')
    robot_T_object = compute_transformation_matrix(init_robot_pos,init_robot_ori,init_object_pos,init_object_ori)

    pybullet_robot_pos = [0.0, 0.0, 0.0]
    pybullet_robot_ori = [0,0,0,1]

    
    #input('Press [ENTER] to compute the pose of object in the pybullet world')

    pybullet_robot_transformation_matrix = transformations.quaternion_matrix(pybullet_robot_ori)
    pw_T_robot = rotation_4_4_to_transformation_4_4(pybullet_robot_transformation_matrix,pybullet_robot_pos)
    pw_T_object = np.dot(pw_T_robot,robot_T_object)
    print("pw_T_object:")
    print(pw_T_object)
    pw_T_object_pos = [pw_T_object[0][3],
                       pw_T_object[1][3],
                       pw_T_object[2][3]]       

    pw_T_object_ori = transformations.quaternion_from_matrix(pw_T_object) 

    optitrack_object_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_real_object_with_visual_small.urdf"),
                                                   pw_T_object_pos,
                                                   pw_T_object_ori)                          

    #input('Press [ENTER] to initial real world model')
    #build an object of class "InitialRealworldModel"
    real_world_object = InitialRealworldModel(ros_listener.current_joint_values)
    #print("ros_listener.current_joint_values:")
    #print(ros_listener.current_joint_values)
    #initialize the real robot in the pybullet
    real_robot_id = real_world_object.initial_robot(robot_pos = pybullet_robot_pos,robot_orientation = pybullet_robot_ori)
    #initialize the real object in the pybullet
    real_object_id = real_world_object.initial_target_object(object_pos = pw_T_object_pos,object_orientation = pw_T_object_ori)
    #build an object of class "Franka_robot"
    franka_robot = Franka_robot(real_robot_id)
    
    #input('Press [ENTER] to initial simulation world model')
    particle_cloud = []
    particle_num = 50
    d_thresh_limitation = 0.05
    initial_parameter = InitialSimulationModel(particle_num,pybullet_robot_pos,pybullet_robot_ori,pw_T_object_pos,pw_T_object_ori)
    initial_parameter.initial_particle() #only position of particle

    #initial_parameter.initial_and_set_simulation_env()
    estimated_object_set = initial_parameter.initial_and_set_simulation_env(ros_listener.current_joint_values)
    estimated_object_pos = [estimated_object_set[0],estimated_object_set[1],estimated_object_set[2]]
    estimated_object_ang = [estimated_object_set[3],estimated_object_set[4],estimated_object_set[5]]
    estimated_object_ori = p_visualisation.getQuaternionFromEuler(estimated_object_ang)
    boss_estimated_pos.append(estimated_object_pos)
    
    initial_parameter.initial_and_set_simulation_env_copy(ros_listener.current_joint_values)
    initial_parameter.display_particle()
    estimated_object_id = p_visualisation.loadURDF(os.path.expanduser("~/phd_project/object/cylinder_estimated_object_with_visual_small.urdf"),
                                                   estimated_object_pos,
                                                   estimated_object_ori)
    error = compute_distance(estimated_object_pos,pw_T_object_pos)
    boss_error_df[0]=[error]
    boss_bsln2_df[0]=[error]
    #initial_parameter.particle_cloud #parameter of particle
    #initial_parameter.pybullet_particle_env_collection #env of simulation
    #initial_parameter.fake_robot_id_collection #id of robot in simulation
    #initial_parameter.cylinder_particle_no_visual_id_collection #id of particle in simulation
    #print(initial_parameter.pybullet_particle_env_collection)
    
    #build an object of class "PFMove"
    robot1 = PFMove()
    #wait for 2 seconds
    #time.sleep(2)
    
    #get real object pos
    real_object_last_update_pos = pw_T_object_pos
    real_object_last_update_ori = pw_T_object_ori
    #run the simulation
    Flag = True
    
    data_old = p_visualisation.getLinkState(real_robot_id,9)
    #input('Press [ENTER] to enter into while loop')
    while True:
        franka_robot.fanka_robot_move(ros_listener.current_joint_values)
        p_visualisation.stepSimulation()
        time.sleep(1./240.)
        
        init_robot_pos = ros_listener.robot_pos
        init_robot_ori = ros_listener.robot_ori 
        init_object_pos = ros_listener.object_pos
        init_object_ori = ros_listener.object_ori
        
        #Determine if particles need to be updated
        robot_T_object = compute_transformation_matrix(ros_listener.robot_pos,
                                                       ros_listener.robot_ori,
                                                       ros_listener.object_pos,
                                                       ros_listener.object_ori)
        pybullet_robot_transformation_matrix = transformations.quaternion_matrix(pybullet_robot_ori)
        pw_T_robot = rotation_4_4_to_transformation_4_4(pybullet_robot_transformation_matrix,pybullet_robot_pos)
        pw_T_object = np.dot(pw_T_robot,robot_T_object)
        pw_T_object_pos = [pw_T_object[0][3],pw_T_object[1][3],pw_T_object[2][3]]       
        pw_T_object_ori = transformations.quaternion_from_matrix(pw_T_object) 
        
        data_new = p_visualisation.getLinkState(real_robot_id,9)
        distance_between_current_and_old = compute_distance(data_new[0],data_old[0])
        real_object_current_pos = pw_T_object_pos
        #distance_between_current_and_old = compute_distance(real_object_current_pos,real_object_last_update_pos)#Cheat        
        if distance_between_current_and_old > d_thresh_limitation:
            obj_cur_pos = real_object_current_pos
            obj_cur_ori = pw_T_object_ori
            rob_cur_pose = copy.deepcopy(data_new)
            rob_cur_pos = data_new[0]
            rob_cur_ori = data_new[1]
            print("Need to update particles")
            #Cheat
            observation = real_object_current_pos #get pos of real object
            
            #execute sim_robot movement 
            Flag = robot1.real_robot_control(observation,
                                             pw_T_object_ori,
                                             ros_listener.current_joint_values,
                                             rob_cur_pos,
                                             rob_cur_ori,
                                             data_old[0],
                                             data_old[1])
            
            real_object_last_update_pos = obj_cur_pos
            real_object_last_update_ori = obj_cur_ori
            data_old = copy.deepcopy(rob_cur_pose)
        if Flag is False:
            break  
    p_visualisation.disconnect()
    
