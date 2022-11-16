#!/usr/bin/python3
#ROS
from concurrent.futures.process import _threads_wakeups
import itertools
import os.path
from pickle import TRUE
from re import T
from ssl import ALERT_DESCRIPTION_ILLEGAL_PARAMETER
from tkinter.tix import Tree
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
import signal
import sys
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
#from sksurgerycore.algorithms.averagequaternions import average_quaternions
from quaternion_averaging import weightedAverageQuaternions
from Particle import Particle
from Object_Pose import Object_Pose

#Class of initialize the real world model
class Realworld():
    def __init__(self, object_num=0):
        self.object_num = object_num
        self.pw_T_objs_opti_pose_list = []
        self.pybullet_realworld_env = []
        self.real_robot_id = 0
        self.optitrack_object_id = 0
        self.obsevatio_object_id = 0
    
    def rotation_4_4_to_transformation_4_4(self, rotation_4_4, pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4
    
    def add_noise_pose(self, sim_par_cur_pos, sim_par_cur_ori):
        normal_x = self.add_noise_2_par(sim_par_cur_pos[0])
        normal_y = self.add_noise_2_par(sim_par_cur_pos[1])
        normal_z = self.add_noise_2_par(sim_par_cur_pos[2])
        pos_added_noise = [normal_x, normal_y, normal_z]
        # add noise on ang of each particle
        quat = copy.deepcopy(sim_par_cur_ori)# x,y,z,w
        quat_QuatStyle = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])# w,x,y,z
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
        nois_quat = Quaternion(x=x_quat, y=y_quat, z=z_quat, w=w_quat)
        new_quat = nois_quat * quat_QuatStyle
        ###pb_quat(x,y,z,w)
        ori_added_noise = [new_quat[1],new_quat[2],new_quat[3],new_quat[0]]
        return pos_added_noise, ori_added_noise
    
    def add_noise_2_par(self, current_pos):
        mean = current_pos
        pos_noise_sigma = 0.01
        sigma = pos_noise_sigma
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def add_noise_2_ang(self, cur_angle):
        boss_sigma_obs_ang = 0.0216773873 * 4
        mean = cur_angle
        sigma = boss_sigma_obs_ang
        ang_noise_sigma = 0.1
        sigma = ang_noise_sigma
        new_angle_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_angle_is_added_noise
    
    # random values generated from a Gaussian distribution
    def take_easy_gaussian_value(self, mean, sigma):
        normal = random.normalvariate(mean, sigma)
        return normal    
    
    def initial_realworld(self):
        pybullet_realworld_env = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT,GUI_SERVER
        self.pybullet_realworld_env.append(pybullet_realworld_env)

        pybullet_realworld_env.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85, cameraTargetPosition=[0.5, 0.3, 0.2])
        pybullet_realworld_env.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet_realworld_env.setGravity(0,0,-9.81)
        fake_plane_id = pybullet_realworld_env.loadURDF("plane.urdf")
        pw_T_rob_opti_pose_list = []
        pw_T_rob_opti_pos = [0.4472889147344443, -0.08, 0.0821006075425945]
        pw_T_rob_opti_ori = [0,1,0,1]
        pw_T_rob_opti_pose_list.append(pw_T_rob_opti_pos)
        pw_T_rob_opti_pose_list.append(pw_T_rob_opti_ori)
        # pw_T_rob_opti_ori = [0.52338279, 0.47884367, 0.52129429, -0.47437481]
        real_robot_id = pybullet_realworld_env.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_cheat_robot.urdf"),
                                                        pw_T_rob_opti_pos,
                                                        pw_T_rob_opti_ori)
        for i in range(240):
            pybullet_realworld_env.stepSimulation()
        self.real_robot_id = real_robot_id
        
        pw_T_objs_opti_pose_list = []
        pw_T_obj0_opti_pose = []
        pw_T_obj1_opti_pose = []
        pw_T_obj_obse_objects_list = []
        pw_T_obj_opti_objects_list = []
        # simulate getting names of objects
        objects_name_list = ["cracker", "soup"]
        pw_T_obj0_opti_pos = [0.4472889147344443, 0.08677179678403951, 0.0821006075425945]
        pw_T_obj0_opti_ori = [0.52338279, 0.47884367, 0.52129429, -0.47437481]
        pw_T_obj0_opti_pose.append(pw_T_obj0_opti_pos)
        pw_T_obj0_opti_pose.append(pw_T_obj0_opti_ori)
        pw_T_objs_opti_pose_list.append(pw_T_obj0_opti_pose)
        pw_T_obj1_opti_pos = [0.4472889147344443, 0.15677179678403951, 0.05]
        pw_T_obj1_opti_ori = [1.0, 0.0, 0.0, 1.0]
        pw_T_obj1_opti_pose.append(pw_T_obj1_opti_pos)
        pw_T_obj1_opti_pose.append(pw_T_obj1_opti_ori)
        pw_T_objs_opti_pose_list.append(pw_T_obj1_opti_pose)
        
        for i in range(self.object_num):
            # simulate getting ground truth of objects from OptiTrack
            opti_T_obj_opti_pos = copy.deepcopy(pw_T_objs_opti_pose_list[i][0])
            opti_T_obj_opti_ori = copy.deepcopy(pw_T_objs_opti_pose_list[i][1])
            opti_object_id = pybullet_realworld_env.loadURDF(os.path.expanduser("~/project/object/"+objects_name_list[i]+"/"+objects_name_list[i]+"_real_obj_hor.urdf"),
                                                                  opti_T_obj_opti_pos,
                                                                  opti_T_obj_opti_ori)
            self.optitrack_object_id.append(opti_object_id)
            opti_object = Object_Pose(objects_name_list[i], opti_object_id, opti_T_obj_opti_pos, opti_T_obj_opti_ori, index=i)
            pw_T_obj_opti_objects_list.append(opti_object)
            # simulate getting observation of objects from observation data
            pw_T_obj_obse_pos, pw_T_obj_obse_ori = self.add_noise_pose(opti_T_obj_opti_pos, opti_T_obj_opti_ori)
            obse_object_id = pybullet_realworld_env.loadURDF(os.path.expanduser("~/project/object/"+objects_name_list[i]+"/"+objects_name_list[i]+"_obse_obj_with_visual_hor.urdf"),
                                                             pw_T_obj_obse_pos,
                                                             pw_T_obj_obse_ori)
            self.obsevatio_object_id.append(obse_object_id)
            obse_object = Object_Pose(objects_name_list[i], obse_object_id, pw_T_obj_obse_pos, pw_T_obj_obse_ori, index=i)
            pw_T_obj_obse_objects_list.append(obse_object)
        return objects_name_list, pw_T_objs_opti_pose_list, pw_T_rob_opti_pose_list
    
    def cheat_robot_move(self, x=0.0, y=0.028, z=0.0):
        self.pybullet_realworld_env[0].resetBaseVelocity(self.real_robot_id, [x, y, z])
        self.pybullet_realworld_env[0].stepSimulation()
        return self.real_robot_id, self.pybullet_realworld_env[0]
        
        
        
        
        
        
        
        
        
        
        
        
