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
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3
from PBPF.msg import object_pose, particle_pose, particle_list, estimated_obj_pose
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
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
#from sksurgerycore.algorithms.averagequaternions import average_quaternions
from quaternion_averaging import weightedAverageQuaternions
from Particle import Particle
from Object_Pose import Object_Pose
import yaml

# - Parmesan
# - Milk
# - SaladDressing
# - Mustard
# - Mayo
# - cracker
# - soup
# - Ketchup

#Class of initialize the simulation model
class SingleENV(multiprocessing.Process):
    def __init__(self, object_num, robot_num, particle_num,
                 pw_T_rob_sim_pose_list_alg, pw_T_obj_obse_obj_list_alg, pw_T_objs_touching_targetObjs_list,
                 update_style_flag, sim_time_step, pf_update_interval_in_real, 
                 result_dict, daemon=True):
        super().__init__(daemon=daemon)
        self.queue = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()
        self.result = result_dict

        self.object_num = object_num
        self.robot_num = robot_num
        self.particle_num = particle_num
        self.pw_T_rob_sim_pose_list_alg = pw_T_rob_sim_pose_list_alg
        self.pw_T_obj_obse_obj_list_alg = pw_T_obj_obse_obj_list_alg
        self.pw_T_objs_touching_targetObjs_list = pw_T_objs_touching_targetObjs_list
        self.update_style_flag = update_style_flag
        self.sim_time_step = sim_time_step
        self.pf_update_interval_in_real = pf_update_interval_in_real

        self.collision_detection_obj_id_collection = []
        self.particle_objects_id_collection = ["None"] * self.object_num
        self.objects_list = ["None"] * self.object_num
        
        self.pf_update_interval_in_sim = self.pf_update_interval_in_real / self.sim_time_step
        self.boss_sigma_obs_pos_init = 0.01 # original value: 16cm/10CM /5cm
        # self.boss_sigma_obs_pos_init = 0.00 # original value: 16cm/10CM /5cm
        # self.boss_sigma_obs_pos_init = 0.09 # original value: 16cm/10CM 
        self.boss_sigma_obs_x = self.boss_sigma_obs_pos_init / math.sqrt(2)
        self.boss_sigma_obs_y = self.boss_sigma_obs_pos_init / math.sqrt(2)
        self.boss_sigma_obs_z = 0.02   # 2cm
        # self.boss_sigma_obs_z = 0.00   # 2cm
        # self.boss_sigma_obs_ang_init = 0.0216773873 * 20 # original value: 0.0216773873 * 20
        # self.boss_sigma_obs_ang_init = 0.0216773873 * 10 # original value: 0.0216773873 * 20
        # self.boss_sigma_obs_ang_init = 0.0216773873 * 10 # original value: 0.0216773873 * 20
        self.boss_sigma_obs_ang_init = 0.0216773873 * 1 # original value: 0.0216773873 * 20
        # self.boss_sigma_obs_ang_init = 0 # original value: 0.0216773873 * 20
        
        
        
        # mark
        # self.boss_sigma_obs_x = 0
        # self.boss_sigma_obs_y = 0
        # self.boss_sigma_obs_z = 0
        # self.boss_sigma_obs_ang_init = 0
        
        with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
            self.parameter_info = yaml.safe_load(file)
        self.gazebo_flag = self.parameter_info['gazebo_flag']
        self.task_flag = self.parameter_info['task_flag']
        self.SIM_REAL_WORLD_FLAG = self.parameter_info['sim_real_world_flag']
        self.SHOW_RAY = self.parameter_info['show_ray'] 
        self.VK_RENDER_FLAG = self.parameter_info['vk_render_flag'] 
        self.OBJS_ARE_NOT_TOUCHING_TARGET_OBJS_NUM = self.parameter_info['objs_are_not_touching_target_objs_num']
        self.OBJS_TOUCHING_TARGET_OBJS_NUM = self.parameter_info['objs_touching_target_objs_num']
        self.OBJECT_NAME_LIST = self.parameter_info['object_name_list']
        self.OBJECT_NUM = self.parameter_info['object_num']
        
        self.PANDA_ROBOT_LINK_NUMBER = self.parameter_info['panda_robot_link_number']
        self.MASS_marker = self.parameter_info['MASS_marker']
        self.FRICTION_marker = self.parameter_info['FRICTION_marker']

        self.MASS_MEAN_list = [0.5] * self.object_num
        self.MASS_MEAN = 1.0 # 0.380
        self.MASS_SIGMA_list = [0.5] * self.object_num
        self.MASS_SIGMA = 0.5 # 0.5
        self.MASS_MIN_VALUE = 0.05
        self.FRICTION_MEAN = 0.1
        self.FRICTION_SIGMA = 0.3
        # self.FRICTION_SIGMA = 0.0
        self.RESTITUTION_MEAN = 0.9
        self.RESTITUTION_SIGMA = 0.2 # 0.2
        # MASS_MEAN = 1.750 # 0.380
        # MASS_SIGMA = 0.5
        # FRICTION_MEAN = 0.1
        # FRICTION_SIGMA = 0.3
        # RESTITUTION_MEAN = 0.9
        # RESTITUTION_SIGMA = 0.2


        # Motion Model Noise
        self.MOTION_MODEL_POS_NOISE = 0.005 # original value = 0.005
        self.MOTION_MODEL_ANG_NOISE = 0.05 # original value = 0.05/0.5 
        # self.MOTION_MODEL_POS_NOISE = 0.0 # original value = 0.005

        self.MOTION_NOISE = True
        self.MASS_NOISE = True
        self.FRICTION_NOISE = True

        if self.MASS_marker == 'mA' or self.MASS_marker == 'mB' or self.MASS_marker == 'mC' or self.MASS_marker == 'mD' or self.MASS_marker == 'mE' or self.MASS_marker == 'mF' or self.MASS_marker == 'mG':
            self.MOTION_NOISE = False
            self.MASS_NOISE = False
            self.FRICTION_NOISE = False
        if self.MASS_marker == 'mAN' or self.MASS_marker == 'mBN' or self.MASS_marker == 'mCN' or self.MASS_marker == 'mDN' or self.MASS_marker == 'mEN' or self.MASS_marker == 'mFN' or self.MASS_marker == 'mGN':
            self.MOTION_NOISE = True
            self.MASS_NOISE = True
            self.FRICTION_NOISE = False
        if self.MASS_marker == 'mANN' or self.MASS_marker == 'mBNN' or self.MASS_marker == 'mCNN' or self.MASS_marker == 'mDNN' or self.MASS_marker == 'mENN' or self.MASS_marker == 'mFNN' or self.MASS_marker == 'mGNN':
            self.MOTION_NOISE = True
            self.MASS_NOISE = True
            self.FRICTION_NOISE = True
                
        # if self.FRICTION_marker == 'fA' or self.FRICTION_marker == 'fB' or self.FRICTION_marker == 'fC' or self.FRICTION_marker == 'fD' or self.FRICTION_marker == 'fE' or self.FRICTION_marker == 'fF' or self.FRICTION_marker == 'fG' or self.FRICTION_marker == 'fH':
        #     self.MOTION_NOISE = False
        #     self.MASS_NOISE = False
        #     self.FRICTION_NOISE = False
        # if self.FRICTION_marker == 'fAN' or self.FRICTION_marker == 'fBN' or self.FRICTION_marker == 'fCN' or self.FRICTION_marker == 'fDN' or self.FRICTION_marker == 'fEN' or self.FRICTION_marker == 'fFN' or self.FRICTION_marker == 'fGN' or self.FRICTION_marker == 'fHN':
        #     self.MOTION_NOISE = True
        #     self.MASS_NOISE = True
        #     self.FRICTION_NOISE = False
        # if self.FRICTION_marker == 'fANN' or self.FRICTION_marker == 'fBNN' or self.FRICTION_marker == 'fCNN' or self.FRICTION_marker == 'fDNN' or self.FRICTION_marker == 'fENN' or self.FRICTION_marker == 'fFNN' or self.FRICTION_marker == 'fGNN' or self.FRICTION_marker == 'fHNN':
        #     self.MOTION_NOISE = True
        #     self.MASS_NOISE = True
        #     self.FRICTION_NOISE = True


        # self.MASS_MEAN_list = [0.06] * self.object_num
        # self.MASS_MEAN_list = np.array(self.MASS_MEAN_list)
        # if self.MASS_marker == 'mA' or self.MASS_marker == 'mAN' or self.MASS_marker == 'mANN':
        #     self.MASS_MEAN_list = [0.01] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * (0.5)
        # elif self.MASS_marker == 'mB' or self.MASS_marker == 'mBN' or self.MASS_marker == 'mBNN':
        #     self.MASS_MEAN_list = [0.5] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 1
        # elif self.MASS_marker == 'mC' or self.MASS_marker == 'mCN' or self.MASS_marker == 'mCNN':
        #     self.MASS_MEAN_list = [0.1] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 2
        # elif self.MASS_marker == 'mD' or self.MASS_marker == 'mDN' or self.MASS_marker == 'mDNN':
        #     self.MASS_MEAN_list = [0.5] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 3
        # elif self.MASS_marker == 'mE' or self.MASS_marker == 'mEN' or self.MASS_marker == 'mENN':
        #     self.MASS_MEAN_list = [1.0] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 4
        # elif self.MASS_marker == 'mF' or self.MASS_marker == 'mFN' or self.MASS_marker == 'mFNN':
        #     self.MASS_MEAN_list = [5.0] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 5
        # elif self.MASS_marker == 'mG' or self.MASS_marker == 'mGN' or self.MASS_marker == 'mGNN':
        #     self.MASS_MEAN_list = [10.0] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 10



        # self.MASS_MEAN_list = [0.06, 0.05, 0.06] 
        # self.MASS_SIGMA_list = [0.1, 0.1, 0.1] 
        # self.MASS_SIGMA_list = np.array(self.MASS_SIGMA_list)
        # if self.MASS_marker == 'mA' or self.MASS_marker == 'mAN' or self.MASS_marker == 'mANN':
        #     self.MASS_SIGMA_list = self.MASS_SIGMA_list / 100.0
        #     self.FRICTION_SIGMA = 0.3 / 100.0
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * (0.5)
        # elif self.MASS_marker == 'mB' or self.MASS_marker == 'mBN' or self.MASS_marker == 'mBNN':
        #     self.MASS_SIGMA_list = self.MASS_SIGMA_list / 1.0
        #     self.FRICTION_SIGMA = 0.3 / 1.0
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 1
        # elif self.MASS_marker == 'mC' or self.MASS_marker == 'mCN' or self.MASS_marker == 'mCNN':
        #     self.MASS_SIGMA_list = self.MASS_SIGMA_list / 10.0
        #     self.FRICTION_SIGMA = 0.3 / 10.0
        #     # self.MASS_SIGMA_list = [0.5] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 2
        # elif self.MASS_marker == 'mD' or self.MASS_marker == 'mDN' or self.MASS_marker == 'mDNN':
        #     self.MASS_SIGMA_list = self.MASS_SIGMA_list * 2
        #     self.FRICTION_SIGMA = 0.3 * 2
        #     # self.MASS_SIGMA_list = [0.5] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 3
        # elif self.MASS_marker == 'mE' or self.MASS_marker == 'mEN' or self.MASS_marker == 'mENN':
        #     self.MASS_SIGMA_list = self.MASS_SIGMA_list * 5
        #     self.FRICTION_SIGMA = 0.3 * 5
        #     # self.MASS_SIGMA_list = [0.5] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 4
        # elif self.MASS_marker == 'mF' or self.MASS_marker == 'mFN' or self.MASS_marker == 'mFNN':
        #     self.MASS_SIGMA_list = self.MASS_SIGMA_list * 10
        #     self.FRICTION_SIGMA = 0.3 * 10
        #     # self.MASS_SIGMA_list = [0.5] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 5
        # elif self.MASS_marker == 'mG' or self.MASS_marker == 'mGN' or self.MASS_marker == 'mGNN':
        #     self.MASS_SIGMA_list = self.MASS_SIGMA_list * 100
        #     self.FRICTION_SIGMA = 0.3 * 100
            # self.MASS_SIGMA_list = [0.5] * self.object_num
            # self.MASS_MEAN_list = self.MASS_MEAN_list * 10

        self.MASS_MEAN_list = [0.06] * self.object_num
        # self.MASS_MEAN_list = np.array(self.MASS_MEAN_list)
        # if self.MASS_marker == 'mA' or self.MASS_marker == 'mAN' or self.MASS_marker == 'mANN':
        #     self.MOTION_MODEL_POS_NOISE = 0.000 # original value = 0.005
        #     self.MOTION_MODEL_ANG_NOISE = 0.00 # original value = 0.05/0.5 
        #     self.MOTION_NOISE = False
        #     # self.MASS_MEAN_list = [0.01] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * (0.5)
        # elif self.MASS_marker == 'mB' or self.MASS_marker == 'mBN' or self.MASS_marker == 'mBNN':
        #     self.MOTION_MODEL_POS_NOISE = 0.005 # original value = 0.005
        #     self.MOTION_MODEL_ANG_NOISE = 0.05 # original value = 0.05/0.5 
        #     # self.MASS_MEAN_list = [0.5] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 1
        # elif self.MASS_marker == 'mC' or self.MASS_marker == 'mCN' or self.MASS_marker == 'mCNN':
        #     self.MOTION_MODEL_POS_NOISE = 0.005 * 2 # original value = 0.005
        #     self.MOTION_MODEL_ANG_NOISE = 0.05 * 2 # original value = 0.05/0.5 
        #     # self.MASS_MEAN_list = [0.1] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 2
        # elif self.MASS_marker == 'mD' or self.MASS_marker == 'mDN' or self.MASS_marker == 'mDNN':
        #     self.MOTION_MODEL_POS_NOISE = 0.005 * 4 # original value = 0.005
        #     self.MOTION_MODEL_ANG_NOISE = 0.05 * 4 # original value = 0.05/0.5 
        #     # self.MASS_MEAN_list = [0.5] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 3
        # elif self.MASS_marker == 'mE' or self.MASS_marker == 'mEN' or self.MASS_marker == 'mENN':
        #     self.MOTION_MODEL_POS_NOISE = 0.005 * 10 # original value = 0.005
        #     self.MOTION_MODEL_ANG_NOISE = 0.05 * 10 # original value = 0.05/0.5 
        #     # self.MASS_MEAN_list = [1.0] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 4
        # elif self.MASS_marker == 'mF' or self.MASS_marker == 'mFN' or self.MASS_marker == 'mFNN':
        #     self.MOTION_MODEL_POS_NOISE = 0.005 * 20 # original value = 0.005
        #     self.MOTION_MODEL_ANG_NOISE = 0.05 * 20 # original value = 0.05/0.5 
        #     # self.MASS_MEAN_list = [5.0] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 5
        # elif self.MASS_marker == 'mG' or self.MASS_marker == 'mGN' or self.MASS_marker == 'mGNN':
        #     self.MOTION_MODEL_POS_NOISE = 0.005 * 40 # original value = 0.005
        #     self.MOTION_MODEL_ANG_NOISE = 0.05 * 40 # original value = 0.05/0.5 
        #     # self.MASS_MEAN_list = [10.0] * self.object_num
        #     # self.MASS_MEAN_list = self.MASS_MEAN_list * 10

        # if self.FRICTION_marker == 'fA' or self.FRICTION_marker == 'fAN' or self.FRICTION_marker == 'fANN':
        #     self.FRICTION_MEAN = 0.5
        # elif self.FRICTION_marker == 'fB' or self.FRICTION_marker == 'fBN' or self.FRICTION_marker == 'fBNN':
        #     self.FRICTION_MEAN = 0.66667
        # elif self.FRICTION_marker == 'fC' or self.FRICTION_marker == 'fCN' or self.FRICTION_marker == 'fCNN':
        #     self.FRICTION_MEAN = 0.83337
        # elif self.FRICTION_marker == 'fD' or self.FRICTION_marker == 'fDN' or self.FRICTION_marker == 'fDNN':
        #     self.FRICTION_MEAN = 1
        # elif self.FRICTION_marker == 'fE' or self.FRICTION_marker == 'fEN' or self.FRICTION_marker == 'fENN':
        #     self.FRICTION_MEAN = 0.1
        # elif self.FRICTION_marker == 'fF' or self.FRICTION_marker == 'fFN' or self.FRICTION_marker == 'fFNN':
        #     self.FRICTION_MEAN = 0.2
        # elif self.FRICTION_marker == 'fG' or self.FRICTION_marker == 'fGN' or self.FRICTION_marker == 'fGNN':
        #     self.FRICTION_MEAN = 0.3
        # elif self.FRICTION_marker == 'fH' or self.FRICTION_marker == 'fHN' or self.FRICTION_marker == 'fHNN':
        #     self.FRICTION_MEAN = 0.4

        if self.FRICTION_marker == 'fA' or self.FRICTION_marker == 'fAN' or self.FRICTION_marker == 'fANN':
            self.FRICTION_MEAN = 0.01 # 0.01
        elif self.FRICTION_marker == 'fB' or self.FRICTION_marker == 'fBN' or self.FRICTION_marker == 'fBNN':
            self.FRICTION_MEAN = 0.38
        elif self.FRICTION_marker == 'fC' or self.FRICTION_marker == 'fCN' or self.FRICTION_marker == 'fCNN':
            self.FRICTION_MEAN = 0.25
        elif self.FRICTION_marker == 'fD' or self.FRICTION_marker == 'fDN' or self.FRICTION_marker == 'fDNN':
            self.FRICTION_MEAN = 0.38
        elif self.FRICTION_marker == 'fE' or self.FRICTION_marker == 'fEN' or self.FRICTION_marker == 'fENN':
            self.FRICTION_MEAN = 0.5
        elif self.FRICTION_marker == 'fF' or self.FRICTION_marker == 'fFN' or self.FRICTION_marker == 'fFNN':
            self.FRICTION_MEAN = 0.75
        elif self.FRICTION_marker == 'fG' or self.FRICTION_marker == 'fGN' or self.FRICTION_marker == 'fGNN':
            self.FRICTION_MEAN = 1
        # elif self.FRICTION_marker == 'fG' or self.FRICTION_marker == 'fGN' or self.FRICTION_marker == 'fGNN':
        #     self.FRICTION_MEAN = 0.3
        # elif self.FRICTION_marker == 'fH' or self.FRICTION_marker == 'fHN' or self.FRICTION_marker == 'fHNN':
        #     self.FRICTION_MEAN = 0.4


        if self.MASS_NOISE == False:
            self.MASS_SIGMA_list = [0.0] * self.object_num
        if self.FRICTION_NOISE ==False:
            self.FRICTION_SIGMA = 0



        # self.MOTION_MODEL_ANG_NOISE = 0.0 # original value = 0.05/0.5 
        self.mass_flag = False
        if self.mass_flag == True:
            self.MASS_MIN_VALUE = 0.02
            for obj_num_index in range(self.object_num):
                if self.OBJECT_NAME_LIST[obj_num_index] == "cracker":
                    mass = 0.45
                    self.MASS_MEAN_list[obj_num_index] = mass
                    mass_sigma = 0.5
                    self.MASS_SIGMA_list[obj_num_index] = mass_sigma
                elif self.OBJECT_NAME_LIST[obj_num_index] == "Parmesan":
                    mass = 0.035
                    # mass = 100
                    self.MASS_MEAN_list[obj_num_index] = mass
                    mass_sigma = 0.1
                    self.MASS_SIGMA_list[obj_num_index] = mass_sigma
                elif self.OBJECT_NAME_LIST[obj_num_index] == "soup":
                    mass = 0.35
                    self.MASS_MEAN_list[obj_num_index] = mass
                    mass_sigma = 0.5
                    self.MASS_SIGMA_list[obj_num_index] = mass_sigma
                elif self.OBJECT_NAME_LIST[obj_num_index] == "Milk":
                    mass = 0.04
                    # mass = 100
                    self.MASS_MEAN_list[obj_num_index] = mass
                    mass_sigma = 0.1
                    self.MASS_SIGMA_list[obj_num_index] = mass_sigma
                elif self.OBJECT_NAME_LIST[obj_num_index] == "Mustard":
                    mass = 0.05
                    self.MASS_MEAN_list[obj_num_index] = mass
                    mass_sigma = 0.1
                    self.MASS_SIGMA_list[obj_num_index] = mass_sigma
                elif self.OBJECT_NAME_LIST[obj_num_index] == "Mayo":
                    mass = 0.06
                    self.MASS_MEAN_list[obj_num_index] = mass
                    mass_sigma = 0.1
                    self.MASS_SIGMA_list[obj_num_index] = mass_sigma
                elif self.OBJECT_NAME_LIST[obj_num_index] == "SaladDressing":
                    mass = 0.06
                    self.MASS_MEAN_list[obj_num_index] = mass
                    mass_sigma = 0.1
                    self.MASS_SIGMA_list[obj_num_index] = mass_sigma
                elif self.OBJECT_NAME_LIST[obj_num_index] == "Ketchup":
                    mass = 0.06
                    self.MASS_MEAN_list[obj_num_index] = mass
                    mass_sigma = 0.1
                    self.MASS_SIGMA_list[obj_num_index] = mass_sigma
                else:
                    mass = 1.5
                    self.MASS_MEAN_list[obj_num_index] = mass
                    mass_sigma = 0.1
                    self.MASS_SIGMA_list[obj_num_index] = mass_sigma
            
            # for index, name in enumerate(self.OBJECT_NAME_LIST):
            #     if name == "cracker":
            #         mass = 1.0
            #         self.MASS_MEAN_list[index] = mass
            #         self.MOTION_MODEL_ANG_NOISE = 0.05 # original value = 0.05/0.5
            #     elif name == "Parmesan":
            #         mass = 1.0
            #         self.MASS_MEAN_list[index] = mass
            #         self.MOTION_MODEL_ANG_NOISE = 0.05 # original value = 0.05/0.5
            #     else:
            #         mass = 1.5
            #         self.MASS_MEAN_list[index] = mass
            #         self.MOTION_MODEL_ANG_NOISE = 0.05 # original value = 0.05


        # Observation Model
        self.OBS_SIGMA_POS_FOR_WEIGHT = 0.1
        for name in self.OBJECT_NAME_LIST:
            if name == "cracker":
                self.OBS_SIGMA_ANG_FOR_WEIGHT = 0.0216773873 * 10 # 30
            else:
                self.OBS_SIGMA_ANG_FOR_WEIGHT = 0.0216773873 * 10


    def run(self):
        # This is needed due to how multiprocessing works which will fork the
        # main process, including the seed for numpy random library and as a result, 
        # if the seed is not re-generated in each process, 
        # each process will generate the same noisy trajectory.
        np.random.seed()
        self.init_pybullet()
        while True:
            with self.lock:
                if not self.queue.empty():
                    method, *args = self.queue.get()
                    result = method(self, *args)
                    for key, value in result:
                        self.result[key] = value
            time.sleep(0.00001)
    
    def dummy(self):
        return [('success', True)]

    def init_pybullet(self):
        if self.SHOW_RAY == True:
            self.p_env = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT,GUI_SERVER
        else:
            self.p_env = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT,GUI_SERVER
        if self.update_style_flag == "time":
            self.p_env.setTimeStep(self.sim_time_step)
        self.p_env.resetDebugVisualizerCamera(cameraDistance=1., cameraYaw=90, cameraPitch=-50, cameraTargetPosition=[0.1,0.15,0.35])  
        self.p_env.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p_env.setGravity(0, 0, -9.81)
        self.p_env.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        
        self.add_robot()
        self.add_static_obstacles()
        self.add_target_objects()

    def add_static_obstacles(self):
        plane_id = self.p_env.loadURDF("plane.urdf")
        if self.task_flag == "1":
            pw_T_pringles_pos = [0.6652218209791124, 0.058946644391304814, 0.8277292172960276]
            pw_T_pringles_ori = [ 0.67280124, -0.20574896, -0.20600051, 0.68012472] # x, y, z, w
            pringles_id = self.p_env.loadURDF(os.path.expanduser("~/project/object/others/pringles.urdf"),
                                              pw_T_pringles_pos, pw_T_pringles_ori, useFixedBase=1)
        if self.task_flag == "4": # slope
            pw_T_Milk_pos = [0.5255412218811237, 0.4092688983400049+0.13, 0.8156348920165202-2*0.0358583]
            pw_T_Milk_ori = [ 0.71226091, -0.00120944, -0.00472836,  0.70189783]
            base_Milk1_id = self.p_env.loadURDF(os.path.expanduser("~/project/object/Milk/Milk_par_no_visual_hor.urdf"),
                                               pw_T_Milk_pos, pw_T_Milk_ori, useFixedBase=1)
            pw_T_Milk_pos = [0.5255412218811237, 0.4092688983400049+0.13, 0.8156348920165202-2*0.0358583]
            pw_T_Milk_ori = [ 0.71226091, -0.00120944, -0.00472836,  0.70189783]
            base_Milk2_id = self.p_env.loadURDF(os.path.expanduser("~/project/object/Milk/Milk_par_no_visual_hor.urdf"),
                                               pw_T_Milk_pos, pw_T_Milk_ori, useFixedBase=1)
            board_pos_4 = [0.5255245316420766, 0.08585146275273556+0.13, 0.7858109052752488]
            board_ori_4 = [0.0921379328294458, -1.0388626282143925e-05, -0.00014271076727580726, 0.9957462432063853]
            self.board_id_4 = self.p_env.loadURDF(os.path.expanduser("~/project/object/others/board.urdf"),
                                                     board_pos_4, 
                                                     board_ori_4,
                                                     useFixedBase = 1)
        

            # self.p_env.changeDynamics(board_id_4, -1, mass = 5, 
            #                       lateralFriction = 1)

            self.collision_detection_obj_id_collection.append(self.board_id_4)
        

        if self.task_flag == "6":
            pw_T_Milk_opti_pos = [0.5639079993582834, 0.06686931205630225, 0.7947410960108179]
            pw_T_Milk_opti_ori = [-0.61877113,  0.33879951,  0.61984684,  0.3436962 ]
            obstacle_Milk_id = self.p_env.loadURDF(os.path.expanduser("~/project/object/Milk/Milk_real_obj_with_visual_hor.urdf"),
                                                  pw_T_Milk_opti_pos,
                                                  pw_T_Milk_opti_ori,
                                                  useFixedBase=1)


        if self.SIM_REAL_WORLD_FLAG == True:
            table_pos_1 = [0.46, -0.01, 0.702] # 0.710
            table_ori_1 = self.p_env.getQuaternionFromEuler([0,0,0])
            self.table_id_1 = self.p_env.loadURDF(os.path.expanduser("~/project/object/others/table.urdf"), table_pos_1, table_ori_1, useFixedBase = 1)


            barry_pos_1 = [-0.694, 0.443, 0.895]
            barry_ori_1 = self.p_env.getQuaternionFromEuler([0,math.pi/2,0])
            barry_id_1 = self.p_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_1, barry_ori_1, useFixedBase = 1)
            
            barry_pos_2 = [-0.694, -0.607, 0.895]
            barry_ori_2 = self.p_env.getQuaternionFromEuler([0,math.pi/2,0])
            barry_id_2 = self.p_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_2, barry_ori_2, useFixedBase = 1)

            barry_pos_3 = [0.459, -0.972, 0.895]
            barry_ori_3 = self.p_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
            barry_id_3 = self.p_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_3, barry_ori_3, useFixedBase = 1)

            # barry_pos_4 = [-0.549, 0.61, 0.895]
            # barry_ori_4 = self.p_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
            # barry_id_4 = self.p_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_4, barry_ori_4, useFixedBase = 1)
            
            # barry_pos_5 = [0.499, 0.61, 0.895]
            # barry_ori_5 = self.p_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
            # barry_id_5 = self.p_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_5, barry_ori_5, useFixedBase = 1)

            if self.task_flag != "4": # slope
                board_pos_1 = [0.274, 0.581, 0.87575]
                board_ori_1 = self.p_env.getQuaternionFromEuler([math.pi/2,math.pi/2,0])
                self.board_id_1 = self.p_env.loadURDF(os.path.expanduser("~/project/object/others/board.urdf"), board_pos_1, board_ori_1, useFixedBase = 1)
                self.collision_detection_obj_id_collection.append(self.board_id_1)

    def add_robot(self):
        real_robot_start_pos = self.pw_T_rob_sim_pose_list_alg[0].pos
        real_robot_start_ori = self.pw_T_rob_sim_pose_list_alg[0].ori
        joint_of_robot = self.pw_T_rob_sim_pose_list_alg[0].joints
        self.robot_id = self.p_env.loadURDF(os.path.expanduser("~/project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                            real_robot_start_pos, real_robot_start_ori, useFixedBase=1)
        self.init_set_sim_robot_JointPosition(joint_of_robot)
        self.collision_detection_obj_id_collection.append(self.robot_id)

    def add_target_objects(self):
        print_object_name_flag = 0
        for obj_index in range(self.object_num):
            obj_obse_pos = self.pw_T_obj_obse_obj_list_alg[obj_index].pos
            obj_obse_ori = self.pw_T_obj_obse_obj_list_alg[obj_index].ori
            obj_obse_name = self.pw_T_obj_obse_obj_list_alg[obj_index].obj_name
            if print_object_name_flag == obj_index:
                print_object_name_flag = print_object_name_flag + 1
            particle_pos, particle_ori = self.generate_random_pose(obj_obse_pos, obj_obse_ori)
            gazebo_contain = ""
            if self.gazebo_flag == True:
                gazebo_contain = "gazebo_"
            if obj_obse_name == "soup2":
                obj_obse_name = "soup"
            particle_no_visual_id = self.p_env.loadURDF(os.path.expanduser("~/project/object/"+gazebo_contain+obj_obse_name+"/"+gazebo_contain+obj_obse_name+"_par_no_visual_hor.urdf"),
                                                        particle_pos, particle_ori)
            self.collision_detection_obj_id_collection.append(particle_no_visual_id)
            self.particle_objects_id_collection[obj_index] = particle_no_visual_id

            conter = 0
            while True:
                flag = 0
                conter = conter + 1
                length_collision_detection_obj_id = len(self.collision_detection_obj_id_collection)
                for check_num in range(length_collision_detection_obj_id-1):
                    self.p_env.stepSimulation()
                    contacts = self.p_env.getContactPoints(bodyA=self.collision_detection_obj_id_collection[check_num], 
                                                           bodyB=self.collision_detection_obj_id_collection[-1])
                    for contact in contacts:
                        contactNormalOnBtoA = contact[7]
                        contact_dis = contact[8]
                        if contact_dis < -0.001:
                            par_x_ = particle_pos[0] + contactNormalOnBtoA[0]*contact_dis/2
                            par_y_ = particle_pos[1] + contactNormalOnBtoA[1]*contact_dis/2
                            par_z_ = particle_pos[2] + contactNormalOnBtoA[2]*contact_dis/2
                            particle_pos = [par_x_, par_y_, par_z_]
                            if conter > 20:
                                print("init more than 20 times")
                                conter = 0
                                particle_pos, particle_ori = self.generate_random_pose(obj_obse_pos, obj_obse_ori)
                            self.p_env.resetBasePositionAndOrientation(particle_no_visual_id, particle_pos, particle_ori)
                            flag = 1
                            break
                    if flag == 1:
                        break
                if flag == 0:
                    break
            objPose = Particle(obj_obse_name, 0, particle_no_visual_id, particle_pos, particle_ori, 1/self.particle_num, 0, 0, 0)
            self.objects_list[obj_index] = objPose

    def get_objects_pose(self, par_index):
        # for time_index in range(int(self.pf_update_interval_in_sim)):
        #     self.p_env.stepSimulation()
        return_results = []
        for obj_index in range(self.object_num):
            obj_id = self.particle_objects_id_collection[obj_index]
            obj_info = self.p_env.getBasePositionAndOrientation(obj_id)    
            obj_name = self.OBJECT_NAME_LIST[obj_index]
            obj_tuple = (obj_name, obj_info)
            return_results.append(obj_tuple)
            pos_ = obj_info[0]
            ori_ = obj_info[1]
            self.objects_list[obj_index].pos = [pos_[0], pos_[1], pos_[2]]
            self.objects_list[obj_index].ori = [ori_[0], ori_[1], ori_[2], ori_[3]] # x, y, z, w
        return_results.append((str(par_index), self.objects_list))
        # self.p_env.disconnect()
        return return_results

    def isAnyParticleInContact(self):
        for obj_index in range(self.object_num):
            # get object ID
            obj_id = self.particle_objects_id_collection[obj_index]
            # check contact 
            pmin, pmax = self.p_env.getAABB(obj_id)
            collide_ids = self.p_env.getOverlappingObjects(pmin, pmax)
            length = len(collide_ids)
            for t_i in range(length):
                if collide_ids[t_i][1] == 8 or collide_ids[t_i][1] == 9 or collide_ids[t_i][1] == 10 or collide_ids[t_i][1] == 11:
                    return [('result', True)]
        return [('result', False)]

    def motion_model(self, joint_states, par_index):
        # change object parameters
        collision_detection_obj_id_ = []
        return_results = []
        for obj_index in range(self.object_num):
            obj_id = self.objects_list[obj_index].no_visual_par_id
            # self.p_env.resetBaseVelocity(obj_id, 0, 0)
            self.p_env.resetBaseVelocity(obj_id,
                                         self.objects_list[obj_index].linearVelocity,
                                         self.objects_list[obj_index].angularVelocity,)
            self.change_obj_parameters(obj_id, obj_index)
        # execute the control
        self.move_robot_JointPosition(joint_states)
        # collision check: add robot
        collision_detection_obj_id_.append(self.robot_id)
        # collision check: add board
        if self.task_flag != "4": # slope 
            collision_detection_obj_id_.append(self.board_id_1)
        # collision check
        for obj_index in range(self.object_num):
            obj_id = self.objects_list[obj_index].no_visual_par_id
            # get linearVelocity and angularVelocity of the object from each particle
            linearVelocity, angularVelocity = self.p_env.getBaseVelocity(obj_id)
            obj_cur_pos, obj_cur_ori = self.get_item_pos(obj_id)
            normal_x = obj_cur_pos[0]
            normal_y = obj_cur_pos[1]
            normal_z = obj_cur_pos[2]
            pb_quat = obj_cur_ori
            # add noise on pose of each particle
            if self.MOTION_NOISE == True:
                normal_x, normal_y, normal_z, pb_quat = self.add_noise_pose(obj_cur_pos, obj_cur_ori)
                self.p_env.resetBasePositionAndOrientation(obj_id, [normal_x, normal_y, normal_z], pb_quat)
                collision_detection_obj_id_.append(obj_id)
                obj_pose_3_1 = [normal_x, normal_y, normal_z, pb_quat]
                normal_x, normal_y, normal_z, pb_quat = self.collision_check(collision_detection_obj_id_,
                                                                             obj_cur_pos, obj_cur_ori,
                                                                             obj_id, obj_index, obj_pose_3_1)

            if obj_index == 0:
                normal_x = normal_x - 0.001
                normal_y = normal_y + 0.000
                normal_z = normal_z + 0.0000
            elif obj_index == 1:
                normal_x = normal_x + 0.002
                normal_y = normal_y - 0.0000
            elif obj_index == 2:
                normal_x = normal_x - 0.002
                normal_y = normal_y + 0.000
            elif obj_index == 3:
                normal_x = normal_x - 0.002
                normal_y = normal_y + 0.000
            elif obj_index == 4:
                normal_x = normal_x - 0.005
                normal_y = normal_y + 0.002

            # if obj_index == 0:
            #     normal_x = normal_x - 0.0005
            #     normal_y = normal_y - 0.0000
            #     normal_z = normal_z + 0.0000
            # elif obj_index == 1:
            #     normal_x = normal_x - 0.0002
            #     normal_y = normal_y - 0.0002
            # elif obj_index == 2:
            #     normal_x = normal_x - 0.0000
            # elif obj_index == 3:
            #     normal_x = normal_x - 0.0000
            # elif obj_index == 4:
            #     normal_x = normal_x - 0.0000

            self.update_object_pose_PB(obj_index, normal_x, normal_y, normal_z, pb_quat, linearVelocity, angularVelocity)
        self.p_env.stepSimulation()
        return_results = self.get_objects_pose(par_index)
        return return_results


    def init_set_sim_robot_JointPosition(self, joint_states):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                self.p_env.resetJointState(self.robot_id,
                                           joint_index+2,
                                           targetValue=joint_states[joint_index])
            else:
                self.p_env.resetJointState(self.robot_id,
                                           joint_index,
                                           targetValue=joint_states[joint_index])

    def move_robot_JointPosition(self, joint_states):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                self.p_env.setJointMotorControl2(self.robot_id, joint_index+2,
                                                 self.p_env.POSITION_CONTROL,
                                                 targetPosition=joint_states[joint_index])
            else:
                self.p_env.setJointMotorControl2(self.robot_id, joint_index,
                                                 self.p_env.POSITION_CONTROL,
                                                 targetPosition=joint_states[joint_index])
        for time_index in range(int(self.pf_update_interval_in_sim)):
            self.p_env.stepSimulation()
        return [("done", True)]

    def compare_distance(self, par_index, pw_T_obj_obse_objects_pose_list, visual_by_DOPE_list, outlier_by_DOPE_list):
        weight =  1.0 / self.particle_num
        weights_list = [weight] * self.object_num
        for obj_index in range(self.object_num):
            self.objects_list[obj_index].w = weight
        # at least one object is detected by camera
        if (sum(visual_by_DOPE_list)<self.object_num) and (sum(outlier_by_DOPE_list)<self.object_num):
            for obj_index in range(self.object_num):
                weight =  1.0 / self.particle_num
                obj_visual = visual_by_DOPE_list[obj_index]
                obj_outlier = outlier_by_DOPE_list[obj_index]
                # obj_visual=0 means DOPE detects the object[obj_index]
                # obj_visual=1 means DOPE does not detect the object[obj_index] and skip this loop
                # obj_outlier=0 means DOPE detects the object[obj_index]
                # obj_outlier=1 means DOPE detects the object[obj_index], but we judge it is outlier and skip this loop
                if obj_visual==0 and obj_outlier==0:
                    obj_x = self.objects_list[obj_index].pos[0]
                    obj_y = self.objects_list[obj_index].pos[1]
                    obj_z = self.objects_list[obj_index].pos[2]
                    obj_ori = self.quaternion_correction(self.objects_list[obj_index].ori)
                    obse_obj_pos = pw_T_obj_obse_objects_pose_list[obj_index].pos
                    obse_obj_ori = pw_T_obj_obse_objects_pose_list[obj_index].ori # pybullet x,y,z,w
                    # make sure theta between -pi and pi
                    obse_obj_ori = self.quaternion_correction(obse_obj_ori)
                    mean = 0
                    # position weight
                    dis_x = abs(obj_x - obse_obj_pos[0])
                    dis_y = abs(obj_y - obse_obj_pos[1])
                    dis_z = abs(obj_z - obse_obj_pos[2])
                    dis_xyz = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
                    weight_xyz = self.normal_distribution(dis_xyz, mean, self.OBS_SIGMA_POS_FOR_WEIGHT)
                    # rotation weight
                    obse_obj_quat = Quaternion(x=obse_obj_ori[0], y=obse_obj_ori[1], z=obse_obj_ori[2], w=obse_obj_ori[3]) # Quaternion(): w,x,y,z
                    par_quat = Quaternion(x=obj_ori[0], y=obj_ori[1], z=obj_ori[2], w=obj_ori[3])
                    err_bt_par_obse = par_quat * obse_obj_quat.inverse
                    err_bt_par_obse_corr = self.quaternion_correction([err_bt_par_obse.x, err_bt_par_obse.y, err_bt_par_obse.z, err_bt_par_obse.w])
                    err_bt_par_obse_corr_quat = Quaternion(x=err_bt_par_obse_corr[0], y=err_bt_par_obse_corr[1], z=err_bt_par_obse_corr[2], w=err_bt_par_obse_corr[3]) # Quaternion(): w,x,y,z
                    cos_theta_over_2 = err_bt_par_obse_corr_quat.w
                    sin_theta_over_2 = math.sqrt(err_bt_par_obse_corr_quat.x ** 2 + err_bt_par_obse_corr_quat.y ** 2 + err_bt_par_obse_corr_quat.z ** 2)
                    theta_over_2 = math.atan2(sin_theta_over_2, cos_theta_over_2)
                    theta = theta_over_2 * 2.0
                    weight_ang = self.normal_distribution(theta, mean, self.OBS_SIGMA_ANG_FOR_WEIGHT)
                    weight = weight_xyz * weight_ang
                    self.objects_list[obj_index].w = weight
                    weights_list[obj_index] = weight
                else:
                    self.objects_list[obj_index].w = weight
                    weights_list[obj_index] = weight

        return [(str(par_index), weights_list)]

    def generate_random_pose(self, pw_T_obj_obse_pos, pw_T_obj_obse_ori):
        quat = pw_T_obj_obse_ori # x,y,z,w
        quat_QuatStyle = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3]) # w,x,y,z
        x = self.add_noise_to_init_par(pw_T_obj_obse_pos[0], self.boss_sigma_obs_x)
        y = self.add_noise_to_init_par(pw_T_obj_obse_pos[1], self.boss_sigma_obs_y)
        z = self.add_noise_to_init_par(pw_T_obj_obse_pos[2], self.boss_sigma_obs_z)
        random_dir = random.uniform(0, 2*math.pi)
        z_axis = random.uniform(-1,1)
        x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
        y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
        angle_noise = self.add_noise_to_init_par(0, self.boss_sigma_obs_ang_init)
        w_quat = math.cos(angle_noise/2.0)
        x_quat = math.sin(angle_noise/2.0) * x_axis
        y_quat = math.sin(angle_noise/2.0) * y_axis
        z_quat = math.sin(angle_noise/2.0) * z_axis
        ###nois_quat(w,x,y,z); new_quat(w,x,y,z)
        nois_quat = Quaternion(x=x_quat, y=y_quat, z=z_quat, w=w_quat)
        new_quat = nois_quat * quat_QuatStyle
        ###pb_quat(x,y,z,w)
        pb_quat = [new_quat[1], new_quat[2], new_quat[3], new_quat[0]]
        return [x, y, z], pb_quat

    def set_particle_in_each_sim_env(self, single_particle):
        for obj_index in range(self.object_num):
            x = single_particle[obj_index].pos[0]
            y = single_particle[obj_index].pos[1]
            z = single_particle[obj_index].pos[2]
            pb_quat = single_particle[obj_index].ori
            linearVelocity = single_particle[obj_index].linearVelocity
            angularVelocity = single_particle[obj_index].angularVelocity
            self.update_object_pose_PB(obj_index, x, y, z, pb_quat, linearVelocity, angularVelocity)
        return [("done", True)]

    def update_object_pose_PB(self, obj_index, x, y, z, pb_quat, linearVelocity, angularVelocity):
        self.objects_list[obj_index].pos = [x, y, z]
        self.objects_list[obj_index].ori = pb_quat
        self.objects_list[obj_index].linearVelocity = linearVelocity
        self.objects_list[obj_index].angularVelocity = angularVelocity
        obj_id = self.objects_list[obj_index].no_visual_par_id
        self.p_env.resetBasePositionAndOrientation(obj_id, [x, y, z], pb_quat)

    def add_noise_to_init_par(self, current_pos, sigma_init):
        mean = current_pos
        sigma = sigma_init
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def take_easy_gaussian_value(self, mean, sigma):
        normal = random.normalvariate(mean, sigma)
        return normal

    def get_item_pos(self, item_id):
        item_info = self.p_env.getBasePositionAndOrientation(item_id)
        return item_info[0], item_info[1]

    def getLinkStates(self):
        all_links_info = self.p_env.getLinkStates(self.robot_id, range(self.PANDA_ROBOT_LINK_NUMBER + 2), computeForwardKinematics=True) # 11+2; range: [0,13)
        all_links_info = all_links_info[:12]
        return [("links_info", all_links_info)]

    # add noise
    def add_noise_pose(self, obj_cur_pos, obj_cur_ori): # obj_cur_pos: x,y,z; obj_cur_ori: x,y,z,w
        # add noise to pos of object
        normal_x = self.add_noise_2_par(obj_cur_pos[0])
        normal_y = self.add_noise_2_par(obj_cur_pos[1])
        normal_z = self.add_noise_2_par(obj_cur_pos[2])
        # add noise to ang of object
        quat_QuatStyle = Quaternion(x=obj_cur_ori[0], y=obj_cur_ori[1], z=obj_cur_ori[2], w=obj_cur_ori[3])# w,x,y,z
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
        ###pb_quat(x,y,z,w); pb_quat(x,y,z,w)
        pb_quat = [new_quat[1],new_quat[2],new_quat[3],new_quat[0]]
        new_angle = p.getEulerFromQuaternion(pb_quat)
        pb_quat = p.getQuaternionFromEuler(new_angle)
        # pipe.send()
        return normal_x, normal_y, normal_z, pb_quat

    def add_noise_2_par(self, current_pos):
        mean = current_pos
        sigma = self.MOTION_MODEL_POS_NOISE
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise

    def add_noise_2_ang(self, cur_angle):
        mean = cur_angle
        sigma = self.MOTION_MODEL_ANG_NOISE
        new_ang_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_ang_is_added_noise
    
    # make sure all quaternions all between -pi and +pi
    def quaternion_correction(self, quaternion): # x,y,z,w
        new_quat = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]) # w,x,y,z
        cos_theta_over_2 = new_quat.w
        sin_theta_over_2 = math.sqrt(new_quat.x ** 2 + new_quat.y ** 2 + new_quat.z ** 2)
        theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
        theta = theta_over_2 * 2.0
        while theta >= math.pi:
            theta = theta - 2.0*math.pi
        while theta <= -math.pi:
            theta = theta + 2.0*math.pi
        new_quaternion = [math.sin(theta/2.0)*(new_quat.x/sin_theta_over_2), math.sin(theta/2.0)*(new_quat.y/sin_theta_over_2), math.sin(theta/2.0)*(new_quat.z/sin_theta_over_2), math.cos(theta/2.0)]
        return new_quaternion

    def normal_distribution(self, x, mean, sigma):
        return sigma * np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)
    

    def collision_check(self, collision_detection_obj_id_, obj_cur_pos, obj_cur_ori, obj_id, obj_index, obj_pose_3_1):
        normal_x = obj_pose_3_1[0]
        normal_y = obj_pose_3_1[1]
        normal_z = obj_pose_3_1[2]
        pb_quat = obj_pose_3_1[3]
        nTries = 0
        collision_id_length = len(collision_detection_obj_id_)
        while nTries < 20:
            nTries = nTries + 1
            flag = 0
            for check_num in range(collision_id_length-1):
                self.p_env.stepSimulation()
                # will return all collision points
                contacts = self.p_env.getContactPoints(bodyA=collision_detection_obj_id_[check_num], # robot, other object...
                                                       bodyB=collision_detection_obj_id_[-1]) # main(target) object
                for contact in contacts:
                    contactNormalOnBtoA = contact[7]
                    contact_dis = contact[8]
                    # if contact_dis < -0.001: # means: positive for separation, negative for penetration
                    # if contact_dis < -0.02: # means: positive for separation, negative for penetration
                    if contact_dis < -0.0005: # means: positive for separation, negative for penetration
                        normal_x, normal_y, normal_z, pb_quat = self.add_noise_pose(obj_cur_pos, obj_cur_ori)
                        self.p_env.resetBasePositionAndOrientation(obj_id, [normal_x, normal_y, normal_z], pb_quat)
                        flag = 1
                        break
                if flag == 1:
                    break
            if flag == 0:
                break
        if nTries >= 20:
            print("WARNING: Could not find a non-colliding pose after motion noise. Moving particle object to noise-less pose.")
            self.p_env.resetBasePositionAndOrientation(obj_id, obj_cur_pos, obj_cur_ori)
        return normal_x, normal_y, normal_z, pb_quat


    # change particle parameters
    def change_obj_parameters(self, obj_id, obj_index):
        # mass_a = self.take_easy_gaussian_value(self.MASS_MEAN, self.MASS_SIGMA)
        mass_a = self.take_easy_gaussian_value(self.MASS_MEAN_list[obj_index], self.MASS_SIGMA_list[obj_index])
        if mass_a < 0.001:
            mass_a = self.MASS_MIN_VALUE
        lateralFriction = self.take_easy_gaussian_value(self.FRICTION_MEAN, self.FRICTION_SIGMA)
        spinningFriction = self.take_easy_gaussian_value(self.FRICTION_MEAN, self.FRICTION_SIGMA)
        rollingFriction = self.take_easy_gaussian_value(self.FRICTION_MEAN, self.FRICTION_SIGMA)
        if lateralFriction < 0.001:
            lateralFriction = 0.001
        if spinningFriction < 0.001:
            spinningFriction = 0.001
        if rollingFriction < 0.001:
            rollingFriction = 0.001
        restitution = self.take_easy_gaussian_value(self.RESTITUTION_MEAN, self.RESTITUTION_SIGMA)
        self.p_env.changeDynamics(obj_id, -1, mass = mass_a, 
                                  lateralFriction = lateralFriction, 
                                  spinningFriction = spinningFriction, 
                                  rollingFriction = rollingFriction, 
                                  restitution = restitution)

        if self.task_flag == "4": # slope
            self.p_env.changeDynamics(self.board_id_4, 0, 
                                    lateralFriction = lateralFriction, 
                                    spinningFriction = spinningFriction, 
                                    rollingFriction = 0.001, 
                                    restitution = restitution)
        else:
            self.p_env.changeDynamics(self.table_id_1, 0, 
                                    lateralFriction = lateralFriction, 
                                    spinningFriction = spinningFriction, 
                                    rollingFriction = 0.001, 
                                    restitution = restitution)
        self.p_env.changeDynamics(self.robot_id, 10, 
                                lateralFriction = lateralFriction, 
                                spinningFriction = spinningFriction, 
                                rollingFriction = 0.001, 
                                restitution = restitution)
        self.p_env.changeDynamics(self.robot_id, 11, 
                                lateralFriction = lateralFriction, 
                                spinningFriction = spinningFriction, 
                                rollingFriction = 0.001, 
                                restitution = restitution)