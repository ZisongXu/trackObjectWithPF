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
from scipy.spatial.transform import Rotation as R
#from sksurgerycore.algorithms.averagequaternions import average_quaternions
from quaternion_averaging import weightedAverageQuaternions
from Particle import Particle
from Object_Pose import Object_Pose
import yaml


class InitLargeNumPar():
    def __init__(self):
        with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
            self.parameter_info = yaml.safe_load(file)
        self.gazebo_flag = self.parameter_info['gazebo_flag']
        self.task_flag = self.parameter_info['task_flag'] # '1', '2', '3', 'basket_retrieve'
        self.SIM_REAL_WORLD_FLAG = self.parameter_info['sim_real_world_flag']
        self.SHOW_PARTICLE = self.parameter_info['show_particle'] 
        self.RENDER_DEPTH_SOFTWARE = self.parameter_info['render_depth_software'] # vk/pd
        self.OBJS_ARE_NOT_TOUCHING_TARGET_OBJS_NUM = self.parameter_info['objs_are_not_touching_target_objs_num']
        self.OBJS_TOUCHING_TARGET_OBJS_NUM = self.parameter_info['objs_touching_target_objs_num']
        ## object name list
        self.OBJECT_NAME_LIST = self.parameter_info['object_name_list']
        self.OBJECT_DETECTED_LIST = self.parameter_info['object_detected_list']
        self.UNSEEN_OBJECT_LIST = list(set(self.OBJECT_NAME_LIST) - set(self.OBJECT_DETECTED_LIST))
        # ================================================================================================================================================================
        self.OBJECT_NUM = self.parameter_info['object_num']
        self.PANDA_ROBOT_LINK_NUMBER = self.parameter_info['panda_robot_link_number']
        self.LOCATE_CAMERA_FLAG = self.parameter_info['locate_camera_flag'] # 'ar', 'opti', 'onTheHolder'
        self.ROBOT_END_EFFECTOR = self.parameter_info['robot_end_effector'] # 'gripper', 'pump', 'pump_with_extention'
        self.TASK_FLAG = self.parameter_info['task_flag'] # '1', '2', '3', 'basket_retrieve'
        self.INCREMENTAL_POSE_GENERATOR_FLAG = self.parameter_info['Incremental_Pose_Generator_Flag']
        self.INIT_METHOD = self.parameter_info['init_method'] # ViDe/Vi/De/normal...
        # particles number
        self.PARTICLE_NUM = self.parameter_info['particle_num'] # small
        self.PARTICLE_NUM_FOR_OBS = self.parameter_info['particle_num_for_obs'] # large
        self.PICK_PARTICLE_RATE_BEFORE_OBSMODEL = self.parameter_info['pick_particle_rate_before_obsModel'] # large
        ## noise for initialization
        self.BOSS_SIGMA_OBS_POS_INIT = self.parameter_info['boss_sigma_obs_pos_init'] # original value: 16cm/10CM/5cm 
        self.BOSS_SIGMA_OBS_X = self.BOSS_SIGMA_OBS_POS_INIT / math.sqrt(2)
        self.BOSS_SIGMA_OBS_Y = self.BOSS_SIGMA_OBS_POS_INIT / math.sqrt(2)
        self.BOSS_SIGMA_OBS_Z = self.parameter_info['boss_sigma_obs_z'] # original value: 2cm/1cm
        self.BOSS_SIGMA_OBS_ANG_INIT = self.parameter_info['boss_sigma_obs_ang_init'] # original value: 0.0216773873 * 20/10
        ## mark
        # self.BOSS_SIGMA_OBS_X = 0
        # self.BOSS_SIGMA_OBS_Y = 0
        # self.BOSS_SIGMA_OBS_Z = 0
        # self.BOSS_SIGMA_OBS_ANG_INIT = 0

    def passing_data(self, pw_T_obj_obse_obj_list_alg=0, pw_T_basket_pose=0):
        ## passing self.data
        self.pw_T_obj_obse_obj_list_alg = pw_T_obj_obse_obj_list_alg
        if self.task_flag == "basket_retrieve":
            self.pw_T_basket_pose = pw_T_basket_pose
            self.pw_T_basket_pos = self.get_position_from_matrix44(self.pw_T_basket_pose)
            self.pw_T_basket_ori = self.get_quaternion_from_matrix(self.pw_T_basket_pose)
        else:
            pass
        ## create new self.data
        self.particle_cloud = [0] * self.PARTICLE_NUM_FOR_OBS
        self.BOSS_SIGMA_OBS_Z_ANG_INIT = 2 * math.pi
        # Motion Model Noise
        self.MOTION_MODEL_POS_NOISE = 0.001/2 # original value = 0.005
        self.MOTION_MODEL_ANG_NOISE = 0.1/2 # original value = 0.05/0.5/0.1
        # self.MOTION_MODEL_POS_NOISE = 0.0 # original value = 0.005
        # self.MOTION_MODEL_ANG_NOISE = 0.0 # original value = 0.05/0.5/0.1
        
    def init_particle_cloud(self):
        if self.task_flag == "basket_retrieve":
            for par_index in range(self.PARTICLE_NUM_FOR_OBS):
                objects_list = ["None"] * len(self.OBJECT_NAME_LIST)
                for obj_index in range(len(self.OBJECT_NAME_LIST)):
                    obj_name = self.pw_T_obj_obse_obj_list_alg[obj_index].obj_name
                    pw_T_obj_obse_pos = self.pw_T_obj_obse_obj_list_alg[obj_index].pos 
                    pw_T_obj_obse_ori = self.pw_T_obj_obse_obj_list_alg[obj_index].ori
                    if obj_name in self.OBJECT_DETECTED_LIST:
                        particle_pos, particle_ori = self.generate_random_pose(pw_T_obj_obse_pos, pw_T_obj_obse_ori)
                        objInfo = Particle(obj_name, 0, 0, particle_pos, particle_ori, 1.0/self.PARTICLE_NUM_FOR_OBS, par_index, obj_index, 0, 0)
                    elif obj_name in self.UNSEEN_OBJECT_LIST:
                        objInfo = 0
                        if self.task_flag == "basket_retrieve": # we need to know the area that hold the unseen object
                            basket_width, basket_lenght, basket_height = self.get_object_shape("basket")
                            x_w,y_l,z_h = 0.0,0.0,0.0
                            pw_T_basketCenter_pos_x = self.pw_T_basket_pos[0]
                            pw_T_basketCenter_pos_y = self.pw_T_basket_pos[1]
                            pw_T_basketCenter_pos_z = self.pw_T_basket_pos[2] + basket_height/2.0
                            pw_T_basketCenter_pos = [pw_T_basketCenter_pos_x, pw_T_basketCenter_pos_y, pw_T_basketCenter_pos_z]
                            BasketCenter_T_points_pose_4_4_list = self.getCenterTPointsList("basket")
                            x_w, y_l, z_h = self.get_object_shape(obj_name)
                            pw_T_randomPoint_pose_results = self.generate_points_in_rotated_rectangular_prism(self.pw_T_basket_pose, BasketCenter_T_points_pose_4_4_list, x_w, y_l, z_h, num_points=1)
                            ## mark hard code
                            pw_T_randomPoint_pose_result = pw_T_randomPoint_pose_results[0]
                            particle_pos = pw_T_randomPoint_pose_result['pos']
                            particle_ori = pw_T_randomPoint_pose_result['ori']
                            # print(particle_pos)
                            objInfo = Particle(obj_name, 0, 0, particle_pos, particle_ori, 1.0/self.PARTICLE_NUM_FOR_OBS, par_index, obj_index, 0, 0)
                        else:
                            input("InitLargeNumPar.py: init_particle_cloud()")
                    objects_list[obj_index] = objInfo
                self.particle_cloud[par_index] = objects_list
        else:
            input("Have not done! Need to stop (InitLargeNumPar.py; init_particle_cloud().)")
        return self.particle_cloud
        
    def init_particle_cloud_part(self, pw_T_obj_obse_par_list_init, init_num_of_each_good_par_list):
        par_index_all = 0
        if self.task_flag == "basket_retrieve":
            # [143,143,143,142,143,143,143]/[143,142,143,143,143,143,143]
            for good_par_index in range(len(init_num_of_each_good_par_list)):
                par_num = init_num_of_each_good_par_list[good_par_index]
                for par_index in range(par_num):
                    objects_list = ["None"] * len(self.OBJECT_NAME_LIST)
                    for obj_index in range(len(self.OBJECT_NAME_LIST)):
                        obj_name = pw_T_obj_obse_par_list_init[good_par_index][obj_index].obj_name
                        pw_T_obj_obse_pos = pw_T_obj_obse_par_list_init[good_par_index][obj_index].pos 
                        pw_T_obj_obse_ori = pw_T_obj_obse_par_list_init[good_par_index][obj_index].ori
                        particle_pos, particle_ori = self.generate_random_pose(pw_T_obj_obse_pos, pw_T_obj_obse_ori)
                        # if obj_name in self.OBJECT_DETECTED_LIST:
                        #     particle_pos, particle_ori = self.generate_random_pose(pw_T_obj_obse_pos, pw_T_obj_obse_ori)
                        # elif obj_name in self.UNSEEN_OBJECT_LIST:
                        #     particle_pos, particle_ori = self.generate_random_pose_addZnoise(pw_T_obj_obse_pos, pw_T_obj_obse_ori)
                        objInfo = Particle(obj_name, 0, 0, particle_pos, particle_ori, 1.0/self.PARTICLE_NUM_FOR_OBS, par_index, obj_index, 0, 0)
                        objects_list[obj_index] = objInfo
                    self.particle_cloud[par_index_all] = objects_list
                    par_index_all = par_index_all + 1
        else:
            input("Have not done! Need to stop (InitLargeNumPar.py; init_particle_cloud_part().)")
        return self.particle_cloud

    def spread_particle(self, particle_cloud, pw_T_obj_obse_objects_pose_list):
        obs_particle_num = int(self.PARTICLE_NUM_FOR_OBS * self.PICK_PARTICLE_RATE_BEFORE_OBSMODEL)
        rest_particle_num = self.PARTICLE_NUM_FOR_OBS - obs_particle_num
        par_num_ = len(particle_cloud)
        # A list consisting of the number of particles assigned to each particle after expansion.
        # For example, if there are 200 particles in the motion model, 
        # the number of particles will be expanded to 1000, 
        # so that each particle will be assigned 4 additional particles: [5, 5, 5, 5, 5, ..., 5, 5, 5, 5]
        par_num_after_assign_list = self.divide_and_shuffle(rest_particle_num, par_num_)
        all_par_index = 0
        # particles around obsData
        for obs_par_index in range(obs_particle_num):
            objects_list = ["None"] * len(self.OBJECT_NAME_LIST)
            for obj_index in range(len(pw_T_obj_obse_objects_pose_list)):
                obj_pos = pw_T_obj_obse_objects_pose_list[obj_index].pos
                obj_ori = pw_T_obj_obse_objects_pose_list[obj_index].ori
                normal_x, normal_y, normal_z, pb_quat = self.add_noise_pose(obj_pos, obj_ori)
                objInfo = Particle(self.OBJECT_NAME_LIST[obj_index], 0, 0, [normal_x, normal_y, normal_z], pb_quat, 1.0/self.PARTICLE_NUM_FOR_OBS, all_par_index, obj_index, 0, 0)
                objects_list[obj_index] = objInfo
            self.particle_cloud[all_par_index] = objects_list
            all_par_index = all_par_index + 1
        # particles around physics particles
        # [5, 5, 5, 5, 5, ..., 5, 5, 5, 5]: 200 set
        for set_index in range(len(par_num_after_assign_list)):
            set_par_num = par_num_after_assign_list[set_index]
            for par_index in range(set_par_num):
                objects_list = ["None"] * len(self.OBJECT_NAME_LIST)
                for obj_index in range(len(pw_T_obj_obse_objects_pose_list)):
                    obj_pos = particle_cloud[set_index][obj_index].pos
                    obj_ori = particle_cloud[set_index][obj_index].ori
                    normal_x, normal_y, normal_z, pb_quat = self.add_noise_pose(obj_pos, obj_ori)
                    objInfo = Particle(self.OBJECT_NAME_LIST[obj_index], 0, 0, [normal_x, normal_y, normal_z], pb_quat, 1.0/self.PARTICLE_NUM_FOR_OBS, all_par_index, obj_index, 0, 0)
                    objects_list[obj_index] = objInfo
                self.particle_cloud[all_par_index] = objects_list
                all_par_index = all_par_index + 1
        return self.particle_cloud

    def generate_random_pose(self, pw_T_obj_obse_pos, pw_T_obj_obse_ori):
        quat = pw_T_obj_obse_ori # x,y,z,w
        quat_QuatStyle = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3]) # w,x,y,z
        x = self.add_noise_to_init_par(pw_T_obj_obse_pos[0], self.BOSS_SIGMA_OBS_X)
        y = self.add_noise_to_init_par(pw_T_obj_obse_pos[1], self.BOSS_SIGMA_OBS_Y)
        z = self.add_noise_to_init_par(pw_T_obj_obse_pos[2], self.BOSS_SIGMA_OBS_Z)
        random_dir = random.uniform(0, 2*math.pi)
        z_axis = random.uniform(-1,1)
        x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
        y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
        angle_noise = self.add_noise_to_init_par(0, self.BOSS_SIGMA_OBS_ANG_INIT)
        w_quat = math.cos(angle_noise/2.0)
        x_quat = math.sin(angle_noise/2.0) * x_axis
        y_quat = math.sin(angle_noise/2.0) * y_axis
        z_quat = math.sin(angle_noise/2.0) * z_axis
        ###noise_quat(w,x,y,z); new_quat(w,x,y,z)
        noise_quat = Quaternion(x=x_quat, y=y_quat, z=z_quat, w=w_quat)
        new_quat = noise_quat * quat_QuatStyle
        ###pb_quat(x,y,z,w)
        pb_quat = [new_quat[1], new_quat[2], new_quat[3], new_quat[0]]
        return [x, y, z], pb_quat

    def generate_random_pose_addZnoise(self, pw_T_obj_obse_pos, pw_T_obj_obse_ori):
        quat = pw_T_obj_obse_ori # x,y,z,w
        quat_QuatStyle = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3]) # w,x,y,z
        x = self.add_noise_to_init_par(pw_T_obj_obse_pos[0], self.BOSS_SIGMA_OBS_X)
        y = self.add_noise_to_init_par(pw_T_obj_obse_pos[1], self.BOSS_SIGMA_OBS_Y)
        z = self.add_noise_to_init_par(pw_T_obj_obse_pos[2], self.BOSS_SIGMA_OBS_Z)
        random_dir = random.uniform(0, 2*math.pi)
        z_axis = random.uniform(-1,1)
        x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
        y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
        angle_noise = self.add_noise_to_init_par(0, self.BOSS_SIGMA_OBS_ANG_INIT)
        w_quat = math.cos(angle_noise/2.0)
        x_quat = math.sin(angle_noise/2.0) * x_axis
        y_quat = math.sin(angle_noise/2.0) * y_axis
        z_quat = math.sin(angle_noise/2.0) * z_axis
        ### noise_quat(w,x,y,z); new_quat(w,x,y,z)
        noise_quat = Quaternion(x=x_quat, y=y_quat, z=z_quat, w=w_quat)
        new_quat = noise_quat * quat_QuatStyle
        ## additional z noise
        z_angle_noise = self.add_noise_to_init_par(0, self.BOSS_SIGMA_OBS_Z_ANG_INIT)
        z_w = math.cos(z_angle_noise/2.0)
        z_z = math.sin(z_angle_noise/2.0)
        ### z_noise_quat(w,x,y,z); new_quat(w,x,y,z)
        z_noise_quat = Quaternion(x=0, y=0, z=z_z, w=z_w)
        new_quat = z_noise_quat * new_quat
        ### pb_quat(x,y,z,w)
        pb_quat = [new_quat[1], new_quat[2], new_quat[3], new_quat[0]]
        return [x, y, z], pb_quat
    
    def add_noise_to_init_par(self, current_pos, sigma_init):
        mean = current_pos
        sigma = sigma_init
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def get_position_from_matrix44(self, a_T_b_4_4):
        x = a_T_b_4_4[0][3]
        y = a_T_b_4_4[1][3]
        z = a_T_b_4_4[2][3]
        position = [x, y, z]
        return position

    # get quaternion from matrix
    def get_quaternion_from_matrix(self, a_T_b_4_4):
        rot_matrix = a_T_b_4_4[:3, :3]
        rotation = R.from_matrix(rot_matrix)
        quaternion = rotation.as_quat()
        return quaternion
    
    def getPwTPointsList(self, center_T_points_pose_4_4_list, pos, ori):
        pw_T_points_pose_4_4_list = []
        # pw_T_center_ori_3_3 = transformations.quaternion_matrix(ori)
        # pw_T_center_ori_4_4 = rotation_4_4_to_transformation_4_4(pw_T_center_ori_3_3, pos)
        pw_T_center_ori_3_3 = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        pw_T_center_ori_3_4 = np.c_[pw_T_center_ori_3_3, pos]  # Add position to create 3x4 matrix
        pw_T_center_ori_4_4 = np.r_[pw_T_center_ori_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
        # mark
        for index in range(len(center_T_points_pose_4_4_list)):
            center_T_p_4_4 = copy.deepcopy(center_T_points_pose_4_4_list[index])
            pw_T_p_4_4 = np.dot(pw_T_center_ori_4_4, center_T_p_4_4)
            pw_T_points_pose_4_4_list.append(pw_T_p_4_4)
        return pw_T_points_pose_4_4_list
    
    def generate_points_in_rotated_rectangular_prism(self, pw_T_basket_pose, BasketCenter_T_points_pose_4_4_list, x_w, y_l, z_h, num_points=1):
        """
        Randomly generates uniformly distributed points in the rotated rectangular space (excluding boundaries).
        Parameters:
        - self.pw_T_basket_pose: transformation matrix from world coordinate system to basket center (4x4)
        - BasketCenter_T_points_pose_4_4_list: local coordinates of 8 points (list of 4x4 matrices)
        - x_w, y_l, z_h: Minimum boundary values (width, length, height) for the exclusion boundaries of the basket.
        - num_points: number of points to generate, default is 1
        Returns:
        - transform_matrix: 4x4 transform matrix
        - pos: position coordinate (3,)
        - ori: quaternion (4,)
        """
        # Extract 8 points of the basket in the local coordinate system
        basket_points_local = np.array([pose[:3, 3] for pose in BasketCenter_T_points_pose_4_4_list])  # Extract the panning portion
        # Find the minimum and maximum points in the local coordinate system of the basket
        min_corner_local = np.min(basket_points_local, axis=0)
        max_corner_local = np.max(basket_points_local, axis=0)
        # computational boundary
        boundary_margin = (min(x_w, y_l, z_h)+0.01)/2.0
        effective_min_local = min_corner_local + boundary_margin
        effective_max_local = max_corner_local - boundary_margin
        # effective_min_local[1] = 0  # Top does not exclude borders
        # effective_max_local[2] = max_corner_local[2]  # Top does not exclude borders
        # print(effective_min_local)
        # print(effective_max_local)
        # Generating random points in a local coordinate system
        local_points = np.random.uniform(low=effective_min_local, high=effective_max_local, size=(num_points, 3))
        # Convert local points to the world coordinate system
        rotation_matrix = copy.deepcopy(pw_T_basket_pose[:3, :3])  # Rotated matrix of world coordinates to the center of the basket
        translation_vector = copy.deepcopy(pw_T_basket_pose[:3, 3])  # Translation vector from world coordinates to the center of the basket
        translation_vector[2] = translation_vector[2] + max_corner_local[2] # mark, do this because the coordinate center of the basket is not at the center of the basket
        world_points = np.dot(local_points, rotation_matrix.T) + translation_vector
        world_matrix = R.random().as_matrix()
        # Generate results
        results = []
        for i in range(num_points):
            pos = world_points[i]
            ori = R.from_matrix(world_matrix).as_quat()  # 提取四元数
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = pos
            results.append({
                "transform_matrix": transform_matrix,
                "pos": pos,
                "ori": ori
            })
        return results
    
    def get_object_shape(self, obj_name):
        if obj_name == "cracker":
            x_w = 0.159
            y_l = 0.21243700408935547
            z_h = 0.06
        elif obj_name == "Ketchup":
            x_w = 0.145
            y_l = 0.042
            z_h = 0.061
        elif obj_name == "Milk":
            x_w = 0.179934
            y_l = 0.0613
            z_h = 0.0613
        elif obj_name == "Mustard":
            x_w = 0.14
            y_l = 0.038
            z_h = 0.055
        elif obj_name == "Mayo":
            x_w = 0.1377716
            y_l = 0.0310130
            z_h = 0.054478
        elif obj_name == "Parmesan":
            x_w = 0.0929022
            y_l = 0.0592842
            z_h = 0.0592842
        elif obj_name == "SaladDressing":
            x_w = 0.1375274
            y_l = 0.036266
            z_h = 0.052722
        elif obj_name == "basket":
            x_w = 0.345
            y_l = 0.473
            z_h = 0.231
        elif obj_name == "soup":
            x_w = 0.032829689025878906
            y_l = 0.032829689025878906
            z_h = 0.099
        # else:
        #     x_w = 0.0851
        #     y_l = 0.0737
        #     z_h = 0.0279
        return x_w, y_l, z_h
    
    def getCenterTPointsList(self, object_name):
        center_T_points_pose_4_4_list = []
        # if object_name == "cracker" or object_name == "gelatin":
        if (object_name != "soup") and (object_name != "basket"):
            if object_name == "cracker":
                x_w = 0.159
                y_l = 0.21243700408935547
                z_h = 0.06
            elif object_name == "Ketchup":
                x_w = 0.145
                y_l = 0.042
                z_h = 0.061
            elif object_name == "Milk":
                x_w = 0.179934
                y_l = 0.0613
                z_h = 0.0613
            elif object_name == "Mustard":
                x_w = 0.14
                y_l = 0.038
                z_h = 0.055
            elif object_name == "Mayo":
                x_w = 0.1377716
                y_l = 0.0310130
                z_h = 0.054478
            elif object_name == "Parmesan":
                x_w = 0.0929022
                y_l = 0.0592842
                z_h = 0.0592842
            elif object_name == "SaladDressing":
                x_w = 0.1375274
                y_l = 0.036266
                z_h = 0.052722
            else:
                x_w = 0.0851
                y_l = 0.0737
                z_h = 0.0279
            vector_list = [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1], [1,0.5,0.5], [1,0.5,-0.5], [1,-0.5,0.5], [1,-0.5,-0.5], [-1,0.5,0.5], [-1,0.5,-0.5], [-1,-0.5,0.5], [-1,-0.5,-0.5], [0.5,1,0.5], [0.5,1,-0.5], [-0.5,1,0.5], [-0.5,1,-0.5], [0.5,-1,0.5], [0.5,-1,-0.5], [-0.5,-1,0.5], [-0.5,-1,-0.5], [0.5,0.5,1], [0.5,-0.5,1], [-0.5,0.5,1], [-0.5,-0.5,1], [0.5,0.5,-1], [0.5,-0.5,-1], [-0.5,0.5,-1], [-0.5,-0.5,-1]]
        elif object_name == "basket":
            x_w = 0.345
            y_l = 0.473
            z_h = 0.231
            vector_list = [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]]
        else:
            x_w = 0.032829689025878906
            y_l = 0.032829689025878906
            z_h = 0.099
            r = math.sqrt(2)
            vector_list = [[0,0,1], [0,0,-1],
                        [r,0,1], [0,r,1], [-r,0,1], [0,-r,1], [r,r,1], [r,-r,1], [-r,r,1], [-r,-r,1],
                        [r,0,0.5], [0,r,0.5], [-r,0,0.5], [0,-r,0.5], [r,r,0.5], [r,-r,0.5], [-r,r,0.5], [-r,-r,0.5],
                        [r,0,0], [0,r,0], [-r,0,0], [0,-r,0], [r,r,0], [r,-r,0], [-r,r,0], [-r,-r,0],
                        [r,0,-0.5], [0,r,-0.5], [-r,0,-0.5], [0,-r,-0.5], [r,r,-0.5], [r,-r,-0.5], [-r,r,-0.5], [-r,-r,-0.5],
                        [r,0,-1], [0,r,-1], [-r,0,-1], [0,-r,-1], [r,r,-1], [r,-r,-1], [-r,r,-1], [-r,-r,-1]]
        for index in range(len(vector_list)):
            center_T_p_x_new = vector_list[index][0] * x_w/2
            center_T_p_y_new = vector_list[index][1] * y_l/2
            center_T_p_z_new = vector_list[index][2] * z_h/2
            center_T_p_pos = [center_T_p_x_new, center_T_p_y_new, center_T_p_z_new]
            center_T_p_ori = [0, 0, 0, 1] # x, y, z, w
            # center_T_p_3_3 = transformations.quaternion_matrix(center_T_p_ori)
            # center_T_p_4_4 = rotation_4_4_to_transformation_4_4(center_T_p_3_3, center_T_p_pos)
            center_T_p_3_3 = np.array(p.getMatrixFromQuaternion(center_T_p_ori)).reshape(3, 3)
            center_T_p_3_4 = np.c_[center_T_p_3_3, center_T_p_pos]  # Add position to create 3x4 matrix
            center_T_p_4_4 = np.r_[center_T_p_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
            center_T_points_pose_4_4_list.append(center_T_p_4_4)
        return center_T_points_pose_4_4_list
    
    # random values generated from a Gaussian distribution
    def take_easy_gaussian_value(self, mean, sigma):
        normal = random.normalvariate(mean, sigma)
        return normal

    # (1000,3) -> (333,333,334)/(334,333,332)
    # (1000,7) -> (143,143,143,142,143,143,143)/(143,142,143,143,143,143,143)
    def divide_and_shuffle(self, total, n):
        base = total // n
        remainder = total % n
        result = [base + 1] * remainder + [base] * (n - remainder)
        random.shuffle(result)
        return result

    # add noise
    def add_noise_pose(self, obj_pos, obj_ori): # obj_pos: x,y,z; obj_ori: x,y,z,w
        # add noise to pos of object
        normal_x = self.add_noise_2_par(obj_pos[0])
        normal_y = self.add_noise_2_par(obj_pos[1])
        normal_z = self.add_noise_2_par(obj_pos[2])
        # add noise to ang of object
        quat_QuatStyle = Quaternion(x=obj_ori[0], y=obj_ori[1], z=obj_ori[2], w=obj_ori[3])# w,x,y,z
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