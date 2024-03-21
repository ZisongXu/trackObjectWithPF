#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:57:49 2021

@author: 12106
"""
from gazebo_msgs.msg import ModelStates
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
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState, CameraInfo, Image, PointCloud2
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3
from PBPF.msg import estimated_obj_pose, object_pose, particle_list, particle_pose
import tf
import tf.transformations as transformations
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import message_filters
import cv2
#pybullet
from pyquaternion import Quaternion
import pybullet as p
import time
import pybullet_data
from pybullet_utils import bullet_client as bc
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import random
import copy
import os
import signal
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
import pandas as pd
import multiprocessing
import yaml
import jax.numpy as jnp
from jax import jit
from collections import namedtuple
from scipy.spatial.transform import Rotation as R

#from sksurgerycore.algorithms.averagequaternions import average_quaternions
from quaternion_averaging import weightedAverageQuaternions
#class in other files
from Franka_robot import Franka_robot
from Ros_Listener import Ros_Listener
from Particle import Particle
from InitialSimulationModel import InitialSimulationModel
from Realworld import Realworld
from Visualisation_World import Visualisation_World
from Create_Scene import Create_Scene
from Object_Pose import Object_Pose
from Robot_Pose import Robot_Pose
from Center_T_Point_for_Ray import Center_T_Point_for_Ray
from launch_camera import LaunchCamera


# parameter_filename = rospy.get_param("~parameter_filename")
with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
    parameter_info = yaml.safe_load(file)

gazebo_flag = parameter_info['gazebo_flag']
# scene
task_flag = parameter_info['task_flag'] # parameter_info['task_flag']
# which algorithm to run
run_alg_flag = parameter_info['run_alg_flag'] # PBPF/CVPF
# update mode (pose/time)
update_style_flag = parameter_info['update_style_flag'] # time/pose
# observation model
pick_particle_rate = parameter_info['pick_particle_rate']
OPTITRACK_FLAG = parameter_info['optitrack_flag']

VERSION = parameter_info['version'] # old/ray/multiray/
RUNNING_MODEL = parameter_info['running_model'] # PBPF_RGB/PBPF_RGBD/PBPF_D

DEBUG_DEPTH_IMG_FLAG = parameter_info['debug_depth_img_flag'] # old/ray/multiray/depth_img
USE_CONVOLUTION_FLAG = parameter_info['use_convolution_flag'] # old/ray/multiray/depth_img
CONVOLUTION_SIZE = parameter_info['convolution_size'] # old/ray/multiray/depth_img
# the flag is used to determine whether the robot touches the particle in the simulation
simRobot_touch_par_flag = 0
OBJECT_NUM = parameter_info['object_num']
ROBOT_NUM = parameter_info['robot_num']
SIM_REAL_WORLD_FLAG = parameter_info['sim_real_world_flag']

LOCATE_CAMERA_FLAG = parameter_info['locate_camera_flag']

PARTICLE_NUM = parameter_info['particle_num']

OBJECT_NAME_LIST = parameter_info['object_name_list']
obstacles_pos = parameter_info['obstacles_pos'] # old/ray/multiray
obstacles_ori = parameter_info['obstacles_ori'] # old/ray/multiray

CAMERA_INFO_TOPIC_COLOR = parameter_info['camera_info_topic_color'] # /camera/color/camera_info
CAMERA_INFO_TOPIC_DEPTH = parameter_info['camera_info_topic_depth'] # /camera/depth/camera_info

NEARVAL = parameter_info['nearVal'] # 57.86
FARVAL = parameter_info['farVal'] # 57.86

DEPTH_IMAGE_CUT_FLAG = parameter_info['depth_image_cut_flag'] 
PERSP_TO_ORTHO_FLAG = parameter_info['persp_to_ortho_flag'] 
ORTHO_TO_PERSP_FLAG = parameter_info['ortho_to_persp_flag'] 
DEPTH_DIFF_VALUE_0_1_FLAG = parameter_info['depth_diff_value_0_1_flag'] 
DEPTH_DIFF_VALUE_0_1_THRESHOLD = parameter_info['depth_diff_value_0_1_threshold'] 
DEPTH_DIFF_VALUE_0_1_ALPHA = parameter_info['depth_diff_value_0_1_alpha'] 
DEPTH_MASK_FLAG = parameter_info['depth_mask_flag'] 
COMBINE_PARTICLE_DEPTH_MASK_FLAG = parameter_info['combine_particle_depth_mask_flag'] 
SHOW_PARTICLE_DEPTH_IMAGE_TO_POINT_CLOUD_FLAG = parameter_info['show_particle_depth_image_to_point_cloud_flag'] 

PRINT_SCORE_FLAG = parameter_info['print_score_flag'] 
SHOW_RAY = parameter_info['show_ray']
VK_RENDER_FLAG = parameter_info['vk_render_flag']
PB_RENDER_FLAG = parameter_info['pb_render_flag']
PANDA_ROBOT_LINK_NUMBER = parameter_info['panda_robot_link_number']
if VK_RENDER_FLAG == True:
    print("I am using Vulkan to generate Depth Image")
if PB_RENDER_FLAG == True: 
    print("I am using Pybullet to generate Depth Image")
CHANGE_SIM_TIME = 1.0/240

# ==============================================================================================================================
# vulkan
from pathlib import Path
import sys
sys.path.insert( 1, str(Path( __file__ ).parent.parent.absolute() / "bin") )
## Import module
import vkdepth


# pdv.release();
# qdv.release();

# print("Launch Vkdepth successfully")
# ==============================================================================================================================
# mark
# - gelatin

#Class of Physics-based Particle Filtering
class PBPFMove():
    def __init__(self, obj_num=0):
        if SHOW_PARTICLE_DEPTH_IMAGE_TO_POINT_CLOUD_FLAG == True:
            self.setup_camera_info()
        # initialize internal parameters
        self.obj_num = obj_num
        self.particle_cloud = copy.deepcopy(initial_parameter.particle_cloud)
        self.particle_no_visual_id_collection = copy.deepcopy(initial_parameter.particle_no_visual_id_collection)
        self.pybullet_env_id_collection = copy.deepcopy(initial_parameter.pybullet_particle_env_collection)
        self.pybullet_sim_fake_robot_id_collection = copy.deepcopy(initial_parameter.fake_robot_id_collection)
        self.pybullet_sim_env_fix_obj_id_collection = copy.deepcopy(initial_parameter.env_fix_obj_id_collection)
        self.pybullet_sim_other_object_id_collection = copy.deepcopy(initial_parameter.other_object_id_collection)
        self.other_obj_num = other_obj_num
        self.joint_num = 7
        self.object_estimate_pose_x = []
        self.object_estimate_pose_y = []
        self.object_real_____pose_x = []
        self.object_real_____pose_y = []
        self.do_obs_update = True
        self.rays_id_list = []
        self.ray_list_empty = True
        self.num = 0

        self.real_depth_image_transferred = 0
        self.real_depth_image_transferred_jax = 0
        self.depth_value_difference_list = [1] * PARTICLE_NUM
        self.rendered_depth_images_list = [1] * PARTICLE_NUM
        self.rendered_depth_image_transferred_list = [1] * PARTICLE_NUM

        self.mask_position_from_segImg_list = [1] * PARTICLE_NUM

        self.cracker_dis_error = [1] * PARTICLE_NUM
        self.cracker_ang_error = [1] * PARTICLE_NUM
        self.cracker_weight_before_ray = [1] * PARTICLE_NUM
        self.cracker_weight__after_ray = [1] * PARTICLE_NUM
        self.soup_dis_error = [1] * PARTICLE_NUM
        self.soup_ang_error = [1] * PARTICLE_NUM
        self.soup_weight_before_ray = [1] * PARTICLE_NUM
        self.soup_weight__after_ray = [1] * PARTICLE_NUM
        
        self.x_min = 0
        self.x_max = 1
        self.y_min = 0
        self.y_max = 1

        self.bridge = CvBridge()

    def get_real_robot_joint(self, pybullet_env_id, real_robot_id):
        real_robot_joint_list = []
        for index in range(self.joint_num):
            real_robot_info = pybullet_env_id.getJointState(real_robot_id,index)[0]
            real_robot_joint_list.append(real_robot_info)
        return real_robot_joint_list
        
    def set_real_robot_JointPosition(self,pybullet_env, robot, position):
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
    
    # executed_control
    def update_particle_filter_PB(self, real_robot_joint_pos, pw_T_obj_obse_objects_pose_list):
        global flag_record_obse
        global flag_record_PBPF
        global flag_record
        
        self.rays_id_list = []

        pybullet_sim_envs = self.pybullet_env_id_collection
        particle_robot_id = self.pybullet_sim_fake_robot_id_collection
        
        self.rendered_depth_images_list = [1] * PARTICLE_NUM
        self.rendered_depth_image_transferred_list = [1] * PARTICLE_NUM

        self.DOPE_rep_flag = 0
        self.times = []
        t1 = time.time()

        print("-------------------------------------")
        print("global_objects_visual_by_DOPE_list: -")
        print(global_objects_visual_by_DOPE_list)
        print("global_objects_outlier_by_DOPE_list:-")
        print(global_objects_outlier_by_DOPE_list)
        print("-------------------------------------")

        # motion model
        self.motion_model(pybullet_sim_envs, particle_robot_id, real_robot_joint_pos)
        t2 = time.time()
        self.times.append(t2-t1)
        print("--------------------------------------------------------")
        print("Motion model cost time:", t2 - t1)
        print("--------------------------------------------------------")
        # OBSERVATION MODEL:
        self.observation_model(pw_T_obj_obse_objects_pose_list, pybullet_sim_envs)
        
        
        # mark
        # resample
        # self.resample_particles_update(pw_T_obj_obse_objects_pose_list)

        self.set_particle_in_each_sim_env()
        
        # Compute mean of particles
        object_estimate_pose, dis_std_list, ang_std_list = self.compute_estimate_pos_of_object(self.particle_cloud)
        # publish pose of particles
        publish_par_pose_info(self.particle_cloud)
        publish_esti_pose_info(object_estimate_pose)
        
        if SHOW_RAY == True:
            for index in range(len(self.pybullet_env_id_collection)):
                self.pybullet_env_id_collection[index].removeAllUserDebugItems()
        
        return object_estimate_pose, dis_std_list, ang_std_list, self.particle_cloud

    # motion model
    def motion_model(self, pybullet_sim_envs, particle_robot_id, real_robot_joint_pos):  
        print("Welcome to the motion model!")
        self.motion_update_PB_parallelised(pybullet_sim_envs, particle_robot_id, real_robot_joint_pos)

    def motion_update_PB_parallelised(self, pybullet_sim_envs, particle_robot_id, real_robot_joint_pos):
        global simRobot_touch_par_flag
        threads = []
        for index, pybullet_env in enumerate(pybullet_sim_envs):
            # execute code in parallel
            if simRobot_touch_par_flag == 1:
                thread = threading.Thread(target=self.motion_update_PB, args=(index, pybullet_env, particle_robot_id, real_robot_joint_pos))
            else:
                thread = threading.Thread(target=self.sim_robot_move_direct, args=(index, pybullet_env, particle_robot_id, real_robot_joint_pos))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        if simRobot_touch_par_flag == 0:
            return
    
    def motion_update_PB(self, index, pybullet_env, particle_robot_id, real_robot_joint_pos):
        collision_detection_obj_id = []
        other_object_id_list_list = self.pybullet_sim_other_object_id_collection # now is empty
        
        # collision check: add other objects (obstacles)
        # now is empty
        for other_obj_index in range(len(other_object_id_list_list)):
            other_object_id = other_object_id_list_list[other_obj_index][index]
            collision_detection_obj_id.append(other_object_id)
        
        # ensure that each update of particles in the simulation inherits the velocity of the previous update 
        for obj_index in range(self.obj_num):
            pw_T_par_sim_id = self.particle_cloud[index][obj_index].no_visual_par_id
            pybullet_env.resetBaseVelocity(pw_T_par_sim_id,
                                           self.particle_cloud[index][obj_index].linearVelocity,
                                           self.particle_cloud[index][obj_index].angularVelocity)
            # change particle parameters
            self.change_obj_parameters(pybullet_env, pw_T_par_sim_id)
        # execute the control
        if update_style_flag == "pose":
            self.pose_sim_robot_move(index, pybullet_env, particle_robot_id, real_robot_joint_pos)
        elif update_style_flag == "time":
            # change simulation time
            pf_update_interval_in_sim = BOSS_PF_UPDATE_INTERVAL_IN_REAL / CHANGE_SIM_TIME
            # make sure all particles are updated
            for time_index in range(int(pf_update_interval_in_sim)):
                self.set_real_robot_JointPosition(pybullet_env, particle_robot_id[index], real_robot_joint_pos)
                pybullet_env.stepSimulation()
        
        
        
        ### ori: x,y,z,w
        # collision check: add robot
        collision_detection_obj_id.append(particle_robot_id[index])
        collision_detection_obj_id.append(self.pybullet_sim_env_fix_obj_id_collection[index])
        # now is empty
        for oto_index in range(self.other_obj_num):
            collision_detection_obj_id.append(self.pybullet_sim_other_object_id_collection[oto_index])

        for obj_index in range(self.obj_num):
            
            pw_T_par_sim_id = self.particle_cloud[index][obj_index].no_visual_par_id
            # get linearVelocity and angularVelocity of each particle
            linearVelocity, angularVelocity = pybullet_env.getBaseVelocity(pw_T_par_sim_id)
            sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(pybullet_env, pw_T_par_sim_id)
            # add noise on pose of each particle
            normal_x, normal_y, normal_z, P_quat = self.add_noise_pose(sim_par_cur_pos, sim_par_cur_ori)
            pybullet_env.resetBasePositionAndOrientation(pw_T_par_sim_id,
                                                         [normal_x, normal_y, normal_z],
                                                         P_quat)
            collision_detection_obj_id.append(pw_T_par_sim_id)

            # check collision
            par_pose_3_1 = [normal_x, normal_y, normal_z, P_quat]
            normal_x, normal_y, normal_z, P_quat = self.collision_check(pybullet_env, 
                                                                        collision_detection_obj_id, 
                                                                        sim_par_cur_pos, sim_par_cur_ori, 
                                                                        pw_T_par_sim_id, index, obj_index, 
                                                                        par_pose_3_1)
            
            self.update_partcile_cloud_pose_PB(index, obj_index, normal_x, normal_y, normal_z, P_quat, linearVelocity, angularVelocity)
        pybullet_env.stepSimulation()
        # pipe.send()

    # observation model
    def observation_model(self, pw_T_obj_obse_objects_pose_list, pybullet_sim_envs):
        print("Welcome to the observation model!")
        # observation model (DEPTH)
        if USING_D_FLAG == True:
            # get real depth image
            self.get_real_depth_image()

            # get rendered depth/seg image
            t_before_render = time.time()
            if PB_RENDER_FLAG == True and VK_RENDER_FLAG == False:
                print("I am using Pybullet to generate Depth Image")
                self.pd_get_rendered_depth_image_parallelised(self.particle_cloud)
            elif VK_RENDER_FLAG == True and PB_RENDER_FLAG == False:
                print("I am using Vulkan to generate Depth Image")
                self.vk_get_rendered_depth_image_parallelised(self.particle_cloud)
            else:
                while True:
                    print("Error!!! VK_RENDER_FLAG and PB_RENDER_FLAG can not be TRUE or FALSE at the same time")
            t_after_render = time.time()
            print("--------------------------------------------------------")
            print("Render cost time:", t_after_render - t_before_render)
            print("--------------------------------------------------------")
            # mark
            if DEPTH_MASK_FLAG == True and COMBINE_PARTICLE_DEPTH_MASK_FLAG == True:
                # flat_mask_position_list_jax = jnp.vstack(self.mask_position_from_segImg_list)
                # # get x_min, x_mad, y_min, y_max
                # self.x_min, self.x_max, self.y_min, self.y_max = self.get_bounding_box(flat_mask_position_list_jax)
                   
                self.x_min = 190
                self.x_max = 399
                self.y_min = 348
                self.y_max = 549

            # compare depth image
            self.compare_depth_image_parallelised(self.particle_cloud)

        # observation model (RGB)
        if USING_RGB_FLAG == True:
            self.observation_update_PB_parallelised(self.particle_cloud, pw_T_obj_obse_objects_pose_list)

    def get_real_depth_image(self):
        depth_image_real = ROS_LISTENER.depth_image # persp

        # mark
        # self.is_ros_depth_image(depth_image_real)

        self.real_depth_image_transferred = self.depthImageRealTransfer(depth_image_real) # persp
        if DEPTH_MASK_FLAG == True:
            self.real_depth_image_transferred_jax = jnp.array(self.real_depth_image_transferred) # persp
            if PERSP_TO_ORTHO_FLAG == True:
                # mark
                print("Begin to change from persp to ortho")
                self.real_depth_image_transferred_jax = _persp_to_ortho(self.real_depth_image_transferred_jax, FY_DEPTH, CX_DEPTH, CY_DEPTH)

    # get rendered depth/seg image model PyBullet
    def pd_get_rendered_depth_image_parallelised(self, particle_cloud):
        threads_obs = []
        for index, particle in enumerate(particle_cloud):
            thread_obs = threading.Thread(target=self.get_rendered_depth_image, args=(index, particle))
            thread_obs.start()
            threads_obs.append(thread_obs)
        for thread_obs in threads_obs:
            thread_obs.join()

    # get rendered depth/seg image model PyBullet
    def get_rendered_depth_image(self, index, particle):
        pybullet_env = self.pybullet_env_id_collection[index]
        pybullet_env.stepSimulation()

        # generate rendered depth/seg image in Pybullet
        width, height, rgbImg, depth_image_render, segImg, nearVal, farVal = _launch_camera.setCameraPicAndGetPic(pybullet_env, _tf_listener, _pw_T_rob_sim_4_4)

        # get target objects ID
        if DEPTH_MASK_FLAG == True:
            mask_position_from_segImg = self.get_target_objects_ID_from_segImg(particle, segImg, index) # jax
            self.mask_position_from_segImg_list[index] = mask_position_from_segImg # list
        self.rendered_depth_images_list[index] = depth_image_render # array/list

    # get rendered depth/seg image model PyBullet
    def vk_get_rendered_depth_image_parallelised(self, particle_cloud):
        # vk mark 
        # get robot link state
        pybullet_sim_envs = self.pybullet_env_id_collection
        pybullet_sim_envs_0 = pybullet_sim_envs[0]
        particle_robot_id_collection = self.pybullet_sim_fake_robot_id_collection
        particle_robot_id_0 = particle_robot_id_collection[0]

        ## Update particle pose->update depth image
        _vk_update_depth_image(_vk_state_list, particle_cloud, pybullet_sim_envs_0, particle_robot_id_0)
        ## Render and Download
        _vk_context.enqueue_render_and_download()
        ## Waiting for rendering and download
        _vk_context.wait()
        ## Get Depth image
        vk_rendered_depth_image_array_list_ = _vk_depth_image_getting()

        # fig, axs = plt.subplots(1, PARTICLE_NUM)
        # for par_index in range(PARTICLE_NUM):
        #     axs[par_index].imshow(vk_rendered_depth_image_array_list_[par_index])
        # plt.show()


        # self.mask_position_from_segImg_list[index] = mask_position_from_segImg # list
        self.rendered_depth_images_list = copy.deepcopy(vk_rendered_depth_image_array_list_) # array/list

    # get target objects ID
    def get_target_objects_ID_from_segImg(self, particle, segImg, index):
        obj_id_array = jnp.array([0] * OBJECT_NUM)
        for obj_index in range(self.obj_num):
            obj_id = particle[obj_index].no_visual_par_id
            obj_id_array = obj_id_array.at[obj_index].set(obj_id)
        segImg_ID_array = segImg & ((1 << 24) - 1)
        mask_position_from_segImg = self.findPositions(segImg_ID_array, obj_id_array) # jax
        return mask_position_from_segImg

    # get bounding box
    def get_bounding_box(self, flat_arr):
        x_min = flat_arr[:, 0].min()
        x_max = flat_arr[:, 0].max()
        y_min = flat_arr[:, 1].min()
        y_max = flat_arr[:, 1].max()
        return x_min, x_max, y_min, y_max

    # compare depth image
    def compare_depth_image_parallelised(self, particle_cloud):
        threads_obs = []
        for index, particle in enumerate(particle_cloud):
            thread_obs = threading.Thread(target=self.compare_depth_image, args=(index, particle))
            thread_obs.start()
            threads_obs.append(thread_obs)
        for thread_obs in threads_obs:
            thread_obs.join()
    
    def setup_camera_info(self, camera_info_topic_name="/camera/depth/camera_info"):
        self.camera_info = None
        self.camera_info_sub = rospy.Subscriber(camera_info_topic_name, CameraInfo, self.camera_info_callback)
        self.camera_info_pub = rospy.Publisher('/camera/camera_info', CameraInfo)

    def camera_info_callback(self, data):
        self.camera_info = data

    # compare depth image
    def compare_depth_image(self, index, particle):
        if USING_D_FLAG == True:
            
            depth_image_render = copy.deepcopy(self.rendered_depth_images_list[index]) # array/list
            if PB_RENDER_FLAG == True:
                rendered_depth_image_transferred = self.renderedDepthImageValueBufferTransfer(depth_image_render) # array
            if VK_RENDER_FLAG == True:
                rendered_depth_image_transferred = copy.deepcopy(depth_image_render)
            # show depth image
            if DEBUG_DEPTH_IMG_FLAG == True:
                if COMBINE_PARTICLE_DEPTH_MASK_FLAG == True:
                    real_depth_img_name = str(_particle_update_time) + "_real_depth_img_"+str(index)+".png"
                    # cv2.imwrite(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+real_depth_img_name, (cv_image).astype(np.uint16))
                    imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+real_depth_img_name, self.real_depth_image_transferred[self.x_min:self.x_max+1, self.y_min:self.y_max+1], cmap='gray')

                    rendered_depth_img_name = str(_particle_update_time)+"_rendered_depth_img_"+str(index)+".png"
                    imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+rendered_depth_img_name, rendered_depth_image_transferred[self.x_min:self.x_max+1, self.y_min:self.y_max+1], cmap='gray')
                    # imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+rendered_depth_img_name, rendered_depth_image_transferred, cmap='gray')

                    # rendered_seg_img_name = str(_particle_update_time)+"_rendered_seg_img_"+str(index)+".png"
                    # imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+rendered_seg_img_name, segImg)
                else:
                    real_depth_img_name = "0_" + str(_particle_update_time) + "_real_depth_img_"+str(index)+".png"
                    # cv2.imwrite(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+real_depth_img_name, (cv_image).astype(np.uint16))
                    imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+real_depth_img_name, self.real_depth_image_transferred, cmap='gray')
                    if PB_RENDER_FLAG == True:
                        rendered_depth_img_name = "pw_0_" + str(_particle_update_time)+"_rendered_depth_img_"+str(index)+".png"
                        imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+rendered_depth_img_name, rendered_depth_image_transferred, cmap='gray')
                    if VK_RENDER_FLAG == True:
                        rendered_depth_img_name = "vk_0_" + str(_particle_update_time)+"_rendered_depth_img_"+str(index)+".png"
                        imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+rendered_depth_img_name, rendered_depth_image_transferred, cmap='gray')
                    
                    # rendered_seg_img_name = str(_particle_update_time)+"_rendered_seg_img_"+str(index)+".png"
                    # imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+rendered_seg_img_name, segImg)
            
            # mark
            if SHOW_PARTICLE_DEPTH_IMAGE_TO_POINT_CLOUD_FLAG == True:
                if self.camera_info:
                    par_depth_img = copy.deepcopy(rendered_depth_image_transferred)
                    cv_image = par_depth_img * 1
                    # imsave('test.png', cv_image)

                    ros_image = self.bridge.cv2_to_imgmsg(cv_image, "passthrough")

                    data = ros_image
                    # print(f'Sim depth: {data.header} {data.height}, {data.width}, {data.encoding}, {data.is_bigendian}, {data.step}, {len(data.data)}, {data.data[0]}')

                    ros_image.header.stamp = self.camera_info.header.stamp
                    ros_image.header.frame_id = "camera_depth_optical_frame"

                    # check image is a cv_img or ros_img
                    # self.is_cv_depth_image(par_depth_img)
                    # self.is_ros_depth_image(ros_image)

                    self.camera_info_pub.publish(self.camera_info)
                    pub_depth_image.publish(ros_image)
                    


            if USE_CONVOLUTION_FLAG == True:
                depth_value_difference = self.compute_difference_bt_2_depthImg_convolution(real_depth_image_transferred, rendered_depth_image_transferred)
            else:
                # use mask  
                if DEPTH_MASK_FLAG == True:
                    rendered_depth_image_transferred_jax = jnp.array(rendered_depth_image_transferred) # jax
                    
                    if ORTHO_TO_PERSP_FLAG == True:
                        # mark
                        rendered_depth_image_transferred_jax = _ortho_to_persp(rendered_depth_image_transferred_jax, FY_DEPTH, CX_DEPTH, CY_DEPTH)

                    if COMBINE_PARTICLE_DEPTH_MASK_FLAG == True:
                        real_depth_image_mask_values = self.real_depth_image_transferred_jax[self.x_min:self.x_max+1, self.y_min:self.y_max+1] # jax 
                        real_depth_image_mask_values = real_depth_image_mask_values.ravel() # jax
                        rendered_depth_image_mask_values = rendered_depth_image_transferred_jax[self.x_min:self.x_max+1, self.y_min:self.y_max+1] # jax
                        rendered_depth_image_mask_values = rendered_depth_image_mask_values.ravel() # jax
                        number_of_pixels = len(rendered_depth_image_mask_values)
                    else:
                        mask_position_from_segImg = self.mask_position_from_segImg_list[index]
                        number_of_pixels = len(mask_position_from_segImg)

                        real_depth_image_mask_values = _extractValues(self.real_depth_image_transferred_jax, mask_position_from_segImg)
                        rendered_depth_image_mask_values = _extractValues(rendered_depth_image_transferred_jax, mask_position_from_segImg)

                        # real_depth_image_mask_values = self.replace_values_real(self.real_depth_image_transferred_jax, mask_position_from_segImg)
                        # rendered_depth_image_mask_values = self.replace_values_render(rendered_depth_image_transferred_jax, mask_position_from_segImg)

                        # test
                        SHOW_ONLY_MASK_IMG_FLAG = True
                        if SHOW_ONLY_MASK_IMG_FLAG == True:
                            a = 1
                            # real_depth_img_name = str(_particle_update_time) + "_real_depth_img_"+str(index)+".png"
                            # # cv2.imwrite(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+real_depth_img_name, (cv_image).astype(np.uint16))
                            # imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+real_depth_img_name, real_depth_image_mask_values)

                            # rendered_depth_img_name = str(_particle_update_time)+"_rendered_depth_img_"+str(index)+".png"
                            # imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+rendered_depth_img_name, rendered_depth_image_mask_values)
                        
                            # real_depth_img_name = "0_" + str(_particle_update_time) + "_real_depth_img_"+str(index)+".png"
                            # # cv2.imwrite(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+real_depth_img_name, (cv_image).astype(np.uint16))
                            # imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+real_depth_img_name, self.real_depth_image_transferred)

                            # rendered_depth_img_name = "0_" + str(_particle_update_time)+"_rendered_depth_img_"+str(index)+".png"
                            # imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+rendered_depth_img_name, rendered_depth_image_transferred) # , cmap='gray'



                    # mark
                    if DEPTH_DIFF_VALUE_0_1_FLAG == True:
                        depth_value_diff_sub_abs_jax = jnp.abs(real_depth_image_mask_values - rendered_depth_image_mask_values)
                        depth_value_diff_sub_abs_0_1_jax, num_zeros = _threshold_array_optimized(depth_value_diff_sub_abs_jax)

                        # rendered_depth_img_name = "Compared_" + str(_particle_update_time)+"_rendered_depth_img_"+str(index)+".png"
                        # imsave(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+rendered_depth_img_name, depth_value_diff_sub_abs_0_1_jax, cmap='gray') # , cmap='gray'


                        e_VSD_o = jnp.sum(depth_value_diff_sub_abs_0_1_jax) / number_of_pixels
                        score = DEPTH_DIFF_VALUE_0_1_ALPHA * (1-e_VSD_o) + (1-DEPTH_DIFF_VALUE_0_1_ALPHA) * num_zeros/number_of_pixels
                        depth_value_difference_jax = score # score_that_particle_get: high->high weight; low->low weight
                    else:
                        depth_value_difference_jax = jnp.linalg.norm(real_depth_image_mask_values - rendered_depth_image_mask_values)
                        depth_value_difference_jax = depth_value_difference_jax / (math.sqrt(number_of_pixels))

                    depth_value_difference = float(depth_value_difference_jax.item())

                else:
                    depth_value_difference = self.compareDifferenceBtTwoDepthImgs(self.real_depth_image_transferred, rendered_depth_image_transferred)
                
            # mark
            # if PRINT_SCORE_FLAG == True:
            #     a = 1
            if DEPTH_DIFF_VALUE_0_1_FLAG == True:
                print("_particle_update_time: ",_particle_update_time,"; Index:", index, "; score_that_particle_get: ",depth_value_difference)
                print("==================================")
            else:
                print("_particle_update_time: ",_particle_update_time,"; Index:", index, "; depth_value_difference: ",depth_value_difference)
                print("==================================")
            
            self.depth_value_difference_list[index] = depth_value_difference

    def cutImage(self, image, up=0, down=0, left=0, right=0):
        image_cutted = image[up:HEIGHT_DEPTH-down, left:WIDTH_DEPTH-right]
        return image_cutted

    def replace_values(self, a, b):
        result = jnp.zeros_like(a)
        x_coords, y_coords = b[:, 0], b[:, 1]
        result = result.at[x_coords, y_coords].set(a[x_coords, y_coords])
        return result

    def replace_values_render(self, a, b):
        result = jnp.zeros_like(a)
        x_coords, y_coords = b[:, 0], b[:, 1]
        result = result.at[x_coords, y_coords].set(a[x_coords, y_coords])
        return result
    
    def replace_values_real(self, a, b):
        result = jnp.full_like(a, 1000)
        x_coords, y_coords = b[:, 0], b[:, 1]
        result = result.at[x_coords, y_coords].set(a[x_coords, y_coords])
        return result

    # update particle cloud particle angle
    def update_partcile_cloud_pose_PB(self, index, obj_index, x, y, z, ori, linearVelocity, angularVelocity):
        self.particle_cloud[index][obj_index].pos = [x, y, z]
        self.particle_cloud[index][obj_index].ori = copy.deepcopy(ori)
        self.particle_cloud[index][obj_index].linearVelocity = linearVelocity
        self.particle_cloud[index][obj_index].angularVelocity = angularVelocity
        # self.particle_cloud[index][obj_index].rayTraceList = [1,2,3]
        
        # mark
        # set each particle pose
        particle_pos = copy.deepcopy([x, y, z])
        particle_ori = copy.deepcopy(ori)
        self.set_particle_in_each_sim_env_single(index, obj_index, particle_pos, particle_ori)

    def compute_difference_bt_2_depthImg_convolution(self, test_arr1, test_arr2):
        compare_root_list = []
        loop_size = 2 * CONVOLUTION_SIZE + 1
        for h_size in range(loop_size):
            for w_size in range(loop_size):
                if h_size <= CONVOLUTION_SIZE:
                    if w_size <= CONVOLUTION_SIZE:
                        a = test_arr1[     0:HEIGHT_DEPTH-h_size,      0:WIDTH_DEPTH-w_size]
                        b = test_arr2[h_size:HEIGHT_DEPTH       , w_size:WIDTH_DEPTH]
                    else:
                        a = test_arr1[     0:HEIGHT_DEPTH-h_size, w_size-CONVOLUTION_SIZE:WIDTH_DEPTH-0]
                        b = test_arr2[h_size:HEIGHT_DEPTH       ,           0:WIDTH_DEPTH-(w_size-CONVOLUTION_SIZE)]
                else:
                    if w_size <= CONVOLUTION_SIZE:
                        a = test_arr1[h_size-CONVOLUTION_SIZE:HEIGHT_DEPTH-0,                  0:WIDTH_DEPTH-w_size]
                        b = test_arr2[          0:HEIGHT_DEPTH-(h_size-CONVOLUTION_SIZE), w_size:WIDTH_DEPTH]
                    else:
                        a = test_arr1[h_size-CONVOLUTION_SIZE:HEIGHT_DEPTH-0,           w_size-CONVOLUTION_SIZE:WIDTH_DEPTH-0]
                        b = test_arr2[          0:HEIGHT_DEPTH-(h_size-CONVOLUTION_SIZE),         0:WIDTH_DEPTH-(w_size-CONVOLUTION_SIZE)]
                compare_array = a - b
                dim_number = compare_array.ndim
                compare_array_square = compare_array ** 2
                compare_array_square_copy = copy.deepcopy(compare_array_square)
                for dim_n in range(dim_number):
                    compare_array_square_copy = sum(compare_array_square_copy)
                compare_array_square_sum_root = math.sqrt(compare_array_square_copy)
                compare_root_list.append(compare_array_square_sum_root)
        compare_root_list_array = np.array(compare_root_list)
        compare_root_list_array_square = compare_root_list_array ** 2
        compare_root_list_array_square_sum = sum(compare_root_list_array_square)
        compare_root_list_array_square_sum_root = math.sqrt(compare_root_list_array_square_sum)
        return compare_root_list_array_square_sum_root

    def compareDifferenceBtTwoDepthImgs(self, real_depth_image_transferred, rendered_depth_image_transferred):
        depth_value_diff_subtraction = self.subtractionTwoDepthImgs(real_depth_image_transferred, rendered_depth_image_transferred)
        depth_value_diff_subtraction_square_root = self.depthValueDifferenceSquareRoot(depth_value_diff_subtraction)
        return depth_value_diff_subtraction_square_root

    def subtractionTwoDepthImgs(self, depthImg1, depthImg2):
        depth_value_diff_subtraction = depthImg1 - depthImg2
        return depth_value_diff_subtraction

    def depthValueDifferenceSquareRoot(self, depth_value_diff_subtraction):
        dim_number = depth_value_diff_subtraction.ndim
        depth_value_diff_subtraction_square = depth_value_diff_subtraction ** 2
        interim_container = copy.deepcopy(depth_value_diff_subtraction_square)
        for dim_n in range(dim_number):
            interim_container = sum(interim_container)
        interim_container = interim_container / (HEIGHT_DEPTH*WIDTH_DEPTH)
        depth_value_diff_subtraction_square_root = math.sqrt(interim_container)
        return depth_value_diff_subtraction_square_root        

    def depthImageRealTransfer(self, depth_image_real):
        cv_image = self.bridge.imgmsg_to_cv2(depth_image_real,"16UC1")
        
        if DEPTH_IMAGE_CUT_FLAG == True:
            cv_image = self.cutImage(cv_image, up=226, down=164, left=400, right=299) # up, down, left, right

        cv_image = cv_image / 1000
        
        return cv_image

    def find_square(self, x):
        if x <= 0.4 or x >= 1.2:
            return 0.0
        else:
            return x

    def renderedDepthImageValueBufferTransfer(self, depth_image_render):
        pybullet_depth_image_value_transferred = FARVAL * NEARVAL / (FARVAL - (FARVAL - NEARVAL) * depth_image_render)
        return pybullet_depth_image_value_transferred

    # judge if any particles are contact
    def isAnyParticleInContact(self):
        for index, particle in enumerate(self.particle_cloud):
            for obj_index in range(OBJECT_NUM):
                # get pose from particle
                pw_T_par_sim_pw_env = self.pybullet_env_id_collection[index]
                # pw_T_par_sim_id = self.particle_no_visual_id_collection[index][obj_index]
                pw_T_par_sim_id = particle[obj_index].no_visual_par_id
                # sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(self.pybullet_env_id_collection[index], pw_T_par_sim_id)
                sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(pw_T_par_sim_pw_env, pw_T_par_sim_id)

                # check contact 
                pmin, pmax = pw_T_par_sim_pw_env.getAABB(pw_T_par_sim_id)
                collide_ids = pw_T_par_sim_pw_env.getOverlappingObjects(pmin, pmax)
                length = len(collide_ids)
                for t_i in range(length):
                    # print("body id: ",collide_ids[t_i][1])
                    if collide_ids[t_i][1] == 8 or collide_ids[t_i][1] == 9 or collide_ids[t_i][1] == 10 or collide_ids[t_i][1] == 11:
                        return True
        return False
    
    # observation model:
    def observation_update_PB_parallelised(self, particle_cloud, pw_T_obj_obse_objects_pose_list):
        threads_obs = []
        for index, particle in enumerate(particle_cloud):
            thread_obs = threading.Thread(target=self.observation_update_PB, args=(index, particle, pw_T_obj_obse_objects_pose_list))
            thread_obs.start()
            threads_obs.append(thread_obs)
        for thread_obs in threads_obs:
            thread_obs.join()
    
    # observation model
    def observation_update_PB(self, index, particle, pw_T_obj_obse_objects_pose_list):

        pybullet_env = self.pybullet_env_id_collection[index]
        weight =  1.0 / PARTICLE_NUM
        for obj_index in range(self.obj_num):
            particle[obj_index].w = weight

        if (sum(global_objects_visual_by_DOPE_list)<OBJECT_NUM) and (sum(global_objects_outlier_by_DOPE_list)<OBJECT_NUM):       
            for obj_index in range(self.obj_num):
                local_obj_visual_by_DOPE_val = global_objects_visual_by_DOPE_list[obj_index]
                local_obj_outlier_by_DOPE_val = global_objects_outlier_by_DOPE_list[obj_index]
                particle_x = particle[obj_index].pos[0]
                particle_y = particle[obj_index].pos[1]
                particle_z = particle[obj_index].pos[2]
                par_ori = quaternion_correction(particle[obj_index].ori)
                # 0 means DOPE detects the object[obj_index]
                # 1 means DOPE does not detect the object[obj_index] and skip this loop
                if local_obj_visual_by_DOPE_val==0 and local_obj_outlier_by_DOPE_val==0: 
                    obse_obj_pos = pw_T_obj_obse_objects_pose_list[obj_index].pos
                    obse_obj_ori = pw_T_obj_obse_objects_pose_list[obj_index].ori # pybullet x,y,z,w
                    # make sure theta between -pi and pi
                    obse_obj_ori_corr = quaternion_correction(obse_obj_ori)
            
                    mean = 0
                    # position weight
                    dis_x = abs(particle_x - obse_obj_pos[0])
                    dis_y = abs(particle_y - obse_obj_pos[1])
                    dis_z = abs(particle_z - obse_obj_pos[2])
                    dis_xyz = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
                    weight_xyz = self.normal_distribution(dis_xyz, mean, boss_sigma_obs_pos)
                    # rotation weight
                    obse_obj_quat = Quaternion(x=obse_obj_ori_corr[0], 
                                            y=obse_obj_ori_corr[1], 
                                            z=obse_obj_ori_corr[2], 
                                            w=obse_obj_ori_corr[3]) # Quaternion(): w,x,y,z
                    par_quat = Quaternion(x=par_ori[0], y=par_ori[1], z=par_ori[2], w=par_ori[3])
                    err_bt_par_obse = par_quat * obse_obj_quat.inverse
                    err_bt_par_obse_corr = quaternion_correction([err_bt_par_obse.x, err_bt_par_obse.y, err_bt_par_obse.z, err_bt_par_obse.w])
                    err_bt_par_obse_corr_quat = Quaternion(x=err_bt_par_obse_corr[0], 
                                                        y=err_bt_par_obse_corr[1], 
                                                        z=err_bt_par_obse_corr[2], 
                                                        w=err_bt_par_obse_corr[3])
                    cos_theta_over_2 = err_bt_par_obse_corr_quat.w
                    sin_theta_over_2 = math.sqrt(err_bt_par_obse_corr_quat.x ** 2 + err_bt_par_obse_corr_quat.y ** 2 + err_bt_par_obse_corr_quat.z ** 2)
                    theta_over_2 = math.atan2(sin_theta_over_2, cos_theta_over_2)
                    theta = theta_over_2 * 2.0
                    weight_ang = self.normal_distribution(theta, mean, boss_sigma_obs_ang)
                    
                    weight = weight_xyz * weight_ang
                    
                    if PRINT_SCORE_FLAG == True and OBJECT_NUM == 2:
                        if obj_index == 0:
                            self.cracker_dis_error[index] = dis_xyz
                            self.cracker_ang_error[index] = theta
                            self.cracker_weight_before_ray[index] = weight
                        elif obj_index == 1:
                            self.soup_dis_error[index] = dis_xyz
                            self.soup_ang_error[index] = theta
                            self.soup_weight_before_ray[index] = weight
                            
                if VERSION == "multiray":
                    par_pos_ = copy.deepcopy([particle_x, particle_y, particle_z])
                    par_ori_ = copy.deepcopy(par_ori)
                    weight = self.multi_ray_tracing(par_pos_, par_ori_, pybullet_env, obj_index, weight, local_obj_visual_by_DOPE_val, local_obj_outlier_by_DOPE_val, particle)
                elif VERSION == "ray":
                    par_pos = copy.deepcopy([particle_x, particle_y, particle_z])
                    weight = self.single_ray_tracing(par_pos, pybullet_env, weight, local_obj_visual_by_DOPE_val, local_obj_outlier_by_DOPE_val, particle)
                particle[obj_index].w = weight
                # print("; Error:", self.cracker_par_error)
                # print("; Error:", self.soup_par_error)  
        else:
            if self.DOPE_rep_flag == 0:
                print("DOPE x")
                self.DOPE_rep_flag = self.DOPE_rep_flag + 1
            for obj_index in range(self.obj_num):
                local_obj_visual_by_DOPE_val = global_objects_visual_by_DOPE_list[obj_index]
                local_obj_outlier_by_DOPE_val = global_objects_outlier_by_DOPE_list[obj_index]
                particle_x = particle[obj_index].pos[0]
                particle_y = particle[obj_index].pos[1]
                particle_z = particle[obj_index].pos[2]
                par_ori = quaternion_correction(particle[obj_index].ori)
                if VERSION == "multiray":
                    # need to change
                    par_pos_ = copy.deepcopy([particle_x, particle_y, particle_z])
                    par_ori_ = copy.deepcopy(par_ori)
                    weight = self.multi_ray_tracing(par_pos_, par_ori_, pybullet_env, obj_index, weight, local_obj_visual_by_DOPE_val, local_obj_outlier_by_DOPE_val, particle)
                elif VERSION == "ray":
                    par_pos = copy.deepcopy([particle_x, particle_y, particle_z])
                    weight = self.single_ray_tracing(par_pos, pybullet_env, weight, local_obj_visual_by_DOPE_val, local_obj_outlier_by_DOPE_val, particle)
                particle[obj_index].w = weight
            
    
    def findPositions(self, matrix, targets):
        match_positions = matrix == targets[:, None, None]
        all_positions = jnp.argwhere(match_positions)
        positions = all_positions[:, 1:]
        return positions

    # synchronizing the motion of the robot in the simulation
    def sim_robot_move_direct(self, index, pybullet_env, robot_id, position):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                pybullet_env.resetJointState(robot_id[index],
                                             joint_index+2,
                                             targetValue=position[joint_index])
            else:
                pybullet_env.resetJointState(robot_id[index],
                                             joint_index,
                                             targetValue=position[joint_index])
    
    def pose_sim_robot_move(self, index, pybullet_env, particle_robot_id, real_robot_joint_pos):
        flag_set_sim = 1
        # ensure the robot arm in the simulation moves to the final state on each update
        while not rospy.is_shutdown():
            if flag_set_sim == 0:
                break
            self.set_real_robot_JointPosition(pybullet_env, particle_robot_id[index], real_robot_joint_pos)
            pybullet_env.stepSimulation()
            real_rob_joint_list_cur = self.get_real_robot_joint(pybullet_env, particle_robot_id[index])
            flag_set_sim = self.compare_rob_joint(real_rob_joint_list_cur, real_robot_joint_pos)
            
    def collision_check(self, pybullet_env, collision_detection_obj_id, sim_par_cur_pos, sim_par_cur_ori, pw_T_par_sim_id, index, obj_index, par_pose_3_1):
        normal_x = par_pose_3_1[0]
        normal_y = par_pose_3_1[1]
        normal_z = par_pose_3_1[2]
        P_quat = par_pose_3_1[3]
        if MOTION_NOISE == True:
            nTries = 0
            while nTries < 20:
                nTries=nTries+1
                # print("checking")
                flag = 0
                length_collision_detection_obj_id = len(collision_detection_obj_id)
                for check_num in range(length_collision_detection_obj_id-1):
                    pybullet_env.stepSimulation()
                    # will return all collision points
                    contacts = pybullet_env.getContactPoints(bodyA=collision_detection_obj_id[check_num], # robot, other object...
                                                                bodyB=collision_detection_obj_id[-1]) # main(target) object
                    # pmin,pmax = pybullet_simulation_env.getAABB(particle_no_visual_id)
                    # collide_ids = pybullet_simulation_env.getOverlappingObjects(pmin,pmax)
                    # length = len(collide_ids)
                    for contact in contacts:
                        contactNormalOnBtoA = contact[7]
                        contact_dis = contact[8]
                        if contact_dis < -0.001:
                            #print("detected contact during initialization. BodyA: %d, BodyB: %d, LinkOfA: %d, LinkOfB: %d", contact[1], contact[2], contact[3], contact[4])
                            
                            
                            # par_x_ = sim_par_cur_pos[0] + contactNormalOnBtoA[0]*contact_dis/2
                            # par_y_ = sim_par_cur_pos[1] + contactNormalOnBtoA[1]*contact_dis/2
                            # par_z_ = sim_par_cur_pos[2] + contactNormalOnBtoA[2]*contact_dis/2
                            # particle_pos = [par_x_, par_y_, par_z_]
                            # normal_x = par_x_
                            # normal_y = par_y_
                            # normal_z = par_z_
                            normal_x, normal_y, normal_z, P_quat = self.add_noise_pose(sim_par_cur_pos, sim_par_cur_ori)
                            pybullet_env.resetBasePositionAndOrientation(pw_T_par_sim_id,
                                                                            [normal_x, normal_y, normal_z],
                                                                            P_quat)
                            flag = 1
                            break
                    if flag == 1:
                        break
                if flag == 0:
                    break
            if nTries >= 10: # This means we could not find a non-colliding particle position.
                print("WARNING: Could not find a non-colliding particle position after motion noise. Moving particle object to noise-less pose. Particle index, object index ", index, obj_index)
                pybullet_env.resetBasePositionAndOrientation(pw_T_par_sim_id, sim_par_cur_pos, sim_par_cur_ori)
        return normal_x, normal_y, normal_z, P_quat

    # add noise
    def add_noise_pose(self, sim_par_cur_pos, sim_par_cur_ori):
        normal_x = self.add_noise_2_par(sim_par_cur_pos[0])
        normal_y = self.add_noise_2_par(sim_par_cur_pos[1])
        normal_z = self.add_noise_2_par(sim_par_cur_pos[2])
        #add noise on ang of each particle
        quat = copy.deepcopy(sim_par_cur_ori)#x,y,z,w
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
        pb_quat = [new_quat[1],new_quat[2],new_quat[3],new_quat[0]]
        new_angle = p_sim.getEulerFromQuaternion(pb_quat)
        P_quat = p_sim.getQuaternionFromEuler(new_angle)
        # pipe.send()
        return normal_x, normal_y, normal_z, P_quat
    
    def single_ray_tracing(self, par_pos, pybullet_env, weight=1, local_obj_visual_by_DOPE_val=0, local_obj_outlier_by_DOPE_val=0):
        pw_T_par_sim_pos = copy.deepcopy(par_pos)
        rayTest_info = pybullet_env.rayTest(pw_T_cam_tf_pos, pw_T_par_sim_pos)
        hit_obj_id = rayTest_info[0][0]
        if SHOW_RAY == True:
            ray_id = pybullet_env.addUserDebugLine(pw_T_cam_tf_pos, pw_T_par_sim_pos, [0,1,0], 2)
        if local_obj_visual_by_DOPE_val == 'motion':
            if hit_obj_id == -1:
                weight = 0.1
                # weight = 0.9
            else:
                weight = 0.9
                # weight = 0.1
        elif local_obj_outlier_by_DOPE_val == 'observation':
            if hit_obj_id == -1:
                weight = weight
            else:
                weight = weight / 2.0
        return weight

    def multi_ray_tracing(self, par_pos, par_ori, pybullet_env, obj_index, weight=1, local_obj_visual_by_DOPE_val=0, local_obj_outlier_by_DOPE_val=0, particle=0):
        
        pw_T_parC_pos = copy.deepcopy(par_pos)
        pw_T_parC_ori = copy.deepcopy(par_ori) # x, y, z, w
        pw_T_parC_ori = quaternion_correction(pw_T_parC_ori)

        # mark
        if OBJECT_NAME_LIST[obj_index] == "soup":
            pw_T_parC_ang = list(p.getEulerFromQuaternion(pw_T_parC_ori))
            pw_T_parC_ang[0] = pw_T_parC_ang[0] + 1.5707963
            pw_T_parC_ori = p.getQuaternionFromEuler(pw_T_parC_ang)


        pw_T_parC_3_3 = np.array(p.getMatrixFromQuaternion(pw_T_parC_ori)).reshape(3, 3)
        pw_T_parC_3_4 = np.c_[pw_T_parC_3_3, pw_T_parC_pos]  # Add position to create 3x4 matrix
        pw_T_parC_4_4 = np.r_[pw_T_parC_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix

        # pw_T_parC_3_3 = transformations.quaternion_matrix(pw_T_parC_ori)
        # pw_T_parC_4_4 = rotation_4_4_to_transformation_4_4(pw_T_parC_3_3, pw_T_parC_pos)
        
        point_list, point_pos_list = generate_point_for_ray(pw_T_parC_pos, pw_T_parC_4_4, obj_index)
        point_pos_list.append(pw_T_parC_pos)
        camera_pos_list = []
        list_length = len(point_pos_list)
        for point_index in range(list_length):
            camera_pos_list.append(pw_T_cam_tf_pos)
            
        rayTestBatch_info = pybullet_env.rayTestBatch(rayFromPositions=camera_pos_list, rayToPositions=point_pos_list)
        # rayTestBatch_info = pybullet_env.rayTestBatch(rayFromPositions=camera_pos_list, rayToPositions=point_pos_list, reportHitNumber=True)
        if SHOW_RAY == True:
            self.show_debug_line(list_length, camera_pos_list, point_pos_list, pybullet_env)
        self_id = particle[obj_index].no_visual_par_id
        
        line_hit_num = 0
        for point_index in range(list_length):
            # hit_obj_id = rayTestBatch_info[point_index]['hitObjectUniqueId']
            hit_obj_id = rayTestBatch_info[point_index][0]
            if (hit_obj_id!=-1) and (hit_obj_id!=self_id):
                line_hit_num = line_hit_num + 1
        visible_score = 1.0 * (list_length - line_hit_num) / list_length

        # local_obj_visual_by_DOPE_val == 0 means DOPE detects the object[obj_index]
        # local_obj_visual_by_DOPE_val == 1 means DOPE does not detect the object[obj_index] and skip this loop
        # local_obj_outlier_by_DOPE_val == 0 means DOPE object[obj_index] correct
        # local_obj_outlier_by_DOPE_val == 1 means DOPE object[obj_index] outlier
        if local_obj_visual_by_DOPE_val==0 and local_obj_outlier_by_DOPE_val==0:
            # visible_score low, weight low
            if visible_score < visible_threshold_dope_is_fresh_list[obj_index]: # 0.5/0.9
                weight = weight / 3.0
                # weight = weight * visible_score
            # visible_score high, weight high
            else:
                weight = weight
        # mark
        # elif local_obj_visual_by_DOPE_val==0 and local_obj_outlier_by_DOPE_val==1:
        #     # visible_score mid, weight high
        #     if visible_threshold_outlier_S_list[obj_index]<=visible_score and visible_score<=visible_threshold_outlier_L_list[obj_index]: # 0.95
        #         # weight = visible_weight_dope_X_smaller_than_threshold_list[obj_index] # 0.75/0.6
        #         weight = weight * visible_weight_dope_X_smaller_than_threshold_list[obj_index] # 0.75
        #         # visible_score > 0.95 high, weight low
        #     elif visible_threshold_outlier_L_list[obj_index] < visible_score:
        #         # weight = visible_weight_outlier_larger_than_threshold_list[obj_index] # 0.25/0.5
        #         weight = weight * (1 - visible_score)
        #     else: # visible_score < 
        #         # weight = visible_weight_outlier_smaller_than_threshold_list[obj_index] # 0.25/0.5
        #         weight = weight * visible_score
        
        # elif local_obj_visual_by_DOPE_val==1 and local_obj_outlier_by_DOPE_val==1:
        else:
            # visible_score<0.95 low, weight high
            if visible_threshold_dope_X_small_list[obj_index]<=visible_score and visible_score<=visible_threshold_dope_X_list[obj_index]: # 0.95
                weight = visible_weight_dope_X_smaller_than_threshold_list[obj_index] * weight # 0.75/0.6
                # weight = weight * (1 - visible_score) # 0.75/0.6
            # visible_score>0.95 high, weight low
            else: 
                weight = visible_weight_dope_X_larger_than_threshold_list[obj_index] * weight # 0.25/0.5

        return weight

    def show_debug_line(self, list_length, camera_pos_list, point_pos_list, p_sim):
        ray_id_list = []
        for list_index in range(list_length):
            ray_id = p_sim.addUserDebugLine(camera_pos_list[list_index], point_pos_list[list_index], [0,1,0], 2)
            ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[0], point_pos_list[1], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[0], point_pos_list[2], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[0], point_pos_list[4], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[1], point_pos_list[3], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[1], point_pos_list[5], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[2], point_pos_list[3], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[2], point_pos_list[6], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[3], point_pos_list[7], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[4], point_pos_list[5], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[4], point_pos_list[6], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[5], point_pos_list[7], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[6], point_pos_list[7], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # self.rays_id_list.append(ray_id_list)

        # ray_id = p_sim.addUserDebugLine(point_pos_list[2], point_pos_list[3], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[3], point_pos_list[5], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[5], point_pos_list[4], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[4], point_pos_list[2], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # # ray_id = p_sim.addUserDebugLine(point_pos_list[34], point_pos_list[35], [0,1,0], 5)
        # # ray_id_list.append(ray_id)
        # # ray_id = p_sim.addUserDebugLine(point_pos_list[35], point_pos_list[37], [0,1,0], 5)
        # # ray_id_list.append(ray_id)
        # # ray_id = p_sim.addUserDebugLine(point_pos_list[37], point_pos_list[36], [0,1,0], 5)
        # # ray_id_list.append(ray_id)
        # # ray_id = p_sim.addUserDebugLine(point_pos_list[36], point_pos_list[34], [0,1,0], 5)
        # # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[2], point_pos_list[34], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[3], point_pos_list[35], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[4], point_pos_list[36], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # ray_id = p_sim.addUserDebugLine(point_pos_list[5], point_pos_list[37], [0,1,0], 5)
        # ray_id_list.append(ray_id)
        # self.rays_id_list.append(ray_id_list)

    def compare_rob_joint(self,real_rob_joint_list_cur,real_robot_joint_pos):
        for i in range(self.joint_num):
            diff = 10
            diff = abs(real_rob_joint_list_cur[i] - real_robot_joint_pos[i])
            if diff > 0.005:
                return 1
        return 0
    
    # change particle parameters
    def change_obj_parameters(self, pybullet_env, par_id):
        mass_a = self.take_easy_gaussian_value(mass_mean, mass_sigma)
        if mass_a < 0.001:
            mass_a = 0.05
        lateralFriction = self.take_easy_gaussian_value(friction_mean, friction_sigma)
        spinningFriction = self.take_easy_gaussian_value(friction_mean, friction_sigma)
        rollingFriction = self.take_easy_gaussian_value(friction_mean, friction_sigma)
        if lateralFriction < 0.001:
            lateralFriction = 0.001
        if spinningFriction < 0.001:
            spinningFriction = 0.001
        if rollingFriction < 0.001:
            rollingFriction = 0.001
        restitution = self.take_easy_gaussian_value(restitution_mean, restitution_sigma)
        # if restitution > 1:
        # mass_a = 0.351
        # fricton_b = 0.30
        # mean_damping = 0.4
        # mean_stiffness = 0.9
        # contactStiffness = self.take_easy_gaussian_value(mean_stiffness, 0.3)
        # contactDamping = self.take_easy_gaussian_value(mean_damping, 0.1)
        pybullet_env.changeDynamics(par_id, -1, mass = mass_a, 
                                    lateralFriction = lateralFriction, 
                                    spinningFriction = spinningFriction, 
                                    rollingFriction = rollingFriction, 
                                    restitution = restitution)
                                    #contactStiffness=contactStiffness,
                                    #contactDamping=contactDamping)

    def get_item_pos(self,pybullet_env,item_id):
        item_info = pybullet_env.getBasePositionAndOrientation(item_id)
        return item_info[0],item_info[1]

    def add_noise_2_par(self,current_pos):
        mean = current_pos
        sigma = pos_noise
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise

    def add_noise_2_ang(self,cur_angle):
        mean = cur_angle
        sigma = boss_sigma_obs_ang
        sigma = ang_noise
        new_angle_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_angle_is_added_noise

    def take_easy_gaussian_value(self,mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal

    def normal_distribution(self, x, mean, sigma):
        return sigma * np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)
    
    def normal_distribution_prob(self, x, mean, sigma):
        x_1 = abs(x)
        x_0 = x_1 * (-1)
        cdf_upper = norm.cdf(x_1, mean, sigma)
        cdf_lower = norm.cdf(x_0, mean, sigma)
        prob = cdf_upper - cdf_lower
        return prob

    def normalize_particles(self):
        flag_1 = 0
        tot_weight = sum([particle.w for particle in self.particle_cloud])
        if tot_weight == 0:
            # print("Error!,PBPF particles total weight is 0")
            tot_weight = 1
            flag_1 = 1
        for particle in self.particle_cloud:
            if flag_1 == 0:
                particle_w = particle.w/tot_weight
                particle.w = particle_w
            else:
                particle.w = 1/PARTICLE_NUM
                
    # old particle angle
    def resample_particles(self):
        particles_w = []
        newParticles = []
        n_particle = len(self.particle_cloud)
        for particle in self.particle_cloud:
            particles_w.append(particle.w)
        particle_array= np.random.choice(a = n_particle, size = n_particle, replace=True, p= particles_w)
        particle_array_list = list(particle_array)
        for index,i in enumerate(particle_array_list):
            particle = Particle(self.particle_cloud[i].pos,
                                self.particle_cloud[i].ori,
                                1.0/PARTICLE_NUM, index)
            newParticles.append(particle)
        self.particle_cloud = copy.deepcopy(newParticles)
    
    # new
    def resample_particles_update(self, pw_T_obj_obse_objects_pose_list):
        local_pick_particle_rate = pick_particle_rate
        # mark
        # if sum(self.do_obs_update_list) == OBJECT_NUM:
        #     local_pick_particle_rate = 0.0
        pw_T_obj_obse_objs_pose_list = copy.deepcopy(pw_T_obj_obse_objects_pose_list)
        n_particle = len(self.particle_cloud)
        par_num_on_obse = int(math.ceil(n_particle * local_pick_particle_rate))
        par_num_for_resample = int(n_particle) - int(par_num_on_obse)
        
        newParticles_list = [[]*self.obj_num for _ in range(n_particle)]
        
        particles_w = []
        base_w = 0
        base_w_list = []
        base_w_list.append(base_w)
        particle_array_list = []

        # mark
        if USING_D_FLAG == True:
            if DEPTH_DIFF_VALUE_0_1_FLAG == True:
                # score_that_particle_get: high->high weight; low->low weight
                weight_depth_img_array = copy.deepcopy(self.depth_value_difference_list)
            else:
                weight_depth_img_array = self.computeWeightFromDepthImage(self.depth_value_difference_list)    
        

        for index, particle in enumerate(self.particle_cloud):
            each_par_weight = 1
            for obj_index in range(self.obj_num):
                each_par_weight = each_par_weight * particle[obj_index].w
            
                           
            # mark
            if USING_D_FLAG == True:
                # print("weight_depth_img: ",weight_depth_img,"; each_par_weight: ", each_par_weight)
                each_par_weight = each_par_weight * weight_depth_img_array[index]

                if PRINT_SCORE_FLAG == True and OBJECT_NUM == 2:
                    figure = 5
                    depth_score_r = round(weight_depth_img_array[index], figure)
                    cracker_obs_score_r = round(particle[0].w, figure)
                    soup_obs_score_r = round(particle[1].w, figure)
                    total_score_r = round(each_par_weight, figure)
                    print("Update_Time: ",_particle_update_time,"; Index:", index, "; depth_score: ",depth_score_r, "; cracker_obs_score: ", cracker_obs_score_r, "; soup_obs_score: ", soup_obs_score_r, "; total_score: ",total_score_r) 
                    print("Cracker Error: ", self.cracker_dis_error[index], self.cracker_ang_error[index], "; Weight: ", self.cracker_weight_before_ray[index], self.cracker_weight__after_ray[index])
                    print("S o u p Error: ", self.soup_dis_error[index], self.soup_ang_error[index], "; Weight: ", self.soup_weight_before_ray[index], self.soup_weight__after_ray[index])
                    
            particles_w.append(each_par_weight) # to compute the sum
            base_w = base_w + each_par_weight
            base_w_list.append(base_w) # [0, 0.02, 0.025, 0.029, 0.031, ..., sum]
        w_sum = sum(particles_w)
        r = random.uniform(0, w_sum)
        
        for index in range(par_num_for_resample):
            if w_sum > 0.00000001:
                position = (r + index * w_sum / PARTICLE_NUM) % w_sum
                position_index = self.computePosition(position, base_w_list)
                particle_array_list.append(position_index)
            else:
                particle_array_list.append(index) # [45, 45, 1, 4, 6, 6, ..., 43]
        index = -1
        
        for obj_index in range(self.obj_num):
            obse_obj_pos = pw_T_obj_obse_objs_pose_list[obj_index].pos
            obse_obj_ori = pw_T_obj_obse_objs_pose_list[obj_index].ori # pybullet x,y,z,w
            for index, i in enumerate(particle_array_list): # particle angle
                particle = Particle(self.particle_cloud[i][obj_index].par_name,
                                    self.particle_cloud[index][obj_index].visual_par_id,
                                    self.particle_cloud[index][obj_index].no_visual_par_id,
                                    self.particle_cloud[i][obj_index].pos,
                                    self.particle_cloud[i][obj_index].ori,
                                    1.0/PARTICLE_NUM, 
                                    index,
                                    self.particle_cloud[i][obj_index].linearVelocity,
                                    self.particle_cloud[i][obj_index].angularVelocity)
                newParticles_list[index].append(particle)
            
            # only work when "local_pick_particle_rate != 0"
            for index_leftover in range(par_num_on_obse):
                index = index + 1
                particle = Particle(self.particle_cloud[index_leftover][obj_index].par_name,
                                    self.particle_cloud[index][obj_index].visual_par_id,
                                    self.particle_cloud[index][obj_index].no_visual_par_id,
                                    obse_obj_pos,
                                    obse_obj_ori,
                                    1.0/PARTICLE_NUM, 
                                    index,
                                    self.particle_cloud[index_leftover][obj_index].linearVelocity,
                                    self.particle_cloud[index_leftover][obj_index].angularVelocity)
                newParticles_list[index].append(particle)
                
        self.particle_cloud = copy.deepcopy(newParticles_list)   

    def computeWeightFromDepthImage(self, depth_value_difference_list):
        depth_difference_max = 0
        depth_difference_min = min(depth_value_difference_list)
        depth_value_difference_list_ = copy.deepcopy(depth_value_difference_list)
        depth_value_difference_list_array_ = np.array(depth_value_difference_list_)
        depth_value_difference_list_array_sub = depth_value_difference_list_array_ - depth_difference_min
        if depth_value_difference_list_array_sub.ndim == 1:
            depth_difference_max = max(depth_value_difference_list_array_sub)
        else:
            print("Error: depth_value_difference_list_array_sub.ndim != 1")
        mean = 0
        sigma = depth_difference_max / 3
        sigma = 0.03
        vectorized_function = np.vectorize(self.array_normal_distribution)
        weight_depth_img_array = vectorized_function(depth_value_difference_list_array_sub, mean, sigma)

        return weight_depth_img_array

    def array_normal_distribution(self, x, mean, sigma):
        return sigma * np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)

    def array_normal_distribution_prob(self, x, mean, sigma):
        x_1 = abs(x)
        x_0 = x_1 * (-1)
        cdf_upper = norm.cdf(x_1, mean, sigma)
        cdf_lower = norm.cdf(x_0, mean, sigma)
        prob = cdf_upper - cdf_lower
        return prob

    def computePosition(self, position, base_w_list):
        for index in range(1, len(base_w_list)):
            if position <= base_w_list[index] and position > base_w_list[index - 1]:
                return index - 1
            else:
                continue
               
    def set_particle_in_each_sim_env(self): # particle angle
        for obj_index in range(self.obj_num):
            for index, pybullet_env in enumerate(self.pybullet_env_id_collection):
                # pw_T_par_sim_id = self.particle_no_visual_id_collection[index][obj_index]
                pw_T_par_sim_id = self.particle_cloud[index][obj_index].no_visual_par_id
                par_obj_pos = self.particle_cloud[index][obj_index].pos
                par_obj_ori = self.particle_cloud[index][obj_index].ori
                pybullet_env.resetBasePositionAndOrientation(pw_T_par_sim_id,
                                                             par_obj_pos,
                                                             par_obj_ori)
        return

    # set
    def set_particle_in_each_sim_env_single(self, index, obj_index, particle_pos, particle_ori):
        pybullet_env = self.pybullet_env_id_collection[index]
        pw_T_par_sim_id = self.particle_cloud[index][obj_index].no_visual_par_id
        pybullet_env.resetBasePositionAndOrientation(pw_T_par_sim_id,
                                                     particle_pos,
                                                     particle_ori)
         
    def draw_contrast_figure(self, estimated_object_pos, observation):
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

    def compute_estimate_pos_of_object(self, particle_cloud): # need to change
        esti_objs_cloud = []
        dis_std_list = []
        ang_std_list = []
        # remenber after resampling weight of each particle is the same
        for obj_index in range(self.obj_num):
            x_set = 0
            y_set = 0
            z_set = 0
            w_set = 0
            quaternions = []
            qws = []
            for index, particle in enumerate(particle_cloud):
                x_set = x_set + particle[obj_index].pos[0] * particle[obj_index].w
                y_set = y_set + particle[obj_index].pos[1] * particle[obj_index].w
                z_set = z_set + particle[obj_index].pos[2] * particle[obj_index].w
                q = quaternion_correction(particle[obj_index].ori)
                qws.append(particle[obj_index].w)
                quaternions.append([q[0], q[1], q[2], q[3]])
                w_set = w_set + particle[obj_index].w
            q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
            ###################################
            esti_obj_pos_x = x_set/w_set
            esti_obj_pos_y = y_set/w_set
            esti_obj_pos_z = z_set/w_set
            esti_obj_pos = [esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z]
            esti_obj_ori_x = q[0]
            esti_obj_ori_y = q[1]
            esti_obj_ori_z = q[2]
            esti_obj_ori_w = q[3]
            esti_obj_ori = [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]
            mean_pose = [esti_obj_pos, esti_obj_ori]
            dis_std, ang_std = self.compute_std(mean_pose, particle_cloud)
            ###################################
            est_obj_pose = Object_Pose(particle[obj_index].par_name, estimated_object_set[obj_index].obj_id, [x_set/w_set, y_set/w_set, z_set/w_set],  [q[0], q[1], q[2], q[3]], obj_index)
            esti_objs_cloud.append(est_obj_pose)
            dis_std_list.append(dis_std)
            ang_std_list.append(ang_std)
        return esti_objs_cloud, dis_std_list, ang_std_list

    def compute_std(self, mean_pose, particle_cloud):
        mean_pos = copy.deepcopy(mean_pose[0])
        mean_ori = copy.deepcopy(mean_pose[1]) # x,y,z,w
        dis_list = []
        ang_list = []
        for index, particle in enumerate(particle_cloud):
            pos_set = copy.deepcopy(particle[obj_index].pos)
            q = quaternion_correction(particle[obj_index].ori)
            ori_set = copy.deepcopy(q)
            dis_mean_eachPar = compute_pos_err_bt_2_points(pos_set, mean_pos)
            ang_mean_eachPar = compute_ang_err_bt_2_points(ori_set, mean_ori)
            dis_list.append(dis_mean_eachPar)
            ang_list.append(ang_mean_eachPar)
        dis_std = np.std(dis_list)
        ang_std = np.std(ang_list)
        return dis_std, ang_std

    def is_cv_depth_image(self, image):
        if not isinstance(image, np.ndarray):
            print("not a cv format image")
            return False
        if len(image.shape) not in [2, 3]:
            print("not a cv format image")
            return False
        if len(image.shape) == 3 and image.shape[2] != 1:
            print("not a cv format image")
            return False
        if image.dtype not in [np.float32, np.uint16]:
            print("not a cv format image")
            return False
        print("cv format image")
        return True

    def is_ros_depth_image(self, image_msg):
        if not isinstance(image_msg, Image):
            print("not a sensor_msgs/Image type")
            return
        if image_msg.encoding == "32FC1":
            print("depth image encoding: 16UC1")
        elif image_msg.encoding == "16UC1":
            print("depth image encoding: 16UC1")
        else:
            print("image is not depth image")

#Class of Constant-velocity Particle Filtering
class CVPFMove():
    def __init__(self, obj_num=0):
        # init internals   
        self.obj_num = obj_num
        self.particle_cloud_CV = copy.deepcopy(initial_parameter.particle_cloud_CV)
        self.particle_no_visual_id_collection_CV = copy.deepcopy(initial_parameter.particle_no_visual_id_collection_CV)
        self.pybullet_env_id_collection_CV = copy.deepcopy(initial_parameter.pybullet_particle_env_collection_CV)
        self.pybullet_sim_fake_robot_id_collection = copy.deepcopy(initial_parameter.fake_robot_id_collection)
        self.pybullet_sim_other_object_id_collection = copy.deepcopy(initial_parameter.other_object_id_collection)
        self.object_estimate_pose_x = []
        self.object_estimate_pose_y = []
        self.object_real_____pose_x = []
        self.object_real_____pose_y = []
        self.rays_id_list = []
        self.ray_list_empty = True

    def compute_pos_err_bt_2_points(self,pos1,pos2):
        x_d = pos1[0]-pos2[0]
        y_d = pos1[1]-pos2[1]
        z_d = pos1[2]-pos2[2]
        distance = math.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)
        return distance

    # executed_control
    def update_particle_filter_CV(self, pw_T_obj_obse_objects_pose_list):
        global flag_record_CVPF
        global flag_record
        # motion model
        self.motion_update_CV(pw_T_obj_obse_objects_pose_list)
        # observation model
        # if sum(self.do_obs_update_list) < OBJECT_NUM:
            
        self.observation_update_CV(pw_T_obj_obse_objects_pose_list)
            
            
        # if (VERSION == "ray" or VERSION == "multiray"):
        #     self.resample_particles_CV_update(pw_T_obj_obse_objects_pose_list)
        #     self.set_particle_in_each_sim_env_CV()
            
        # Compute mean of particles
        object_estimate_pose_CV, dis_std_list, ang_std_list = self.compute_estimate_pos_of_object(self.particle_cloud_CV)
        
        boss_est_pose_CVPF.append(object_estimate_pose_CV)

        # publish pose of particles
        publish_par_pose_info(self.particle_cloud_CV)
        publish_esti_pose_info(object_estimate_pose_CV)
        return object_estimate_pose_CV, dis_std_list, ang_std_list, self.particle_cloud_CV

    def isAnyParticleInContact(self):
        for index, particle in enumerate(self.particle_cloud_CV):
            for obj_index in range(OBJECT_NUM):
                # get pose from particle
                pw_T_par_sim_pw_env = self.pybullet_env_id_collection_CV[index]
                # pw_T_par_sim_id = self.particle_no_visual_id_collection[index][obj_index]
                pw_T_par_sim_id = particle[obj_index].no_visual_par_id
                # sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(self.pybullet_env_id_collection[index], pw_T_par_sim_id)
                sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(pw_T_par_sim_pw_env, pw_T_par_sim_id)
                # check contact 
                pmin, pmax = pw_T_par_sim_pw_env.getAABB(pw_T_par_sim_id)
                collide_ids = pw_T_par_sim_pw_env.getOverlappingObjects(pmin, pmax)
                length = len(collide_ids)
                for t_i in range(length):
                    # print("body id: ",collide_ids[t_i][1])
                    if collide_ids[t_i][1] == 8 or collide_ids[t_i][1] == 9 or collide_ids[t_i][1] == 10 or collide_ids[t_i][1] == 11:
                        return True
        return False
    
    def robot_arm_move_CV(self, position):
        for index, pw_sim_env in enumerate(self.pybullet_env_id_collection_CV):
            fake_rob_id = self.pybullet_sim_fake_robot_id_collection[index]
            num_joints = 9
            for joint_index in range(num_joints):
                if joint_index == 7 or joint_index == 8:
                    pw_sim_env.resetJointState(fake_rob_id,
                                                 joint_index+2,
                                                 targetValue=position[joint_index])
                else:
                    pw_sim_env.resetJointState(fake_rob_id,
                                                 joint_index,
                                                 targetValue=position[joint_index])
                
    def motion_update_CV(self, pw_T_obj_obse_objects_pose_list):
        # t0, t1: use observation data (obs0, obs1) to update motion
        if flag_update_num_CV < 2:
            length = len(boss_obs_pose_CVPF)
            obs_curr_pose_list = copy.deepcopy(boss_obs_pose_CVPF[length-1]) # [obse_obj1_n,   obse_obj2_n]
            obs_last_pose_list = copy.deepcopy(boss_obs_pose_CVPF[length-1]) # [obse_obj1_n-1, obse_obj2_n-1]
            for obj_index in range (self.obj_num):
                obs_curr_pose = obs_curr_pose_list[obj_index] # class objext
                obs_last_pose = obs_last_pose_list[obj_index] # class objext
                obs_last_pos = obs_last_pose.pos           
                obs_last_ori = obs_last_pose.ori
                obs_curr_pos = obs_curr_pose.pos
                # print("obs_last_pos: ",obs_last_pos)
                # print("obs_curr_pos: ",obs_curr_pos)
                obs_curr_ori = obs_curr_pose.ori
                obsO_T_obsN = compute_transformation_matrix(obs_last_pos, obs_last_ori, obs_curr_pos, obs_curr_ori)
                parO_T_parN = copy.deepcopy(obsO_T_obsN)
                self.update_particle_in_motion_model_CV(obj_index, parO_T_parN, pw_T_obj_obse_objects_pose_list)
        # after t1: use (est0, est1) to update motion
        else:
            length = len(boss_est_pose_CVPF)
            est_curr_pose_list = copy.deepcopy(boss_est_pose_CVPF[length-1]) # [esti_obj1_n,   esti_obj2_n]
            est_last_pose_list = copy.deepcopy(boss_est_pose_CVPF[length-2]) # [esti_obj1_n,   esti_obj2_n]
            for obj_index in range (self.obj_num):
                est_curr_pose = est_curr_pose_list[obj_index]
                est_last_pose = est_last_pose_list[obj_index]
                est_curr_pos = est_curr_pose.pos
                est_curr_ori = est_curr_pose.ori
                est_last_pos = est_last_pose.pos
                # print("est_last_pos: ",est_last_pos)
                # print("est_curr_pos: ",est_curr_pos)
                est_last_ori = est_last_pose.ori
                estO_T_estN = compute_transformation_matrix(est_last_pos, est_last_ori, est_curr_pos, est_curr_ori)
                parO_T_parN = copy.deepcopy(estO_T_estN)
                self.update_particle_in_motion_model_CV(obj_index, parO_T_parN, pw_T_obj_obse_objects_pose_list)
        return

    def observation_update_CV(self, pw_T_obj_obse_objects_pose_list):
        for obj_index in range(self.obj_num):
            obse_obj_pos = pw_T_obj_obse_objects_pose_list[obj_index].pos
            obse_obj_ori = pw_T_obj_obse_objects_pose_list[obj_index].ori # pybullet x,y,z,w
            # make sure theta between -pi and pi
            obse_obj_ori_corr = quaternion_correction(obse_obj_ori)

            for index,particle in enumerate(self.particle_cloud_CV):
                particle_x = particle[obj_index].pos[0]
                particle_y = particle[obj_index].pos[1]
                particle_z = particle[obj_index].pos[2]
                mean = 0
                # position weight
                dis_x = abs(particle_x - obse_obj_pos[0])
                dis_y = abs(particle_y - obse_obj_pos[1])
                dis_z = abs(particle_z - obse_obj_pos[2])
                dis_xyz = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
                weight_xyz = self.normal_distribution(dis_xyz, mean, boss_sigma_obs_pos)
                # rotation weight    
                par_ori = quaternion_correction(particle[obj_index].ori)
                obse_obj_quat = Quaternion(x=obse_obj_ori_corr[0], 
                                           y=obse_obj_ori_corr[1], 
                                           z=obse_obj_ori_corr[2], 
                                           w=obse_obj_ori_corr[3]) # Quaternion(): w,x,y,z
                par_quat = Quaternion(x=par_ori[0], y=par_ori[1], z=par_ori[2], w=par_ori[3])
                err_bt_par_obse = par_quat * obse_obj_quat.inverse
                cos_theta_over_2 = err_bt_par_obse.w
                sin_theta_over_2 = math.sqrt(err_bt_par_obse.x ** 2 + err_bt_par_obse.y ** 2 + err_bt_par_obse.z ** 2)
                theta_over_2 = math.atan2(sin_theta_over_2, cos_theta_over_2)
                theta = theta_over_2 * 2
                weight_ang = self.normal_distribution(theta, mean, boss_sigma_obs_ang)
                weight = weight_xyz * weight_ang
            
                particle[obj_index].w = weight
        # old resample function
        # Flag = self.normalize_particles_CV()
        # self.resample_particles_CV()
        # new resample function
        self.resample_particles_CV_update(pw_T_obj_obse_objects_pose_list)
        self.set_particle_in_each_sim_env_CV()
        return

    def update_particle_in_motion_model_CV(self, obj_index, parO_T_parN, pw_T_obj_obse_objects_list):
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection_CV):
            pw_T_parO_pos = copy.deepcopy(self.particle_cloud_CV[index][obj_index].pos)
            pw_T_parO_ori = copy.deepcopy(self.particle_cloud_CV[index][obj_index].ori)
            # pw_T_parO_3_3 = transformations.quaternion_matrix(pw_T_parO_ori)
            # pw_T_parO_4_4 = rotation_4_4_to_transformation_4_4(pw_T_parO_3_3,pw_T_parO_pos)

            pw_T_parO_3_3 = np.array(p.getMatrixFromQuaternion(pw_T_parO_ori)).reshape(3, 3)
            pw_T_parO_3_4 = np.c_[pw_T_parO_3_3, pw_T_parO_pos]  # Add position to create 3x4 matrix
            pw_T_parO_4_4 = np.r_[pw_T_parO_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix


            pw_T_parN = np.dot(pw_T_parO_4_4, parO_T_parN)
            pw_T_parN_pos = [pw_T_parN[0][3], pw_T_parN[1][3], pw_T_parN[2][3]]
            pw_T_parN_ori = transformations.quaternion_from_matrix(pw_T_parN)
            pw_T_parN_ori = quaternion_correction(pw_T_parN_ori)
            # pw_T_parN_ang = pybullet_env.getEulerFromQuaternion(pw_T_parN_ori)
            
            # add noise on particle filter
            normal_x = self.add_noise_2_par(pw_T_parN_pos[0])
            normal_y = self.add_noise_2_par(pw_T_parN_pos[1])
            normal_z = self.add_noise_2_par(pw_T_parN_pos[2])
            
            # quat = copy.deepcopy(pw_T_obj_obse_objects_list[obj_index].ori) # use ori from dope
            quat = copy.deepcopy(pw_T_parN_ori) # x,y,z,w / use ori from CV
            quat_QuatStyle = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3]) # w,x,y,z
            random_dir = random.uniform(0, 2*math.pi)
            z_axis = random.uniform(-1,1)
            x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
            y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
            angle_noise = self.add_noise_2_ang(0)
            w_quat = math.cos(angle_noise/2.0)
            x_quat = math.sin(angle_noise/2.0) * x_axis
            y_quat = math.sin(angle_noise/2.0) * y_axis
            z_quat = math.sin(angle_noise/2.0) * z_axis
            ### nois_quat(w,x,y,z); new_quat(w,x,y,z)
            nois_quat = Quaternion(x=x_quat,y=y_quat,z=z_quat,w=w_quat)
            new_quat = nois_quat * quat_QuatStyle
            ### pb_quat(x,y,z,w)
            pb_quat = [new_quat[1], new_quat[2], new_quat[3], new_quat[0]]

            self.particle_cloud_CV[index][obj_index].pos = [normal_x, normal_y, normal_z]
            self.particle_cloud_CV[index][obj_index].ori = copy.deepcopy(pb_quat)
            
    def get_item_pos(self,pybullet_env, item_id):
        item_info = pybullet_env.getBasePositionAndOrientation(item_id)
        return item_info[0], item_info[1]

    def add_noise_2_par(self,current_pos):
        mean = current_pos
        sigma = pos_noise
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def add_noise_2_ang(self,cur_angle):
        mean = cur_angle
        sigma = boss_sigma_obs_ang
        sigma = ang_noise
        new_angle_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_angle_is_added_noise

    def take_easy_gaussian_value(self,mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal

    def normal_distribution(self, x, mean, sigma):
        return sigma * np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)
    
    def normal_distribution_prob(self, x, mean, sigma):
        x_1 = abs(x)
        x_0 = x_1 * (-1)
        cdf_upper = norm.cdf(x_1, mean, sigma)
        cdf_lower = norm.cdf(x_0, mean, sigma)
        prob = cdf_upper - cdf_lower
        return prob
    
    def normalize_particles_CV(self):
        flag_1 = 0
        tot_weight = sum([particle.w for particle in self.particle_cloud_CV])
        if tot_weight == 0:
            # print("Error!,CVPF particles total weight is 0")
            tot_weight = 1
            flag_1 = 1
        for particle in self.particle_cloud_CV:
            if flag_1 == 0:
                particle_w = particle.w/tot_weight
                particle.w = particle_w
            else:
                particle.w = 1.0/PARTICLE_NUM
    
    # old particle angle
    def resample_particles_CV(self):
        particles_w = []
        newParticles = []
        n_particle = len(self.particle_cloud_CV)
        for particle in self.particle_cloud_CV:
            particles_w.append(particle.w)
        particle_array= np.random.choice(a = n_particle, size = n_particle, replace=True, p= particles_w)
        particle_array_list = list(particle_array)
        for index,i in enumerate(particle_array_list):
            particle = Particle(self.particle_cloud_CV[i].pos,
                                self.particle_cloud_CV[i].ori,
                                1.0/PARTICLE_NUM,index)
            newParticles.append(particle)
        self.particle_cloud_CV = copy.deepcopy(newParticles)
        
    def resample_particles_CV_update(self, pw_T_obj_obse_objects_pose_list):
        local_pick_particle_rate = pick_particle_rate

        # makr
        # if sum(self.do_obs_update_list) == OBJECT_NUM:
        #     local_pick_particle_rate = 0.0
        pw_T_obj_obse_objs_pose_list = copy.deepcopy(pw_T_obj_obse_objects_pose_list)
        n_particle = len(self.particle_cloud_CV)
        par_num_on_obse = int(math.ceil(n_particle * local_pick_particle_rate))
        par_num_for_resample = int(n_particle) - int(par_num_on_obse)

        newParticles_list = [[]*self.obj_num for _ in range(n_particle)]
        for obj_index in range(self.obj_num):
            obse_obj_pos = pw_T_obj_obse_objs_pose_list[obj_index].pos
            obse_obj_ori = pw_T_obj_obse_objs_pose_list[obj_index].ori # pybullet x,y,z,w

            particles_w = []
            # newParticles = []
            base_w = 0
            base_w_list = []
            base_w_list.append(base_w)
            particle_array_list = []
            # compute sum of weight
            for particle in self.particle_cloud_CV:
                particles_w.append(particle[obj_index].w)
                base_w = base_w + particle[obj_index].w
                base_w_list.append(base_w)
            w_sum = sum(particles_w)
            r = random.uniform(0, w_sum)

            for index in range(par_num_for_resample):
                if w_sum > 0.00000001:
                    position = (r + index * w_sum / PARTICLE_NUM) % w_sum
                    position_index = self.computePosition_CV(position, base_w_list)
                    particle_array_list.append(position_index)
                else:
                    particle_array_list.append(index)

            for index,i in enumerate(particle_array_list): # particle angle
                particle = Particle(self.particle_cloud_CV[i][obj_index].par_name,
                                    self.particle_cloud_CV[index][obj_index].visual_par_id,
                                    self.particle_cloud_CV[index][obj_index].no_visual_par_id,
                                    self.particle_cloud_CV[i][obj_index].pos,
                                    self.particle_cloud_CV[i][obj_index].ori,
                                    1.0/PARTICLE_NUM, 
                                    index,
                                    self.particle_cloud_CV[i][obj_index].linearVelocity,
                                    self.particle_cloud_CV[i][obj_index].angularVelocity)
                newParticles_list[index].append(particle)
            for index_leftover in range(par_num_on_obse):
                index = index + 1
                particle = Particle(self.particle_cloud_CV[index_leftover][obj_index].par_name,
                                    self.particle_cloud_CV[index][obj_index].visual_par_id,
                                    self.particle_cloud_CV[index][obj_index].no_visual_par_id,
                                    obse_obj_pos,
                                    obse_obj_ori,
                                    1.0/PARTICLE_NUM, 
                                    index,
                                    self.particle_cloud_CV[index_leftover][obj_index].linearVelocity,
                                    self.particle_cloud_CV[index_leftover][obj_index].angularVelocity)
                newParticles_list[index].append(particle)

                # newParticles.append(particle)
        self.particle_cloud_CV = copy.deepcopy(newParticles_list)
        
    def computePosition_CV(self, position, base_w_list):
        for index in range(1, len(base_w_list)):
            if position <= base_w_list[index] and position > base_w_list[index - 1]:
                return index - 1
            else:
                continue
            
    def set_particle_in_each_sim_env_CV(self):
        for obj_index in range(self.obj_num):
            for index, pybullet_env in enumerate(self.pybullet_env_id_collection_CV):
                pw_T_par_sim_id = self.particle_cloud_CV[index][obj_index].no_visual_par_id
                par_obj_pos = self.particle_cloud_CV[index][obj_index].pos
                par_obj_ori = self.particle_cloud_CV[index][obj_index].ori
                pybullet_env.resetBasePositionAndOrientation(pw_T_par_sim_id,
                                                             par_obj_pos,
                                                             par_obj_ori)
        return

    def draw_contrast_figure(self,estimated_object_pos,observation):
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
        esti_objs_cloud = []
        dis_std_list = []
        ang_std_list = []
        for obj_index in range(self.obj_num):
            x_set = 0
            y_set = 0
            z_set = 0
            w_set = 0
            quaternions = []
            qws = []
            for index, particle in enumerate(particle_cloud):
                x_set = x_set + particle[obj_index].pos[0] * particle[obj_index].w
                y_set = y_set + particle[obj_index].pos[1] * particle[obj_index].w
                z_set = z_set + particle[obj_index].pos[2] * particle[obj_index].w
                q = quaternion_correction(particle[obj_index].ori)
                qws.append(particle[obj_index].w)
                quaternions.append([q[0], q[1], q[2], q[3]])
                w_set = w_set + particle[obj_index].w
            q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
            ###################################
            esti_obj_pos_x = x_set/w_set
            esti_obj_pos_y = y_set/w_set
            esti_obj_pos_z = z_set/w_set
            esti_obj_pos = [esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z]
            esti_obj_ori_x = q[0]
            esti_obj_ori_y = q[1]
            esti_obj_ori_z = q[2]
            esti_obj_ori_w = q[3]
            esti_obj_ori = [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]
            mean_pose = [esti_obj_pos, esti_obj_ori]
            dis_std, ang_std = self.compute_std(mean_pose, particle_cloud)
            ###################################
            est_obj_pose = Object_Pose(particle[obj_index].par_name,
                                       estimated_object_set[obj_index].obj_id,
                                       [x_set/w_set, y_set/w_set, z_set/w_set],
                                       [q[0], q[1], q[2], q[3]], obj_index)
            esti_objs_cloud.append(est_obj_pose)
            dis_std_list.append(dis_std)
            ang_std_list.append(ang_std)
        return esti_objs_cloud, dis_std_list, ang_std_list
    
    def compute_std(self, mean_pose, particle_cloud):
        mean_pos = copy.deepcopy(mean_pose[0])
        mean_ori = copy.deepcopy(mean_pose[1]) # x,y,z,w
        dis_list = []
        ang_list = []
        for index, particle in enumerate(particle_cloud):
            pos_set = copy.deepcopy(particle[obj_index].pos)
            q = quaternion_correction(particle[obj_index].ori)
            ori_set = copy.deepcopy(q)
            dis_mean_eachPar = compute_pos_err_bt_2_points(pos_set, mean_pos)
            ang_mean_eachPar = compute_ang_err_bt_2_points(ori_set, mean_ori)
            dis_list.append(dis_mean_eachPar)
            ang_list.append(ang_mean_eachPar)
        dis_std = np.std(dis_list)
        ang_std = np.std(ang_list)
        return dis_std, ang_std

# function independent of Class
# add position into transformation matrix
# def rotation_4_4_to_transformation_4_4(rotation_4_4, pos):
#     rotation_4_4[0][3] = pos[0]
#     rotation_4_4[1][3] = pos[1]
#     rotation_4_4[2][3] = pos[2]
#     return rotation_4_4
# compute the position distance between two objects

# Jax
@jit
def _extractValues(matrix, positions):         
    values = matrix[positions[:, 0], positions[:, 1]]     
    return values 

# @jit
# def _compute_projection_parameters(fov, resolution):
#     h, w = resolution
#     # f = 0.5 * w / jnp.tan(fov * 0.5)  # fov: horizontal
#     f = 0.5 * h / jnp.tan(fov * 0.5)  # fov: vertical
#     return f
 
@jit
def _ortho_to_persp(depth_ortho, fy, cx, cy):
    y, x = jnp.indices(depth_ortho.shape)
    z = depth_ortho
    x_persp = (x - cx) * z / fy
    y_persp = (y - cy) * z / fy
    depth_persp = jnp.sqrt(x_persp**2 + y_persp**2 + z**2)
    return depth_persp
 
@jit
def _persp_to_ortho(depth_persp, fy, cx, cy):    
    y, x = jnp.indices(depth_persp.shape)   
    z = depth_persp 
    depth_ortho = z / jnp.sqrt(((x - cx) / fy)**2 + ((y - cy) / fy)**2 + 1)     
    return depth_ortho

@jit
def _threshold_array_optimized(arr):
    modified_array = (arr > DEPTH_DIFF_VALUE_0_1_THRESHOLD).astype(jnp.int32)
    num_zeros = jnp.sum(modified_array == 0)
    return modified_array, num_zeros


# ===============================================================================================================

def compute_pos_err_bt_2_points(pos1, pos2):
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
# compute the angle distance between two objects
def compute_ang_err_bt_2_points(object1_ori, object2_ori):
    #[x, y, z, w]
    obj1_ori = copy.deepcopy(object1_ori)
    obj2_ori = copy.deepcopy(object2_ori)
    obj1_ori_quat = quaternion_correction(obj1_ori) # x,y,z,w
    obj2_ori_quat = quaternion_correction(obj2_ori) # x,y,z,w

    #[w, x, y, z]
    obj1_quat = Quaternion(x = obj1_ori_quat[0], y = obj1_ori_quat[1], z = obj1_ori_quat[2], w = obj1_ori_quat[3]) # Quaternion(): w,x,y,z
    obj2_quat = Quaternion(x = obj2_ori_quat[0], y = obj2_ori_quat[1], z = obj2_ori_quat[2], w = obj2_ori_quat[3]) # Quaternion(): w,x,y,z
    diff_bt_o1_o2 = obj2_quat * obj1_quat.inverse
    cos_theta_over_2 = diff_bt_o1_o2.w
    sin_theta_over_2 = math.sqrt(diff_bt_o1_o2.x ** 2 + diff_bt_o1_o2.y ** 2 + diff_bt_o1_o2.z ** 2)
    theta_over_2 = math.atan2(sin_theta_over_2, cos_theta_over_2)
    theta = theta_over_2 * 2
    theta = abs(theta)
    return theta

def compute_diff_bt_two_pose(obj_index, particle_cloud_pub, pw_T_obj_obse_pose_new):
    par_cloud_for_compute = copy.deepcopy(particle_cloud_pub)
    obj_obse_pose_new = copy.deepcopy(pw_T_obj_obse_pose_new)
    obj_obse_pos_new = obj_obse_pose_new[0]
    obj_obse_ori_new = obj_obse_pose_new[1]
    par_dis_list = []
    par_ang_list = []
    par_cloud_length = len(par_cloud_for_compute)
    for par_index in range(par_cloud_length):
        par_pos = par_cloud_for_compute[par_index][obj_index].pos
        par_ori = par_cloud_for_compute[par_index][obj_index].ori

        dis_obseCur_parOld = compute_pos_err_bt_2_points(obj_obse_pos_new, par_pos)
        ang_obseCur_parOld = compute_ang_err_bt_2_points(obj_obse_ori_new, par_ori)
        par_dis_list.append(dis_obseCur_parOld)
        par_ang_list.append(ang_obseCur_parOld)

    minDis_obseCur_parOld = min(par_dis_list)
    minAng_obseCur_parOld = min(par_ang_list)
    return minDis_obseCur_parOld, minAng_obseCur_parOld

# compute the transformation matrix represent that the pose of object in the robot world
def compute_transformation_matrix(a_pos, a_ori, b_pos, b_ori):
    # ow_T_a_3_3 = transformations.quaternion_matrix(a_ori)
    # ow_T_a_4_4 = rotation_4_4_to_transformation_4_4(ow_T_a_3_3,a_pos)
    # ow_T_b_3_3 = transformations.quaternion_matrix(b_ori)
    # ow_T_b_4_4 = rotation_4_4_to_transformation_4_4(ow_T_b_3_3,b_pos)
    # a_T_ow_4_4 = np.linalg.inv(ow_T_a_4_4)
    # a_T_b_4_4 = np.dot(a_T_ow_4_4,ow_T_b_4_4)
    ow_T_a_3_3 = np.array(p.getMatrixFromQuaternion(a_ori)).reshape(3, 3)
    ow_T_a_3_4 = np.c_[ow_T_a_3_3, a_pos]  # Add position to create 3x4 matrix
    ow_T_a_4_4 = np.r_[ow_T_a_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix

    ow_T_b_3_3 = np.array(p.getMatrixFromQuaternion(b_ori)).reshape(3, 3)
    ow_T_b_3_4 = np.c_[ow_T_b_3_3, b_pos]  # Add position to create 3x4 matrix
    ow_T_b_4_4 = np.r_[ow_T_b_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix

    a_T_ow_4_4 = np.linalg.inv(ow_T_a_4_4)
    a_T_b_4_4 = np.dot(a_T_ow_4_4,ow_T_b_4_4)
    return a_T_b_4_4

# get pose of item
def get_item_pos(pybullet_env, item_id):
    item_info = pybullet_env.getBasePositionAndOrientation(item_id)
    return item_info[0],item_info[1]
# random values generated from a Gaussian distribution
def take_easy_gaussian_value(mean, sigma):
    normal = random.normalvariate(mean, sigma)
    return normal

# make sure all angles all between -pi and +pi
def angle_correction(angle):
    if math.pi <= angle <= (3.0 * math.pi):
        angle = angle - 2 * math.pi
    elif -(3.0 * math.pi) <= angle <= -math.pi:
        angle = angle + 2 * math.pi
    angle = abs(angle)
    return angle
# make sure all quaternions all between -pi and +pi
def quaternion_correction(quaternion): # x,y,z,w
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
    #if theta >= math.pi or theta <= -math.pi:
    #    new_quaternion = [-quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
    #    return new_quaternion
    #return quaternion # x,y,z,w
    return new_quaternion

def _get_position_from_matrix44(a_T_b_4_4):
    x = a_T_b_4_4[0][3]
    y = a_T_b_4_4[1][3]
    z = a_T_b_4_4[2][3]
    position = [x, y, z]
    return position

# get quaternion from matrix
def _get_quaternion_from_matrix(a_T_b_4_4):
    rot_matrix = a_T_b_4_4[:3, :3]
    rotation = R.from_matrix(rot_matrix)
    quaternion = rotation.as_quat()
    return quaternion

#def publish_ray_trace_info(particle_cloud_pub):
#    par_pose_list = list(range(PARTICLE_NUM))
#    for par_index in range(PARTICLE_NUM):
#        par_pose = particle_pose()
#        obj_pose_list = []
#        for obj_index in range(OBJECT_NUM):
#            obj_pose = object_pose()
#            obj_info = particle_cloud_pub[par_index][obj_index]
#            obj_pose.name = obj_info.par_name
#            obj_pose.pose.position.x = obj_info.pos[0]
#            obj_pose.pose.position.y = obj_info.pos[1]
#            obj_pose.pose.position.z = obj_info.pos[2]
#            obj_pose_list.append(obj_pose)
#        par_pose.objects = obj_pose_list
#        par_pose_list[par_index] = par_pose
#        
#    par_list.particles = par_pose_list
#    pub_ray_trace.publish(par_list)

def publish_par_pose_info(particle_cloud_pub):
    par_pose_list = list(range(PARTICLE_NUM))
    for par_index in range(PARTICLE_NUM):
        par_pose = particle_pose()
        obj_pose_list = []
        for obj_index in range(OBJECT_NUM):
            obj_pose = object_pose()
            obj_info = particle_cloud_pub[par_index][obj_index]
            obj_pose.name = obj_info.par_name
            obj_pose.pose.position.x = obj_info.pos[0]
            obj_pose.pose.position.y = obj_info.pos[1]
            obj_pose.pose.position.z = obj_info.pos[2]
            obj_pose.pose.orientation.x = obj_info.ori[0]
            obj_pose.pose.orientation.y = obj_info.ori[1]
            obj_pose.pose.orientation.z = obj_info.ori[2]
            obj_pose.pose.orientation.w = obj_info.ori[3]
            obj_pose_list.append(obj_pose)
        par_pose.objects = obj_pose_list
        par_pose_list[par_index] = par_pose
        
    par_list.particles = par_pose_list
    pub_par_pose.publish(par_list)
            
def publish_esti_pose_info(estimated_object_set):
    esti_pose_list = []
    for obj_index in range(OBJECT_NUM):
        esti_pose = object_pose()
        esti_obj_info = estimated_object_set[obj_index]
        esti_pose.name = esti_obj_info.obj_name
        esti_pose.pose.position.x = esti_obj_info.pos[0]
        esti_pose.pose.position.y = esti_obj_info.pos[1]
        esti_pose.pose.position.z = esti_obj_info.pos[2]
        esti_pose.pose.orientation.x = esti_obj_info.ori[0]
        esti_pose.pose.orientation.y = esti_obj_info.ori[1]
        esti_pose.pose.orientation.z = esti_obj_info.ori[2]
        esti_pose.pose.orientation.w = esti_obj_info.ori[3]
        esti_pose_list.append(esti_pose)
    esti_obj_list.objects = esti_pose_list 
    pub_esti_pose.publish(esti_obj_list)
    for obj_index in range(OBJECT_NUM):
        pose_PBPF = PoseStamped()
        pose_PBPF.pose.position.x = estimated_object_set[obj_index].pos[0]
        pose_PBPF.pose.position.y = estimated_object_set[obj_index].pos[1]
        pose_PBPF.pose.position.z = estimated_object_set[obj_index].pos[2]
        pose_PBPF.pose.orientation.x = estimated_object_set[obj_index].ori[0]
        pose_PBPF.pose.orientation.y = estimated_object_set[obj_index].ori[1]
        pose_PBPF.pose.orientation.z = estimated_object_set[obj_index].ori[2]
        pose_PBPF.pose.orientation.w = estimated_object_set[obj_index].ori[3]
        pub_PBPF_list[obj_index].publish(pose_PBPF)

# need to change
def process_esti_pose_from_rostopic(estimated_object_set):
    esti_pose_list = []
    for obj_index in range(OBJECT_NUM):
        esti_obj_info = copy.deepcopy(estimated_object_set[obj_index])
        esti_obj_pos_x = esti_obj_info.pos[0]
        esti_obj_pos_y = esti_obj_info.pos[1]
        esti_obj_pos_z = esti_obj_info.pos[2]
        esti_obj_ori_x = esti_obj_info.ori[0]
        esti_obj_ori_y = esti_obj_info.ori[1]
        esti_obj_ori_z = esti_obj_info.ori[2]
        esti_obj_ori_w = esti_obj_info.ori[3]
        esti_pose = [[esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z], [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]]
        esti_pose_list.append(esti_pose)
    return esti_pose_list


def generate_point_for_ray(pw_T_c_pos, pw_T_parC_4_4, obj_index):
    vector_list = [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
                   [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
                   [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
                   [1,0.5,0.5], [1,0.5,-0.5], [1,-0.5,0.5], [1,-0.5,-0.5],
                   [-1,0.5,0.5], [-1,0.5,-0.5], [-1,-0.5,0.5], [-1,-0.5,-0.5],
                   [0.5,1,0.5], [0.5,1,-0.5], [-0.5,1,0.5], [-0.5,1,-0.5],
                   [0.5,-1,0.5], [0.5,-1,-0.5], [-0.5,-1,0.5], [-0.5,-1,-0.5],
                   [0.5,0.5,1], [0.5,-0.5,1], [-0.5,0.5,1], [-0.5,-0.5,1],
                   [0.5,0.5,-1], [0.5,-0.5,-1], [-0.5,0.5,-1], [-0.5,-0.5,-1]]
    r = math.sqrt(2)
    if OBJECT_NAME_LIST[obj_index] == "soup":
        vector_list = [[0,0,1], [0,0,-1],
                       [r,0,1], [0,r,1], [-r,0,1], [0,-r,1], [r,r,1], [r,-r,1], [-r,r,1], [-r,-r,1],
                       [r,0,0.5], [0,r,0.5], [-r,0,0.5], [0,-r,0.5], [r,r,0.5], [r,-r,0.5], [-r,r,0.5], [-r,-r,0.5],
                       [r,0,0], [0,r,0], [-r,0,0], [0,-r,0], [r,r,0], [r,-r,0], [-r,r,0], [-r,-r,0],
                       [r,0,-0.5], [0,r,-0.5], [-r,0,-0.5], [0,-r,-0.5], [r,r,-0.5], [r,-r,-0.5], [-r,r,-0.5], [-r,-r,-0.5],
                       [r,0,-1], [0,r,-1], [-r,0,-1], [0,-r,-1], [r,r,-1], [r,-r,-1], [-r,r,-1], [-r,-r,-1]]
        # vector_list = [[0,0,1], [0,0,-1],
        #                [2,2,1], [2,-2,1], [-2,2,1], [-2,-2,1], [r,r,1], [r,-r,1], [-r,r,1], [-r,-r,1],
        #                [r,0,0.5], [0,r,0.5], [-r,0,0.5], [0,-r,0.5], [r,r,0.5], [r,-r,0.5], [-r,r,0.5], [-r,-r,0.5],
        #                [r,0,0], [0,r,0], [-r,0,0], [0,-r,0], [r,r,0], [r,-r,0], [-r,r,0], [-r,-r,0],
        #                [r,0,-0.5], [0,r,-0.5], [-r,0,-0.5], [0,-r,-0.5], [r,r,-0.5], [r,-r,-0.5], [-r,r,-0.5], [-r,-r,-0.5],
        #                [2,2,-1], [2,-2,-1], [-2,2,-1], [-2,-2,-1], [r,r,-1], [r,-r,-1], [-r,r,-1], [-r,-r,-1]]
    point_list = []
    point_pos_list = []
    for index in range(len(vector_list)):
        parC_T_p_x_new = vector_list[index][0] * x_w_list[obj_index]/2
        parC_T_p_y_new = vector_list[index][1] * y_l_list[obj_index]/2
        parC_T_p_z_new = vector_list[index][2] * z_h_list[obj_index]/2
        parC_T_p_pos = [parC_T_p_x_new, parC_T_p_y_new, parC_T_p_z_new]
        parC_T_p_ori = [0, 0, 0, 1] # x, y, z, w
        # parC_T_p_3_3 = transformations.quaternion_matrix(parC_T_p_ori)
        # parC_T_p_4_4 = rotation_4_4_to_transformation_4_4(parC_T_p_3_3, parC_T_p_pos)
        parC_T_p_3_3 = np.array(p.getMatrixFromQuaternion(parC_T_p_ori)).reshape(3, 3)
        parC_T_p_3_4 = np.c_[parC_T_p_3_3, parC_T_p_pos]  # Add position to create 3x4 matrix
        parC_T_p_4_4 = np.r_[parC_T_p_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
        pw_T_p_4_4 = np.dot(pw_T_parC_4_4, parC_T_p_4_4)
        pw_T_p_pos = [pw_T_p_4_4[0][3], pw_T_p_4_4[1][3], pw_T_p_4_4[2][3]]
        pw_T_p_ori = transformations.quaternion_from_matrix(pw_T_p_4_4)
        pw_T_p_pose = Center_T_Point_for_Ray(pw_T_p_pos, pw_T_p_ori, parC_T_p_4_4, index)
        point_list.append(pw_T_p_pose)
        point_pos_list.append(pw_T_p_pos)
    return point_list, point_pos_list

def track_fk_sim_world():
    # if SHOW_RAY == True:
    #     p_track_fk_env = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT,GUI_SERVER
    # else:
    #     p_track_fk_env = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT,GUI_SERVER
    p_track_fk_env = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT,GUI_SERVER
    p_track_fk_env.setAdditionalSearchPath(pybullet_data.getDataPath())
    track_fk_plane_id = p_track_fk_env.loadURDF("plane.urdf")
    p_track_fk_env.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0.3,0.1,0.2])
    
    if SIM_REAL_WORLD_FLAG == True:
        table_pos_1 = [0.46, -0.01, 0.710]
        table_ori_1 = p_track_fk_env.getQuaternionFromEuler([0,0,0])
        table_id_1 = p_track_fk_env.loadURDF(os.path.expanduser("~/project/object/others/table.urdf"), table_pos_1, table_ori_1, useFixedBase = 1)
    else:
        table_pos_1 = [0, 0, 0]
    track_fk_rob_id = p_track_fk_env.loadURDF(os.path.expanduser("~/project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                              [0, 0, 0.02+table_pos_1[2]],
                                              [0, 0, 0, 1],
                                              useFixedBase=1)

    if task_flag == "1":
        track_fk_obst_big_id = p_track_fk_env.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_obstacle_big.urdf"),
                                                   pw_T_obst_opti_pos_big,
                                                   pw_T_obst_opti_ori_big,
                                                   useFixedBase=1)
        # track_fk_obst_small_id = p_track_fk_env.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_obstacle_small.urdf"),
        #                                            pw_T_obst_opti_pos_small,
        #                                            pw_T_obst_opti_ori_small,
        #                                            useFixedBase=1)
    
    return p_track_fk_env, track_fk_rob_id, track_fk_plane_id

def track_fk_world_rob_mv(p_sim, sim_rob_id, position):
    num_joints = 9
    for joint_index in range(num_joints):
        if joint_index == 7 or joint_index == 8:
            p_sim.resetJointState(sim_rob_id,
                                  joint_index+2,
                                  targetValue=position[joint_index])
        else:
            p_sim.resetJointState(sim_rob_id,
                                  joint_index,
                                  targetValue=position[joint_index])

# get camera intrinsic params
def _get_camera_intrinsic_params(camera_info_topic_name):
    camera_info = None

    def camera_info_callback(data):
        nonlocal camera_info
        camera_info = data

    rospy.Subscriber(camera_info_topic_name, CameraInfo, camera_info_callback)

    while camera_info is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    fx = camera_info.K[0]  # Focal length in x-axis
    fy = camera_info.K[4]  # Focal length in y-axis
    cx = camera_info.K[2]  # Principal point in x-axis
    cy = camera_info.K[5]  # Principal point in y-axis

    image_height = camera_info.height
    image_width = camera_info.width

    Camera = namedtuple("Camera", "fx fy cx cy image_height image_width")
    camera_intrinsic_parameters = Camera(fx, fy, cx, cy, image_height, image_width)
    return camera_intrinsic_parameters 

def _vk_config_setting():
    depth_img_height = RESOLUTION_DEPTH[0] # 480
    depth_img_width = RESOLUTION_DEPTH[1] # 848
    ## Create context
    vk_config = vkdepth.ContextConfig()
    vk_config.render_size(depth_img_width, depth_img_height) # width(848), height(480)
    return vk_config

def _vk_camera_setting(pw_T_camD_tf_4_4, camD_T_camVk_4_4):
    ## Setup vk_camera
    vk_camera = vkdepth.Camera()

    pw_T_camD_tf_4_4_ = copy.deepcopy(pw_T_camD_tf_4_4)
    camD_T_camVk_4_4_ = copy.deepcopy(camD_T_camVk_4_4)
    pw_T_camVk_4_4_ = np.dot(pw_T_camD_tf_4_4_, camD_T_camVk_4_4_)
    
    pw_T_camVk_pos = _get_position_from_matrix44(pw_T_camVk_4_4_)
    x_pos = pw_T_camVk_pos[0]
    y_pos = pw_T_camVk_pos[1]
    z_pos = pw_T_camVk_pos[2]
    pw_T_camVk_ori = _get_quaternion_from_matrix(pw_T_camVk_4_4_) # x, y, z, w
    x_ori = pw_T_camVk_ori[0]
    y_ori = pw_T_camVk_ori[1]
    z_ori = pw_T_camVk_ori[2]
    w_ori = pw_T_camVk_ori[3]

    vk_camera.set_near_far(NEARVAL, FARVAL)
    vk_camera.set_aspect_wh(WIDTH_DEPTH, HEIGHT_DEPTH) # width: 848, height: 480
    vk_camera.set_position(x_pos, y_pos, z_pos) # x, y, z
    vk_camera.set_orientation_quat(w_ori, x_ori, y_ori, z_ori) # w, x, y, z

    return vk_camera, pw_T_camVk_4_4_

def _vk_load_meshes():
    global _vk_context
    vk_obj_id_list = [0] * OBJECT_NUM
    vk_rob_link_id_list = [0] * PANDA_ROBOT_LINK_NUMBER # 11
    vk_other_id_list = []
    # object
    # a, b
    for obj_index in range(OBJECT_NUM):
        obj_name = OBJECT_NAME_LIST[obj_index] # "cracker"/"soup"
        if obj_index == 0:
            obj_id = _vk_context.load_model("assets/meshes/cracker.vkdepthmesh")
        elif obj_index == 1:
            obj_id = _vk_context.load_model("assets/meshes/005_tomato_soup_can.vkdepthmesh")
        vk_obj_id_list[obj_index] = obj_id
    # robot
    # There are actually 13 links, of which "link8" and "panda_grasptarget" have no entities.
    ## "link0,1,2,3,4,5,6,7", "panda_hand", "panda_left_finger", "panda_right_finger"
    # index:0,1,2,3,4,5,6,7,   9,            10,                  11    
    for link_index in range(PANDA_ROBOT_LINK_NUMBER):
        if link_index < 8:
            rob_link_id = _vk_context.load_model("assets/meshes/link"+str(link_index)+".vkdepthmesh")
        elif link_index == 8:
            rob_link_id = _vk_context.load_model("assets/meshes/hand.vkdepthmesh")
        elif link_index == 9:
            rob_link_id = _vk_context.load_model("assets/meshes/left_finger.vkdepthmesh")
        elif link_index == 10:
            rob_link_id = _vk_context.load_model("assets/meshes/right_finger.vkdepthmesh")
        vk_rob_link_id_list[link_index] = rob_link_id
    # table
    other_obj_id = _vk_context.load_model("assets/meshes/table.vkdepthmesh")
    vk_other_id_list.append(other_obj_id)
    # board
    other_obj_id = _vk_context.load_model("assets/meshes/board.vkdepthmesh")
    vk_other_id_list.append(other_obj_id)
    # barrier 1,2,3
    other_obj_id = _vk_context.load_model("assets/meshes/barrier.vkdepthmesh")
    vk_other_id_list.append(other_obj_id)
    other_obj_id = _vk_context.load_model("assets/meshes/barrier.vkdepthmesh")
    vk_other_id_list.append(other_obj_id)
    other_obj_id = _vk_context.load_model("assets/meshes/barrier.vkdepthmesh")
    vk_other_id_list.append(other_obj_id)

    # obj_id = _vk_context.load_model()
    # vk_other_id_list.append(obj_id)

    return vk_obj_id_list, vk_rob_link_id_list, vk_other_id_list

# "particle setting"
def _vk_state_setting(vk_particle_cloud, pw_T_camVk_4_4, pybullet_env, par_robot_id):
    global _vk_context
    vk_state_list = [0] * PARTICLE_NUM
    pw_T_camVk_4_4_ = copy.deepcopy(pw_T_camVk_4_4)
    camVk_T_pw_4_4_ = np.linalg.inv(pw_T_camVk_4_4_)
    for index, particle in enumerate(vk_particle_cloud):
        vk_state = vkdepth.State()
        vk_state_list[index] = vk_state
        ## add object in particle
        for obj_index in range(OBJECT_NUM):
            vk_T_par_pos = copy.deepcopy(particle[obj_index].pos)
            x_pos = vk_T_par_pos[0]
            y_pos = vk_T_par_pos[1]
            z_pos = vk_T_par_pos[2]
            vk_T_par_ori = copy.deepcopy(particle[obj_index].ori)
            x_ori = vk_T_par_ori[0]
            y_ori = vk_T_par_ori[1]
            z_ori = vk_T_par_ori[2]
            w_ori = vk_T_par_ori[3]

            vk_state.add_instance(_vk_obj_id_list[obj_index],
                                  x_pos, y_pos, z_pos,
                                  w_ori, x_ori, y_ori, z_ori) # w, x, y, z
            # vk_state.add_instance(_vk_obj_id_list[obj_index],
            #                       0, 0, 0,
            #                       1, 0, 0, 0) # w, x, y, z

        # vk mark 
        ## add table/robot... in particle
        # There are actually 13 links, of which "link8" and "panda_grasptarget" have no entities.
        ## "link    0,1,2,3,4,5,6,7", "panda_hand", "panda_left_finger", "panda_right_finger"
        # index:    0,1,2,3,4,5,6,7,   9,            10,                  11
        # loop:     0,1,2,3,4,5,6,7,   8,            9,                   10
        # linkstate:x,0,1,2,3,4,5,6,   8,            9,                   10
        all_links_info = pybullet_env.getLinkStates(par_robot_id, range(PANDA_ROBOT_LINK_NUMBER + 2), computeForwardKinematics=True) # 11+2; range: [0,13)
        for rob_link_index in range(PANDA_ROBOT_LINK_NUMBER): # 11: [0, 11)
            if rob_link_index == 0:
                link_info = pybullet_env.getBasePositionAndOrientation(par_robot_id) # base (link0)
                vk_T_link_pos = link_info[0]
                vk_T_link_ori = link_info[1]
            elif rob_link_index < 8 and rob_link_index > 0:
                link_info = all_links_info[rob_link_index-1]
                vk_T_link_pos = link_info[4]
                vk_T_link_ori = link_info[5]
            else:
                link_info = all_links_info[rob_link_index]
                vk_T_link_pos = link_info[4]
                vk_T_link_ori = link_info[5]
            x_pos = vk_T_link_pos[0]
            y_pos = vk_T_link_pos[1]
            z_pos = vk_T_link_pos[2]
            x_ori = vk_T_link_ori[0]
            y_ori = vk_T_link_ori[1]
            z_ori = vk_T_link_ori[2]
            w_ori = vk_T_link_ori[3]
            if rob_link_index == 10:
                y_ori = -y_ori
                z_ori = -z_ori
            vk_state.add_instance(_vk_rob_link_id_list[rob_link_index],
                                    x_pos, y_pos, z_pos,
                                    w_ori, x_ori, y_ori, z_ori) # w, x, y, z
            # vk_state.add_instance(_vk_rob_link_id_list[rob_link_index],
            #                         0, 0, 0,
            #                         1, 0, 0, 0) # w, x, y, z
        # other objects
        other_obj_number = len(_vk_other_id_list)
        # table
        table_pos_1 = [0.46, -0.01, 0.710]
        table_ori_1 = p.getQuaternionFromEuler([0,0,0]) # x, y, z, w
        vk_state.add_instance(_vk_other_id_list[0],
                              table_pos_1[0], table_pos_1[1], table_pos_1[2],
                              table_ori_1[3], table_ori_1[0], table_ori_1[1], table_ori_1[2]) # w, x, y, z
        # board
        board_pos_1 = [0.274, 0.581, 0.87575]
        board_ori_1 = p.getQuaternionFromEuler([math.pi/2,math.pi/2,0]) # x, y, z, w
        vk_state.add_instance(_vk_other_id_list[1],
                              board_pos_1[0], board_pos_1[1], board_pos_1[2],
                              board_ori_1[3], board_ori_1[0], board_ori_1[1], board_ori_1[2]) # w, x, y, z
        # barrier 1,2,3
        barrier_pos_1 = [-0.694, 0.443, 0.895]
        barrier_ori_1 = p.getQuaternionFromEuler([0,math.pi/2,0]) # x, y, z, w
        vk_state.add_instance(_vk_other_id_list[2],
                              barrier_pos_1[0], barrier_pos_1[1], barrier_pos_1[2],
                              barrier_ori_1[3], barrier_ori_1[0], barrier_ori_1[1], barrier_ori_1[2]) # w, x, y, z
        barrier_pos_2 = [-0.694, -0.607, 0.895]
        barrier_ori_2 = p.getQuaternionFromEuler([0,math.pi/2,0]) # x, y, z, w
        vk_state.add_instance(_vk_other_id_list[3],
                              barrier_pos_2[0], barrier_pos_2[1], barrier_pos_2[2],
                              barrier_ori_2[3], barrier_ori_2[0], barrier_ori_2[1], barrier_ori_2[2]) # w, x, y, z
        barrier_pos_3 = [0.459, -0.972, 0.895]
        barrier_ori_3 = p.getQuaternionFromEuler([0,math.pi/2,math.pi/2]) # x, y, z, w
        vk_state.add_instance(_vk_other_id_list[4],
                              barrier_pos_3[0], barrier_pos_3[1], barrier_pos_3[2],
                              barrier_ori_3[3], barrier_ori_3[0], barrier_ori_3[1], barrier_ori_3[2]) # w, x, y, z


        _vk_context.add_state(vk_state)
        # vk_state: 
        # 70
        # vk_state->add_instance: 
        # object1, object2, ..., link0, link1, ..., link7, panda_hand, "panda_left_finger, panda_right_finger, table, barrier, ...
        
    return vk_state_list

# get vk rendered depth image
def _vk_depth_image_getting():
    global _vk_context
    vk_rendered_depth_image_array_list = []
    for par_index in range(PARTICLE_NUM):
        vk_rendered_depth_image_vkdepth = _vk_context.view(par_index) # <class 'vkdepth.DepthView'>
        vk_rendered_depth_image_array = np.array(vk_rendered_depth_image_vkdepth, copy = False) # <class 'numpy.ndarray'>
        vk_rendered_depth_image_array_list.append(vk_rendered_depth_image_array)
    return vk_rendered_depth_image_array_list

# update vk rendered depth image
def _vk_update_depth_image(vk_state_list, vk_particle_cloud, pybullet_env, par_robot_id):
    for index, particle in enumerate(vk_particle_cloud):
        objs_states = np.array(vk_state_list[index].view(), copy = False)
        for obj_index in range(OBJECT_NUM):
            objs_states[obj_index, 1] = particle[obj_index].pos[0] # x_pos
            objs_states[obj_index, 2] = particle[obj_index].pos[1] # y_pos
            objs_states[obj_index, 3] = particle[obj_index].pos[2] # z_pos
            objs_states[obj_index, 4] = particle[obj_index].ori[3] # w_ori
            objs_states[obj_index, 5] = particle[obj_index].ori[0] # x_ori
            objs_states[obj_index, 6] = particle[obj_index].ori[1] # y_ori
            objs_states[obj_index, 7] = particle[obj_index].ori[2] # z_ori
        all_links_info = pybullet_env.getLinkStates(par_robot_id, range(PANDA_ROBOT_LINK_NUMBER + 2), computeForwardKinematics=True) # 11+2; range: [0,13)
        for rob_link_index in range(PANDA_ROBOT_LINK_NUMBER):
            if rob_link_index == 0:
                link_info = pybullet_env.getBasePositionAndOrientation(par_robot_id) # base (link0)
                vk_T_link_pos = link_info[0]
                vk_T_link_ori = link_info[1]
            elif rob_link_index < 8 and rob_link_index > 0:
                link_info = all_links_info[rob_link_index-1]
                vk_T_link_pos = link_info[4]
                vk_T_link_ori = link_info[5]
            else:
                link_info = all_links_info[rob_link_index]
                vk_T_link_pos = link_info[4]
                vk_T_link_ori = link_info[5]
            y_ori = vk_T_link_ori[1]
            z_ori = vk_T_link_ori[2]
            if rob_link_index == 10:
                y_ori = -vk_T_link_ori[1]
                z_ori = -vk_T_link_ori[2]
            objs_states[OBJECT_NUM+rob_link_index, 1] = vk_T_link_pos[0] # x_pos
            objs_states[OBJECT_NUM+rob_link_index, 2] = vk_T_link_pos[1] # y_pos
            objs_states[OBJECT_NUM+rob_link_index, 3] = vk_T_link_pos[2] # z_pos
            objs_states[OBJECT_NUM+rob_link_index, 4] = vk_T_link_ori[3] # w_ori
            objs_states[OBJECT_NUM+rob_link_index, 5] = vk_T_link_ori[0] # x_ori
            objs_states[OBJECT_NUM+rob_link_index, 6] = y_ori # y_ori
            objs_states[OBJECT_NUM+rob_link_index, 7] = z_ori # z_ori
        
        # other_obj_number = len(_vk_other_id_list)
        # table_pos_1 = [0.46, -0.01, 0.710]
        # table_ori_1 = [0, 0, 0, 1] # x, y, z, w
        # for other_obj_index in range(other_obj_number):
        #     objs_states[OBJECT_NUM+PANDA_ROBOT_LINK_NUMBER+other_obj_index, 1] = table_pos_1[0] # x_pos
        #     objs_states[OBJECT_NUM+PANDA_ROBOT_LINK_NUMBER+other_obj_index, 2] = table_pos_1[1] # y_pos
        #     objs_states[OBJECT_NUM+PANDA_ROBOT_LINK_NUMBER+other_obj_index, 3] = table_pos_1[2] # z_pos
        #     objs_states[OBJECT_NUM+PANDA_ROBOT_LINK_NUMBER+other_obj_index, 4] = table_ori_1[3] # w_ori
        #     objs_states[OBJECT_NUM+PANDA_ROBOT_LINK_NUMBER+other_obj_index, 5] = table_ori_1[0] # x_ori
        #     objs_states[OBJECT_NUM+PANDA_ROBOT_LINK_NUMBER+other_obj_index, 6] = table_ori_1[1] # y_ori
        #     objs_states[OBJECT_NUM+PANDA_ROBOT_LINK_NUMBER+other_obj_index, 7] = table_ori_1[2] # z_ori

        


# ctrl-c write down the error file
def signal_handler(sig, frame):
    sys.exit()


reset_flag = True

while reset_flag == True:
    reset_flag = False
    if __name__ == '__main__':
        # CVPF Pose list (motion model)
        boss_obs_pose_CVPF = []
        boss_est_pose_CVPF = []
        rospy.init_node('PF_for_obse') # ros node
        signal.signal(signal.SIGINT, signal_handler) # interrupt judgment
        # publish
        pub_ray_trace = rospy.Publisher('/ray_trace_list', particle_list, queue_size = 10)
        ray_trace_list = particle_list()
        pub_par_pose = rospy.Publisher('/par_list', particle_list, queue_size = 10)
        par_list = particle_list()
        pub_esti_pose = rospy.Publisher('/esti_obj_list', estimated_obj_pose, queue_size = 10)
        esti_obj_list = estimated_obj_pose()
        pub_depth_image = rospy.Publisher("/camera/particle_depth_image_converted", Image, queue_size=5)
        particle_depth_image_converted = Image()
        
        # only for drawing box
        publish_DOPE_pose_flag = True


        if RUNNING_MODEL == "PBPF_RGB":
            USING_RGB_FLAG = True
            USING_D_FLAG = False
        elif RUNNING_MODEL == "PBPF_RGBD":
            USING_RGB_FLAG = True
            USING_D_FLAG = True
        else: # PBPF_D
            USING_RGB_FLAG = False
            USING_D_FLAG = True

        _particle_update_time = 0

        if task_flag == "4": 
            other_obj_num = 1 # parameter_info['other_obj_num']
        else:
            other_obj_num = 0 # parameter_info['other_obj_num']

        if run_alg_flag == 'CVPF':
            PARTICLE_NUM = 150
        
        # ============================================================================

        # get camera intrinsic info
        # CAMERA_INFO_TOPIC_COLOR: "/camera/color/camera_info"
        # CAMERA_INFO_TOPIC_DEPTH: "/camera/depth/camera_info"
        camera_intrinsic_parameters_color = _get_camera_intrinsic_params(CAMERA_INFO_TOPIC_COLOR)
        camera_intrinsic_parameters_depth = _get_camera_intrinsic_params(CAMERA_INFO_TOPIC_DEPTH)

        HEIGHT_DEPTH = camera_intrinsic_parameters_depth.image_height # 720/480
        WIDTH_DEPTH = camera_intrinsic_parameters_depth.image_width # 1280/848
        CX_DEPTH = camera_intrinsic_parameters_depth.cx
        CY_DEPTH = camera_intrinsic_parameters_depth.cy
        FX_DEPTH = camera_intrinsic_parameters_depth.fx
        FY_DEPTH = camera_intrinsic_parameters_depth.fy
        
        FOV_H_DEPTH = math.degrees(2 * math.atan(WIDTH_DEPTH / (2*FX_DEPTH))) # fov: horizontal / x
        FOV_V_DEPTH = math.degrees(2 * math.atan(HEIGHT_DEPTH / (2*FY_DEPTH))) # fov: vertical / y
        
        RESOLUTION_DEPTH = (HEIGHT_DEPTH, WIDTH_DEPTH) # 480 848

        # ============================================================================
        
        # print(obstacles_pos)
        # print(type(obstacles_pos))
        pub_DOPE_list = []
        pub_PBPF_list = []
        for obj_index in range(OBJECT_NUM):
            pub_DOPE = rospy.Publisher('DOPE_pose_'+OBJECT_NAME_LIST[obj_index], PoseStamped, queue_size = 1)
            pub_PBPF = rospy.Publisher('PBPF_pose_'+OBJECT_NAME_LIST[obj_index], PoseStamped, queue_size = 1)
            pub_DOPE_list.append(pub_DOPE)
            pub_PBPF_list.append(pub_PBPF)
        
        print("This is "+update_style_flag+" update in scene"+task_flag)    
        # some parameters
        d_thresh = 0.005
        a_thresh = 0.01
        d_thresh_obse = 0.15
        a_thresh_obse = math.pi * 2 / 3.0
        d_thresh_CV = 0.0002
        a_thresh_CV = 0.0010
        flag_record = 0
        flag_record_obse = 0
        flag_record_PBPF = 0
        flag_record_CVPF = 0
        flag_update_num_CV = 0
        flag_update_num_PB = 0
        
        if run_alg_flag == "PBPF" and VERSION == "old" and USING_D_FLAG == False:
            print("1: run_alg_flag: ",run_alg_flag,"; VERSION: ", VERSION, "; USING_D_FLAG: ", USING_D_FLAG)
            BOSS_PF_UPDATE_INTERVAL_IN_REAL = 0.05 # original value = 0.16
            PF_UPDATE_TIME_ONCE = 0.4 # rosbag slow down 0.125
        # elif run_alg_flag == "PBPF" and VERSION == "multiray" and USING_D_FLAG == False:
        elif RUNNING_MODEL == "PBPF_RGB":
            print("2: RUNNING_MODEL: ",RUNNING_MODEL)
            BOSS_PF_UPDATE_INTERVAL_IN_REAL = 0.20 # original value = 0.16
            PF_UPDATE_TIME_ONCE = 0.32 # 70 particles -> 2s
        elif RUNNING_MODEL == "PBPF_RGBD":
            print("3: RUNNING_MODEL: ",RUNNING_MODEL)
            BOSS_PF_UPDATE_INTERVAL_IN_REAL = 0.30 # original value = 0.16 
            PF_UPDATE_TIME_ONCE = 0.32 # 70 particles -> 35s
            PF_UPDATE_TIME_ONCE = 0.1 # 70 particles -> 35s
        else: # run_alg_flag == "CVPF":
            print("4: RUNNING_MODEL: ",RUNNING_MODEL)
            BOSS_PF_UPDATE_INTERVAL_IN_REAL = 0.30 # original value = 0.16
            PF_UPDATE_TIME_ONCE = 0.32 # rosbag slow down 0.02 0.3*(1/0.02)=15s
        PF_UPDATE_RATE = rospy.Rate(1.0/BOSS_PF_UPDATE_INTERVAL_IN_REAL)
        PF_UPDATE_RATE = rospy.Rate(1.0/PF_UPDATE_TIME_ONCE)
        print("PF_UPDATE_TIME_ONCE")
        print(PF_UPDATE_TIME_ONCE)
        # # error in xyz axis obse before recalibrating
        # boss_sigma_obs_x = 0.03973017808163751 / 2.0
        # boss_sigma_obs_y = 0.01167211468503462 / 2.0
        # boss_sigma_obs_z = 0.02820930183351492 / 2.0
        # # new obse error
        # boss_sigma_obs_x = 0.032860982 * 2.0
        # boss_sigma_obs_y = 0.012899399 * 1.5
        # boss_sigma_obs_z = 0.01
        # boss_sigma_obs_ang_init = 0.0216773873 * 2.0

        # Motion model Noise
        pos_noise = 0.01 # original value = 0.005
        ang_noise = 0.1 # original value = 0.05
        MOTION_NOISE = True
        
        
        # MOTION_NOISE = True

        # Standard deviation of computing the weight
        # boss_sigma_obs_ang = 0.216773873
        # boss_sigma_obs_ang = 0.0216773873
        
        for obj_index in range(OBJECT_NUM):
            object_name = OBJECT_NAME_LIST[obj_index]
            if object_name == "cracker":
                # boss_sigma_obs_ang = 0.0216773873 * 30
                # boss_sigma_obs_pos = 0.25 # 0.02 need to increase
                boss_sigma_obs_ang = 0.0216773873 * 30
                boss_sigma_obs_pos = 0.10 
                pos_noise = 0.001 * 5.0 # 5
                ang_noise = 0.05 * 3.0
                # mark
                # boss_sigma_obs_ang = 0.0
                # boss_sigma_obs_pos = 0.0
                pos_noise = 0.0
                ang_noise = 0.0
            else:
                boss_sigma_obs_ang = 0.0216773873 * 10
                # boss_sigma_obs_ang = 0.0216773873 * 20
                # boss_sigma_obs_ang = 0.0216773873 * 60
                # boss_sigma_obs_pos = 0.038226405
                # boss_sigma_obs_pos = 0.004
                boss_sigma_obs_pos = 0.10 # 0.02 need to increase
                # boss_sigma_obs_pos = 0.10 # 0.02 need to increase
                pos_noise = 0.001 * 5.0
                ang_noise = 0.05 * 3.0
                # mark
                pos_noise = 0.0
                ang_noise = 0.0

        # mark
        mass_mean = 0.380 # 0.380
        mass_sigma = 0.5
        friction_mean = 0.1
        friction_sigma = 0.3
        restitution_mean = 0.9
        restitution_sigma = 0.2
        all_frame = 0
        
        PBPF_time_cosuming_list = []
        
        pw_T_obst_opti_pos_small = [0.852134144216095, 0.14043691336334274, 0.10014295215002848]
        pw_T_obst_opti_ori_small = [0.00356749, -0.00269526, 0.28837681, 0.95750657]
        pw_T_obst_opti_pos_big = [0.7575524745560446, 0.3267505178967816, 0.14765408574692843]
        pw_T_obst_opti_ori_big = [0.70782892, 0.06771696, 0.0714355, 0.69949239]
        
        pw_T_obst_opti_pos_big = obstacles_pos[0]
        pw_T_obst_opti_ori_big = obstacles_ori[0]
        
        
        # multi-objects/robot list
        pw_T_rob_sim_pose_list_alg = []
        pw_T_obj_obse_obj_list_alg = []
        pw_T_obj_obse_oto_list_alg = []
        # need to change
        dis_std_list = [d_thresh_obse]
        ang_std_list = [a_thresh_obse]
        # build an object of class "Ros_Listener"
        ROS_LISTENER = Ros_Listener()
        create_scene = Create_Scene(OBJECT_NUM, ROBOT_NUM, other_obj_num)
        _tf_listener = tf.TransformListener()
        _launch_camera = LaunchCamera(WIDTH_DEPTH, HEIGHT_DEPTH, FOV_V_DEPTH)
        
        time.sleep(0.5)
        
        pw_T_rob_sim_pose_list_alg = create_scene.initialize_robot()
        print("Finish initializing robot")
        _pw_T_rob_sim_4_4 = pw_T_rob_sim_pose_list_alg[0].trans_matrix
        # get cameraDepth pose
        _pw_T_camD_tf_4_4 = _launch_camera.getCameraInPybulletWorldPose44(_tf_listener, _pw_T_rob_sim_4_4)
        print("========================")
        print("Camera depth len pose in Pybullet world:")
        print(_pw_T_camD_tf_4_4)
        print("========================")
        pw_T_obj_obse_obj_list_alg, trans_ob_list, rot_ob_list = create_scene.initialize_object()
        print("trans_ob_list, rot_ob_list:")
        print(trans_ob_list, rot_ob_list)
        print("========================")
        print("Finish initializing scene")
        for obj_index in range(other_obj_num):
            pw_T_obj_obse_oto_list_alg = create_scene.initialize_base_of_cheezit()

        initial_parameter = InitialSimulationModel(OBJECT_NUM, ROBOT_NUM, other_obj_num, PARTICLE_NUM, 
                                                pw_T_rob_sim_pose_list_alg, 
                                                pw_T_obj_obse_obj_list_alg,
                                                pw_T_obj_obse_oto_list_alg,
                                                update_style_flag, CHANGE_SIM_TIME)
        
        
        # get estimated object
        print("Begin initializing particles...")
        if run_alg_flag == "PBPF":
            estimated_object_set, particle_cloud_pub, p_par_env_list = initial_parameter.initial_and_set_simulation_env()
        if run_alg_flag == "CVPF":
            estimated_object_set, particle_cloud_pub, p_par_env_list = initial_parameter.initial_and_set_simulation_env_CV()
            boss_est_pose_CVPF.append(estimated_object_set) # [esti_obj1, esti_obj2]
        print("Finish initializing particles")

        # publish particles/estimated object
        publish_par_pose_info(particle_cloud_pub)
        publish_esti_pose_info(estimated_object_set)
        estimated_object_set_old = copy.deepcopy(estimated_object_set)
        estimated_object_set_old_list = process_esti_pose_from_rostopic(estimated_object_set_old)
        print("Before locating the pose of the camera")
        # if VERSION == "ray" or VERSION == "multiray":
        if OPTITRACK_FLAG == True and LOCATE_CAMERA_FLAG == "opti": # ar/opti
            realsense_tf = '/RealSense' # (use Optitrack)
        else:
            realsense_tf = '/ar_tracking_camera_frame' # (do not use Optitrack)
        while_loop_time = 0
        print(realsense_tf)
        while not rospy.is_shutdown():
            while_loop_time =  while_loop_time + 1
            # if while_loop_time > 50:
                # print("WARNING: ")
            if gazebo_flag == True:
                realsense_tf = '/realsense_camera'
            try:
                (trans_camera, rot_camera) = _tf_listener.lookupTransform('/panda_link0', realsense_tf, rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        print("I am here2")
        rob_T_cam_tf_pos = list(trans_camera)
        rob_T_cam_tf_ori = list(rot_camera)
        # rob_T_cam_tf_3_3 = transformations.quaternion_matrix(rob_T_cam_tf_ori)
        # rob_T_cam_tf_4_4 = rotation_4_4_to_transformation_4_4(rob_T_cam_tf_3_3, rob_T_cam_tf_pos)
        rob_T_cam_tf_3_3 = np.array(p.getMatrixFromQuaternion(rob_T_cam_tf_ori)).reshape(3, 3)
        rob_T_cam_tf_3_4 = np.c_[rob_T_cam_tf_3_3, rob_T_cam_tf_pos]  # Add position to create 3x4 matrix
        rob_T_cam_tf_4_4 = np.r_[rob_T_cam_tf_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
        pw_T_cam_tf = np.dot(_pw_T_rob_sim_4_4, rob_T_cam_tf_4_4)
        pw_T_cam_tf_pos = [pw_T_cam_tf[0][3], pw_T_cam_tf[1][3], pw_T_cam_tf[2][3]]
        print("==============================================")
        print("Camera RGB len pose in Robot world:")
        print(rob_T_cam_tf_4_4)
        print("Camera RGB len pose in Pybullet world:")
        print(pw_T_cam_tf)
        print("==============================================")        

        # run the simulation
        Flag = True
        # compute pose of robot arm
        
        # get pose of the end-effector of the robot arm from joints of robot arm 
        p_sim, sim_rob_id, sim_plane_id = track_fk_sim_world()
        track_fk_world_rob_mv(p_sim, sim_rob_id, ROS_LISTENER.current_joint_values)
        rob_link_9_pose_old = p_sim.getLinkState(sim_rob_id, 9) # position = rob_link_9_pose_old[0], quaternion = rob_link_9_pose_old[1]
        # rob_T_obj_obse_pos_old = list(trans_ob)
        # rob_T_obj_obse_ori_old = list(rot_ob)

        # ============================================================================
        # initialisation of vk configuration
        if VK_RENDER_FLAG == True:
            print("Begin initializing vulkon...")
            _camD_T_camVk_4_4 = np.array([[1, 0, 0, 0],
                                          [0,-1, 0, 0],
                                          [0, 0,-1, 0],
                                          [0, 0, 0, 1]])
            ## Setup vk_config
            _vk_config = _vk_config_setting()
            ## Setup vk_camera
            _vk_camera, _pw_T_camVk_4_4 = _vk_camera_setting(_pw_T_camD_tf_4_4, _camD_T_camVk_4_4)
            ## create context
            _vk_context = vkdepth.initialize(_vk_config)
            _vk_context.update_camera(_vk_camera)
            ## Load meshes
            _vk_obj_id_list, _vk_rob_link_id_list, _vk_other_id_list = _vk_load_meshes()
            ## Create states
            ## state -> particle
            ## instance -> object
            ## if we have many particles we can create many "q = vkdepth.State()"
            _vk_particle_cloud = copy.deepcopy(particle_cloud_pub)
            _vk_state_list = _vk_state_setting(_vk_particle_cloud, _pw_T_camVk_4_4, p_sim, sim_rob_id)
            ## Render and Download
            _vk_context.enqueue_render_and_download()
            ## Waiting for rendering and download
            _vk_context.wait()
            ## Get Depth image
            vk_rendered_depth_image_array_list = _vk_depth_image_getting()
            # fig, axs = plt.subplots(1, PARTICLE_NUM)
            # for par_index in range(PARTICLE_NUM):
            #     axs[par_index].imshow(vk_rendered_depth_image_array_list[par_index])
            # plt.show()
                    
        # ============================================================================


        print("Welcome to Our Approach ! RUNNING MODEL: ", RUNNING_MODEL)
        PBPF_alg = PBPFMove(OBJECT_NUM) # PF_alg
        CVPF_alg = CVPFMove(OBJECT_NUM) 
        # while True:
        #     print(ROS_LISTENER.detection_flag)
        t_begin = time.time()

        latest_obse_time_list = [0] * OBJECT_NUM
        old_obse_time_list = [0] * OBJECT_NUM
        check_dope_work_flag_init_list = [0] * OBJECT_NUM
        
        outlier_dis_list = [0] * OBJECT_NUM
        outlier_ang_list = [0] * OBJECT_NUM

        while not rospy.is_shutdown():
            
            if reset_flag == False:
                continue_to_run = True
            elif reset_flag == True:
                break
            

            dope_detection_flag_list = [0] * OBJECT_NUM
            global_objects_visual_by_DOPE_list = [0] * OBJECT_NUM
            global_objects_outlier_by_DOPE_list = [0] * OBJECT_NUM
            visible_threshold_dope_is_fresh_list = [0] * OBJECT_NUM
            visible_threshold_dope_X_list = [0] * OBJECT_NUM 
            visible_threshold_dope_X_small_list = [0] * OBJECT_NUM
            visible_threshold_outlier_XS_list = [0] * OBJECT_NUM 
            visible_threshold_outlier_S_list = [0] * OBJECT_NUM 
            visible_threshold_outlier_L_list = [0] * OBJECT_NUM
            visible_threshold_outlier_XL_list = [0] * OBJECT_NUM
            visible_weight_dope_X_smaller_than_threshold_list = [0] * OBJECT_NUM
            visible_weight_dope_X_larger_than_threshold_list = [0] * OBJECT_NUM
            visible_weight_outlier_larger_than_threshold_list = [0] * OBJECT_NUM
            visible_weight_outlier_smaller_than_threshold_list = [0] * OBJECT_NUM
            x_w_list = [0] * OBJECT_NUM
            y_l_list = [0] * OBJECT_NUM
            z_h_list = [0] * OBJECT_NUM

            

            #panda robot moves in the visualization window
            temp_pw_T_obj_obse_objs_list = []
            track_fk_world_rob_mv(p_sim, sim_rob_id, ROS_LISTENER.current_joint_values)
            for obj_index in range(OBJECT_NUM):
                

                # need to change
                object_name = OBJECT_NAME_LIST[obj_index]

                if object_name == "cracker":
                    x_w_list[obj_index] = 0.159
                    y_l_list[obj_index] = 0.21243700408935547
                    z_h_list[obj_index] = 0.06
                    visible_threshold_dope_X_list[obj_index] = 0.45 # 0.95
                    visible_threshold_dope_X_small_list[obj_index] = 0
                    # visible_threshold_outlier_XS_list[obj_index] = 0.45
                    visible_threshold_outlier_S_list[obj_index] = 0.45
                    visible_threshold_outlier_L_list[obj_index] = 0.6
                    # visible_threshold_outlier_XL_list[obj_index] = 0.6
                    visible_threshold_dope_is_fresh_list[obj_index] = 0.5
                    visible_weight_dope_X_smaller_than_threshold_list[obj_index] = 0.75
                    visible_weight_dope_X_larger_than_threshold_list[obj_index] = 0.05 # 0.25
                    visible_weight_outlier_larger_than_threshold_list[obj_index] = 0.25
                    visible_weight_outlier_smaller_than_threshold_list[obj_index] = 0.45

                    outlier_dis_list[obj_index] = 0.1
                    outlier_ang_list[obj_index] = math.pi * 1 / 4.0

                elif object_name == "soup":
                    x_w_list[obj_index] = 0.032829689025878906
                    y_l_list[obj_index] = 0.032829689025878906
                    z_h_list[obj_index] = 0.099
                    visible_threshold_dope_X_list[obj_index] = 0.3 # 0.95
                    visible_threshold_dope_X_small_list[obj_index] = 0
                    # visible_threshold_outlier_XS_list[obj_index] = 0.3
                    visible_threshold_outlier_S_list[obj_index] = 0.4
                    visible_threshold_outlier_L_list[obj_index] = 0.65
                    # visible_threshold_outlier_XL_list[obj_index] = 0.75
                    visible_threshold_dope_is_fresh_list[obj_index] = 0.6
                    visible_weight_dope_X_smaller_than_threshold_list[obj_index] = 0.75 # 0.6
                    visible_weight_dope_X_larger_than_threshold_list[obj_index] = 0.25 # 0.55
                    visible_weight_outlier_larger_than_threshold_list[obj_index] = 0.25
                    visible_weight_outlier_smaller_than_threshold_list[obj_index] = 0.55

                    outlier_dis_list[obj_index] = 0.07
                    outlier_ang_list[obj_index] = math.pi * 1 / 2.0

                # mark
                # elif object_name == "gelatin":
                else:
                    x_w_list[obj_index] = 0.159
                    y_l_list[obj_index] = 0.21243700408935547
                    z_h_list[obj_index] = 0.06
                    visible_threshold_dope_X_list[obj_index] = 0.95
                    visible_threshold_dope_X_small_list[obj_index] = 0
                    visible_threshold_outlier_S_list[obj_index] = 0.4
                    visible_threshold_outlier_L_list[obj_index] = 0.5
                    visible_threshold_dope_is_fresh_list[obj_index] = 0.5
                    visible_weight_dope_X_smaller_than_threshold_list[obj_index] = 0.75
                    visible_weight_dope_X_larger_than_threshold_list[obj_index] = 0.25
                    visible_weight_outlier_larger_than_threshold_list[obj_index] = 0.25
                    visible_weight_outlier_smaller_than_threshold_list[obj_index] = 0.45

                use_gazebo = ""
                if gazebo_flag == True:
                    use_gazebo = '_noise'

                try:
                    latest_obse_time = _tf_listener.getLatestCommonTime('/panda_link0', '/'+object_name+use_gazebo)
                    latest_obse_time_list[obj_index] = latest_obse_time
                    # print("rospy.get_time():")
                    # print(rospy.get_time())
                    # print("latest_obse_time.to_sec():")
                    # print(latest_obse_time.to_sec())
                    # old_obse_time = latest_obse_time.to_sec()
                    # if (rospy.get_time() - latest_obse_time.to_sec()) < 0.1:
                    #     (trans_ob,rot_ob) = _tf_listener.lookupTransform('/panda_link0', '/'+object_name+use_gazebo, rospy.Time(0))
                    #     print("obse is FRESH")

                    if check_dope_work_flag_init_list[obj_index] == 0:
                        check_dope_work_flag_init_list[obj_index] = 1
                        old_obse_time_list[obj_index] = latest_obse_time_list[obj_index].to_sec()
                    # print("latest_obse_time.to_sec():")
                    # print(latest_obse_time.to_sec())
                    # print("difference:", latest_obse_time.to_sec() - old_obse_time)
                    
                    if (latest_obse_time_list[obj_index].to_sec() > old_obse_time_list[obj_index]):
                        (trans_ob,rot_ob) = _tf_listener.lookupTransform('/panda_link0', '/'+object_name+use_gazebo, rospy.Time(0))
                        global_objects_visual_by_DOPE_list[obj_index] = 0
                        t_after = time.time()
                        trans_ob_list[obj_index] = trans_ob
                        rot_ob_list[obj_index] = rot_ob
                        # print(t_after - t_begin - 14)
                        # print("obse is FRESH:", obj_index)
                    else:
                        # obse has not been updating for a while
                        global_objects_visual_by_DOPE_list[obj_index] = 1
                        global_objects_outlier_by_DOPE_list[obj_index] = 1
                        # print("obse is NOT fresh:", obj_index)
                    old_obse_time_list[obj_index] = latest_obse_time_list[obj_index].to_sec()
                    # break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    
                    print(obj_index)
                    print("from main")
                    print("can not find tf")
                    
                rob_T_obj_obse_pos = list(trans_ob_list[obj_index])
                rob_T_obj_obse_ori = list(rot_ob_list[obj_index])
                # rob_T_obj_obse_3_3 = transformations.quaternion_matrix(rob_T_obj_obse_ori)
                # rob_T_obj_obse_4_4 = rotation_4_4_to_transformation_4_4(rob_T_obj_obse_3_3,rob_T_obj_obse_pos)
                rob_T_obj_obse_3_3 = np.array(p.getMatrixFromQuaternion(rob_T_obj_obse_ori)).reshape(3, 3)
                rob_T_obj_obse_3_4 = np.c_[rob_T_obj_obse_3_3, rob_T_obj_obse_pos]  # Add position to create 3x4 matrix
                rob_T_obj_obse_4_4 = np.r_[rob_T_obj_obse_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
                # mark
                bias_obse_x = -0.0
                bias_obse_y = 0
                bias_obse_z = 0.0
                pw_T_obj_obse = np.dot(_pw_T_rob_sim_4_4, rob_T_obj_obse_4_4)
                pw_T_obj_obse_pos = [pw_T_obj_obse[0][3]+bias_obse_x, pw_T_obj_obse[1][3]+bias_obse_y, pw_T_obj_obse[2][3]+bias_obse_z]
                pw_T_obj_obse_ori = transformations.quaternion_from_matrix(pw_T_obj_obse)

                # mark
                # if OPTITRACK_FLAG == False:
                #     # opti
                #     pw_T_obj_obse_pos = [0.42126008888811123, 0.170478291576308, 0.7841043693538842]
                #     pw_T_obj_obse_ori = [ 0.71232545,  0.01881896,  0.70146424, -0.01364641]
                #     # ar
                #     pw_T_obj_obse_pos = [0.4111173962619566, 0.2417332651514922, 0.7841262399213144]
                #     pw_T_obj_obse_ori = [7.13209378e-01, 4.71191996e-03, 7.00935199e-01, 1.64560070e-04]
                
                # need to change when we run alg in multi-object tracking scene
                # in the futrue we need to use "for obj_index in range(OBJECT_NUM):"
                pw_T_obj_obse_pos_new = copy.deepcopy(pw_T_obj_obse_pos)
                pw_T_obj_obse_ori_new = copy.deepcopy(pw_T_obj_obse_ori)
                pw_T_esti_obj_pose_old = copy.deepcopy(estimated_object_set_old_list[obj_index])
                pw_T_esti_obj_pos_old = copy.deepcopy(pw_T_esti_obj_pose_old[0])
                pw_T_esti_obj_ori_old = copy.deepcopy(pw_T_esti_obj_pose_old[1])

                dis_obseCur_estiOld = compute_pos_err_bt_2_points(pw_T_obj_obse_pos_new, pw_T_esti_obj_pos_old)
                ang_obseCur_estiOld = compute_ang_err_bt_2_points(pw_T_obj_obse_ori_new, pw_T_esti_obj_ori_old)
                pw_T_obj_obse_pose_new = [pw_T_obj_obse_pos_new, pw_T_obj_obse_ori_new]

                minDis_obseCur_parOld, minAng_obseCur_parOld = compute_diff_bt_two_pose(obj_index, particle_cloud_pub, pw_T_obj_obse_pose_new)            
                
                all_frame = all_frame + 1
                if run_alg_flag == "PBPF":
                    if minDis_obseCur_parOld > outlier_dis_list[obj_index] or minAng_obseCur_parOld > outlier_ang_list[obj_index]:
                        # print("DOPE becomes crazy")
                        global_objects_outlier_by_DOPE_list[obj_index] = 1
                # if :
                #     if minDis_obseCur_parOld > 0.10 or minAng_obseCur_parOld > math.pi * 1 / 2.0:
                #         # print("DOPE becomes crazy")
                # if dis_obseCur_estiOld > dis_std_list[obj_index]*3 or ang_obseCur_estiOld > ang_std_list[obj_index]*3:
                # if dis_obseCur_estiOld > 0.30:# or ang_obseCur_estiOld > math.pi * 1 / 2.0:
                #     # print("DOPE becomes crazy")

                # only for drawing BOX/ need to change
                if publish_DOPE_pose_flag == True:
                    
                    pose_DOPE = PoseStamped()
                    pose_DOPE.pose.position.x = pw_T_obj_obse_pos[0]
                    pose_DOPE.pose.position.y = pw_T_obj_obse_pos[1]
                    pose_DOPE.pose.position.z = pw_T_obj_obse_pos[2]
                    pose_DOPE.pose.orientation.x = pw_T_obj_obse_ori[0]
                    pose_DOPE.pose.orientation.y = pw_T_obj_obse_ori[1]
                    pose_DOPE.pose.orientation.z = pw_T_obj_obse_ori[2]
                    pose_DOPE.pose.orientation.w = pw_T_obj_obse_ori[3]
                    pub_DOPE_list[obj_index].publish(pose_DOPE)

                    
                    # print('DOPE_pose_'+OBJECT_NAME_LIST[obj_index])
                    # print(pw_T_obj_obse_pos[0], pw_T_obj_obse_pos[1], pw_T_obj_obse_pos[2])
                    
                pw_T_obj_obse_name = object_name
                pw_T_obj_obse_id = 0
                obse_object = Object_Pose(pw_T_obj_obse_name, pw_T_obj_obse_id, pw_T_obj_obse_pos, pw_T_obj_obse_ori, index=obj_index)
                temp_pw_T_obj_obse_objs_list.append(obse_object)
                
            pw_T_obj_obse_objects_list = copy.deepcopy(temp_pw_T_obj_obse_objs_list)
            
            # compute distance between old robot and cur robot (position and angle)
            rob_link_9_pose_cur = p_sim.getLinkState(sim_rob_id, 9)
            rob_link_9_ang_cur = p_sim.getEulerFromQuaternion(rob_link_9_pose_cur[1])
            
            dis_robcur_robold = compute_pos_err_bt_2_points(rob_link_9_pose_cur[0], rob_link_9_pose_old[0])
            
            # only for drawing box
            # obse_obj_pos_draw = copy.deepcopy(pw_T_obj_obse_objects_list[0].pos)
            # obse_obj_ori_draw = copy.deepcopy(pw_T_obj_obse_objects_list[0].ori) # pybullet x,y,z,w
            # pose_DOPE = PoseStamped()
            # pose_DOPE.pose.position.x = obse_obj_pos_draw[0]
            # pose_DOPE.pose.position.y = obse_obj_pos_draw[1]
            # pose_DOPE.pose.position.z = obse_obj_pos_draw[2]
            # pose_DOPE.pose.orientation.x = obse_obj_ori_draw[0]
            # pose_DOPE.pose.orientation.y = obse_obj_ori_draw[1]
            # pose_DOPE.pose.orientation.z = obse_obj_ori_draw[2]
            # pose_DOPE.pose.orientation.w = obse_obj_ori_draw[3]
            # # print(pose_DOPE)
            # pub_DOPE.publish(pose_DOPE)

            # update according to the pose
            if update_style_flag == "pose":
                # PBPF algorithm
                if run_alg_flag == "PBPF":
                    if (dis_robcur_robold > d_thresh):
                        # judgement for any particles contact
                        if PBPF_alg.isAnyParticleInContact():
                            simRobot_touch_par_flag = 1
                            _particle_update_time = _particle_update_time + 1
                            print("_particle_update_time:")
                            print(_particle_update_time)
                            t_begin_PBPF = time.time()
                            flag_update_num_PB = flag_update_num_PB + 1
                            pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                            # execute PBPF algorithm movement
                            estimated_object_set, dis_std_list, ang_std_list, particle_cloud_pub = PBPF_alg.update_particle_filter_PB(ROS_LISTENER.current_joint_values, # joints of robot arm
                                                                                    pw_T_obj_obse_objects_pose_list) # flag for judging obse work
                            rob_link_9_pose_old = copy.deepcopy(rob_link_9_pose_cur)

                            # print("Average time of updating: ",np.mean(PBPF_alg.times))
                            t_finish_PBPF = time.time()
                            PBPF_time_cosuming_list.append(t_finish_PBPF - t_begin_PBPF)
                            # print("Time consuming:", t_finish_PBPF - t_begin_PBPF)
                            # print("Max value:", PBPF_time_cosuming_list.max())
                            simRobot_touch_par_flag = 0
                        else:
                            # also update the pose of the robot arm in the simulation when particles are not touched
                            PBPF_alg.motion_model(initial_parameter.pybullet_particle_env_collection,
                                                                initial_parameter.fake_robot_id_collection,
                                                                ROS_LISTENER.current_joint_values)
    #                else:
    #                    PBPF_alg.motion_model(initial_parameter.pybullet_particle_env_collection,
    #                                                             initial_parameter.fake_robot_id_collection,
    #                                                             ROS_LISTENER.current_joint_values)
                # CVPF algorithm
                if run_alg_flag == "CVPF":
                    # if (dis_betw_cur_and_old_CV > d_thresh_CV) or (ang_betw_cur_and_old_CV > a_thresh_CV) or (dis_robcur_robold_CV > d_thresh_CV):
                    if (dis_robcur_robold > d_thresh_CV):
                        if CVPF_alg.isAnyParticleInContact():
                            flag_update_num_CV = flag_update_num_CV + 1
                            boss_obs_pose_CVPF.append(pw_T_obj_obse_objects_list)
                            # execute CVPF algorithm movement
                            pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                            estimated_object_set, dis_std_list, ang_std_list, particle_cloud_pub = CVPF_alg.update_particle_filter_CV(pw_T_obj_obse_objects_pose_list) # flag for judging obse work
                            rob_link_9_pose_old = copy.deepcopy(rob_link_9_pose_cur)
                        else:
                            CVPF_alg.robot_arm_move_CV(ROS_LISTENER.current_joint_values) # joints of robot arm
    #                else:
    #                    CVPF_alg.robot_arm_move_CV(ROS_LISTENER.current_joint_values) # joints of robot arm
                        
            # update according to the time
            elif update_style_flag == "time":
                while not rospy.is_shutdown():
                    t_begin_sleep = time.time()
                    # PBPF algorithm
                    Only_update_robot_flag = False
                    if run_alg_flag == "PBPF":
                        # mark
                        # if PBPF_alg.isAnyParticleInContact() and (dis_robcur_robold > 0.002):
                        if True:
                            print("Run ", RUNNING_MODEL)
                            simRobot_touch_par_flag = 1
                            _particle_update_time = _particle_update_time + 1
                            t_begin_PBPF = time.time()
                            flag_update_num_PB = flag_update_num_PB + 1
                            pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                            # execute PBPF algorithm movement
                            estimated_object_set, dis_std_list, ang_std_list, particle_cloud_pub = PBPF_alg.update_particle_filter_PB(ROS_LISTENER.current_joint_values, # joints of robot arm
                                                                                    pw_T_obj_obse_objects_pose_list)
                            rob_link_9_pose_old = copy.deepcopy(rob_link_9_pose_cur)
                            t_finish_PBPF = time.time()
                            PBPF_time_cosuming_list.append(t_finish_PBPF - t_begin_PBPF)
                            # mark
                            print("Time consuming:", t_finish_PBPF - t_begin_PBPF)
                            print("Max value:", max(PBPF_time_cosuming_list))
                            simRobot_touch_par_flag = 0
                        else:
                            Only_update_robot_flag = True
                            print("Just Update Robot")
                            PBPF_alg.motion_model(initial_parameter.pybullet_particle_env_collection,
                                                                initial_parameter.fake_robot_id_collection,
                                                                ROS_LISTENER.current_joint_values)
                    # CVPF algorithm
                    elif run_alg_flag == "CVPF":
                        # if CVPF_alg.isAnyParticleInContact():
                        # if dis_robcur_robold > 0.002:
                        flag_update_num_CV = flag_update_num_CV + 1
                        boss_obs_pose_CVPF.append(pw_T_obj_obse_objects_list)
                        # execute CVPF algorithm movement
                        pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                        estimated_object_set, dis_std_list, ang_std_list, particle_cloud_pub = CVPF_alg.update_particle_filter_CV(pw_T_obj_obse_objects_pose_list) # flag for judging obse work
                        rob_link_9_pose_old = copy.deepcopy(rob_link_9_pose_cur)
                        # else:
                        #     CVPF_alg.robot_arm_move_CV(ROS_LISTENER.current_joint_values) # joints of robot arm
                            
                    estimated_object_set_old = copy.deepcopy(estimated_object_set)
                    estimated_object_set_old_list = process_esti_pose_from_rostopic(estimated_object_set_old)
                    
                    if Only_update_robot_flag == False:
                        print("Waiting for next loop")
                        PF_UPDATE_RATE.sleep()
                        t_finish_sleep = time.time()
                        print("sleep time:", t_finish_sleep - t_begin_sleep)
                        print("========================================")
                    break    
            t_end_while = time.time()
            if Flag is False:
                break
        
        
        p_sim.disconnect()
        par_length = len(p_par_env_list)
        for i in range(par_length):
            p_par_env_list[i].disconnect()



