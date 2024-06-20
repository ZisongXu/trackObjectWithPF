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
        self.boss_sigma_obs_pos_init = 0.08 # original value: 16cm/10CM 
        self.boss_sigma_obs_x = self.boss_sigma_obs_pos_init / math.sqrt(2)
        self.boss_sigma_obs_y = self.boss_sigma_obs_pos_init / math.sqrt(2)
        self.boss_sigma_obs_z = 0.02
        self.boss_sigma_obs_ang_init = 0.0216773873 * 10 # original value: 0.0216773873 * 20
        
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
        
    def run(self):
        self.init_pybullet()
        while True:
            with self.lock:
                if not self.queue.empty():
                    method, *args = self.queue.get()
                    result = method(self, *args)
                    for key, value in result:
                        self.result[key] = value

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
        if self.SIM_REAL_WORLD_FLAG == True:
            table_pos_1 = [0.46, -0.01, 0.710]
            table_ori_1 = self.p_env.getQuaternionFromEuler([0,0,0])
            table_id_1 = self.p_env.loadURDF(os.path.expanduser("~/project/object/others/table.urdf"), table_pos_1, table_ori_1)

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

            board_pos_1 = [0.274, 0.581, 0.87575]
            board_ori_1 = self.p_env.getQuaternionFromEuler([math.pi/2,math.pi/2,0])
            board_id_1 = self.p_env.loadURDF(os.path.expanduser("~/project/object/others/board.urdf"), board_pos_1, board_ori_1, useFixedBase = 1)

            self.collision_detection_obj_id_collection.append(board_id_1)

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
                print("Generate particles for the target object:", obj_obse_name)
            particle_pos, particle_ori = self.generate_random_pose(obj_obse_pos, obj_obse_ori)
            gazebo_contain = ""
            if self.gazebo_flag == True:
                gazebo_contain = "gazebo_"
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
            
    def get_objects_pose(self):
        return_results = []
        for obj_index in range(self.object_num):
            obj_id = self.particle_objects_id_collection[obj_index]
            obj_info = self.p_env.getBasePositionAndOrientation(obj_id)
            obj_name = self.OBJECT_NAME_LIST[obj_index]
            obj_tuple = (obj_name, obj_info)
            return_results.append(obj_tuple)
        return_results.append(("one_particle", self.objects_list))
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
    
    def init_set_sim_robot_JointPosition(self, position):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                self.p_env.resetJointState(self.robot_id,
                                           joint_index+2,
                                           targetValue=position[joint_index])
            else:
                self.p_env.resetJointState(self.robot_id,
                                           joint_index,
                                           targetValue=position[joint_index])

    def move_robot_JointPosition(self, position):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                self.p_env.setJointMotorControl2(self.robot_id, joint_index+2,
                                                 pybullet_env.POSITION_CONTROL,
                                                 targetPosition=position[joint_index])
            else:
                self.p_env.setJointMotorControl2(self.robot_id, joint_index,
                                                 pybullet_env.POSITION_CONTROL,
                                                 argetPosition=position[joint_index])
        for time_index in range(int(self.pf_update_interval_in_sim)):
            self.p_env.stepSimulation()

    def add_noise_to_init_par(self, current_pos, sigma_init):
        mean = current_pos
        sigma = sigma_init
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def take_easy_gaussian_value(self, mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal
    
    # make sure all quaternions all between -pi and +pi
    def quaternion_correction(self, quaternion): # x,y,z,w
        new_quat = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]) # w,x,y,z
        cos_theta_over_2 = new_quat.w
        sin_theta_over_2 = math.sqrt(new_quat.x ** 2 + new_quat.y ** 2 + new_quat.z ** 2)
        theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
        theta = theta_over_2 * 2
        if theta >= math.pi or theta <= -math.pi:
            new_quaternion = [-quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
            return new_quaternion
        return quaternion
