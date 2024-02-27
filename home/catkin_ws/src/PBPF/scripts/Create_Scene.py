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
from Ros_Listener import Ros_Listener
from Object_Pose import Object_Pose
from Robot_Pose import Robot_Pose
import yaml
#Class of initialize the real world model
class Create_Scene():
    def __init__(self, target_obj_num=0, rob_num=0, other_obj_num=0):
        self.target_obj_num = target_obj_num
        self.rob_num = rob_num
        self.other_obj_num = other_obj_num

        self.table_pos_1 = [0.46, -0.01, 0.710]

        self.pw_T_target_obj_opti_pose_lsit = []
        self.pw_T_target_obj_obse_pose_lsit = []
        self.trans_ob_list = []
        self.rot_ob_list = []
        self.trans_gt_list = []
        self.rot_gt_list = []
        self.pw_T_rob_sim_pose_list = []
        self.pw_T_other_obj_opti_pose_list = []
        self.ros_listener = Ros_Listener()
        self.listener = tf.TransformListener()
        
        with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
            self.parameter_info = yaml.safe_load(file)
        self.gazebo_flag = self.parameter_info['gazebo_flag']
        self.object_name_list = self.parameter_info['object_name_list']
        self.oto_name_list = self.parameter_info['oto_name_list']
        self.dope_flag = self.parameter_info['dope_flag']
        self.SIM_REAL_WORLD_FLAG = self.parameter_info['sim_real_world_flag']
        self.optitrack_prom = True
        self.optitrack_flag = self.parameter_info['optitrack_flag']
        
    def initialize_object(self):
#        if self.gazebo_flag == True:
#            time.sleep(0.5)
#            for obj_index in range(self.target_obj_num):
#                _, model_pose_added_noise = self.ros_listener.listen_2_object_pose(self.object_name_list[obj_index])
#                panda_pose = self.ros_listener.listen_2_robot_pose()
#                print(model_pose_added_noise)
#                gazebo_T_obj_pos_obse = model_pose_added_noise[0]
#                gazebo_T_obj_ori_obse = model_pose_added_noise[1]
#                gazebo_T_rob_pos = panda_pose[0]
#                gazebo_T_rob_ori = panda_pose[1]
#                
#                pw_T_rob_sim_pos = self.pw_T_rob_sim_pose_list[0].pos
#                pw_T_rob_sim_ori = self.pw_T_rob_sim_pose_list[0].ori
#                pw_T_rob_sim_3_3 = transformations.quaternion_matrix(pw_T_rob_sim_ori)
#                pw_T_rob_sim_4_4 = self.rotation_4_4_to_transformation_4_4(pw_T_rob_sim_3_3, pw_T_rob_sim_pos)
#                self.pw_T_rob_sim_pose_list[0].trans_matrix = pw_T_rob_sim_4_4
#                
##                robpw_T_robga_4_4 = [[1., 0., 0.,    0.],
##                                     [0., 1., 0.,    0.],
##                                     [0., 0., 1., -0.06],
##                                     [0., 0., 0.,    1.]]
##                robpw_T_robga_4_4 = np.array(robpw_T_robga_4_4)
#                
#                # obse object                
#                rob_T_obj_obse_4_4 = self.compute_transformation_matrix(gazebo_T_rob_pos, gazebo_T_rob_ori, gazebo_T_obj_pos_obse, gazebo_T_obj_ori_obse)
##                rob_T_obj_obse_4_4 = np.dot(robpw_T_robga_4_4, rob_T_obj_obse_4_4)
#                pw_T_obj_obse = np.dot(pw_T_rob_sim_4_4, rob_T_obj_obse_4_4)
#                pw_T_obj_obse_pos = [pw_T_obj_obse[0][3], pw_T_obj_obse[1][3], pw_T_obj_obse[2][3]]
#                pw_T_obj_obse_ori = transformations.quaternion_from_matrix(pw_T_obj_obse)
#                obse_obj = Object_Pose(self.object_name_list[obj_index], 0, pw_T_obj_obse_pos, pw_T_obj_obse_ori, obj_index)
#                self.pw_T_target_obj_obse_pose_lsit.append(obse_obj)
#                
#            return self.pw_T_target_obj_obse_pose_lsit
                
        for obj_index in range(self.target_obj_num):
            pw_T_rob_sim_4_4 = self.pw_T_rob_sim_pose_list[0].trans_matrix
            
            # observation
            use_gazebo = ""
            if self.gazebo_flag == True:
                use_gazebo = '_noise'
                if self.dope_flag == True:
                    use_gazebo  = ""
            while_time = 0
            print("self.object_name_list[obj_index]+use_gazebo")
            print(self.object_name_list[obj_index]+use_gazebo)
            while True:
                while_time = while_time + 1
                if while_time > 1000:
                    print("WARNING: Problem happened in create_scene.py; maybe there is a problem on DOPE")
                    a = 1
                    # if self.optitrack_flag == False:
                    #     # mark
                    #     trans_ob = [0, 0, 0]
                    #     rot_ob = [0, 0, 0, 1]
                    #     break
                try:
                    (trans_ob, rot_ob) = self.listener.lookupTransform('/panda_link0', '/'+self.object_name_list[obj_index]+use_gazebo, rospy.Time(0))
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
                
                
            
            rob_T_obj_obse_pos = list(trans_ob)
            rob_T_obj_obse_ori = list(rot_ob)
            rob_T_obj_obse_3_3 = transformations.quaternion_matrix(rob_T_obj_obse_ori)
            rob_T_obj_obse_4_4 = self.rotation_4_4_to_transformation_4_4(rob_T_obj_obse_3_3, rob_T_obj_obse_pos)
            
#            if self.gazebo_flag == True:
#                robpw_T_robga_4_4 = [[1., 0., 0.,    0.],
#                                     [0., 1., 0.,    0.],
#                                     [0., 0., 1., -0.06],
#                                     [0., 0., 0.,    1.]]
#                robpw_T_robga_4_4 = np.array(robpw_T_robga_4_4)                
#                rob_T_obj_obse_4_4 = np.dot(robpw_T_robga_4_4, rob_T_obj_obse_4_4)

    
            pw_T_obj_obse = np.dot(pw_T_rob_sim_4_4, rob_T_obj_obse_4_4)
            pw_T_obj_obse_pos = [pw_T_obj_obse[0][3], pw_T_obj_obse[1][3], pw_T_obj_obse[2][3]]
            pw_T_obj_obse_ori = transformations.quaternion_from_matrix(pw_T_obj_obse)

            # mark
            # ar
            pw_T_obj_obse_pos = [0.3084858323441205, 0.09149436877842132, 0.7853899450676181]
            pw_T_obj_obse_ori = [ 0.59071692,  0.39490482,  0.58601579, -0.38947297]
            # if obj_index == 0:
            #     pw_T_obj_obse_pos = [0.34260080122584496, 0.08167958773431294, 0.7853546082956354]
            #     pw_T_obj_obse_ori = [ 0.5385327,   0.46348129,  0.53486163, -0.45726433]
            # else:
            #     pw_T_obj_obse_pos = [0.3271812145099885, 0.1747841428896269, 0.7525646138525897]
            #     pw_T_obj_obse_ori = [ 0.46895268, -0.52027885,  0.53579401, -0.47150618]




            obse_obj = Object_Pose(self.object_name_list[obj_index], 0, pw_T_obj_obse_pos, pw_T_obj_obse_ori, obj_index)
            self.pw_T_target_obj_obse_pose_lsit.append(obse_obj)
            self.trans_ob_list.append(trans_ob)
            self.rot_ob_list.append(rot_ob) # need to update
            # print("here") 
        
        return self.pw_T_target_obj_obse_pose_lsit, self.trans_ob_list, self.rot_ob_list
            
    def initialize_robot(self):
        time.sleep(0.5)
        # mark
        for rob_index in range(self.rob_num):
            if self.SIM_REAL_WORLD_FLAG == True:
                self.table_pos_1[2] = self.table_pos_1[2]
            else:
                self.table_pos_1[2] = 0
            pw_T_rob_sim_pos = [0.0, 0.0, 0.026+self.table_pos_1[2]]
            pw_T_rob_sim_pos = [0.0, 0.0, 0.02+self.table_pos_1[2]]
            pw_T_rob_sim_ori = [0, 0, 0, 1]
            pw_T_rob_sim_3_3 = transformations.quaternion_matrix(pw_T_rob_sim_ori)
            pw_T_rob_sim_4_4 = self.rotation_4_4_to_transformation_4_4(pw_T_rob_sim_3_3, pw_T_rob_sim_pos)
            joint_states = self.ros_listener.current_joint_values
            rob_pose = Robot_Pose("pandaRobot", 0, pw_T_rob_sim_pos, pw_T_rob_sim_ori, joint_states, pw_T_rob_sim_4_4, rob_index)
            self.pw_T_rob_sim_pose_list.append(rob_pose)
        return self.pw_T_rob_sim_pose_list
    
    def initialize_base_of_cheezit(self):
        opti_T_rob_opti_pos = self.ros_listener.listen_2_robot_pose()[0]
        opti_T_rob_opti_ori = self.ros_listener.listen_2_robot_pose()[1]
        pw_T_rob_sim_4_4 = self.pw_T_rob_sim_pose_list[0].trans_matrix
        for oto_index in range(self.other_obj_num):
            base_of_cheezit_pos = self.ros_listener.listen_2_object_pose("base")[0]
            base_of_cheezit_ori = self.ros_listener.listen_2_object_pose("base")[1]
            robot_T_base = self.compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, base_of_cheezit_pos, base_of_cheezit_ori)
            pw_T_base = np.dot(pw_T_rob_sim_4_4, robot_T_base)
            pw_T_base_pos = [pw_T_base[0][3], pw_T_base[1][3], pw_T_base[2][3]]
            pw_T_base_ori = transformations.quaternion_from_matrix(pw_T_base)
            opti_obj = Object_Pose(self.oto_name_list[oto_index], 0, pw_T_base_pos, pw_T_base_ori, oto_index)
            self.pw_T_other_obj_opti_pose_list.append(opti_obj)
        return self.pw_T_other_obj_opti_pose_list


    def initialize_ground_truth_objects(self):
        if self.gazebo_flag == True:
            for obj_index in range(self.target_obj_num):
                pw_T_rob_sim_4_4 = self.pw_T_rob_sim_pose_list[0].trans_matrix
                trans_gt = [0,0,0]
                rot_gt = [0,0,0,1]
                gt_name = ""
                if self.dope_flag == True:
                    gt_name = "_gt"
                    # gt_name = "_opti"

                # ground truth
                while True:
                    try:
                        (trans_gt,rot_gt) = self.listener.lookupTransform('/panda_link0', '/'+self.object_name_list[obj_index]+gt_name, rospy.Time(0))
                        break
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        continue
                rob_T_obj_opti_pos = list(trans_gt)
                rob_T_obj_opti_ori = list(rot_gt)
                rob_T_obj_opti_3_3 = transformations.quaternion_matrix(rob_T_obj_opti_ori)
                rob_T_obj_opti_4_4 = self.rotation_4_4_to_transformation_4_4(rob_T_obj_opti_3_3, rob_T_obj_opti_pos)
                
                pw_T_obj_opti = np.dot(pw_T_rob_sim_4_4, rob_T_obj_opti_4_4)
                pw_T_obj_opti_pos = [pw_T_obj_opti[0][3], pw_T_obj_opti[1][3], pw_T_obj_opti[2][3]]
                pw_T_obj_opti_ori = transformations.quaternion_from_matrix(pw_T_obj_opti)
                opti_obj = Object_Pose(self.object_name_list[obj_index], 0, pw_T_obj_opti_pos, pw_T_obj_opti_ori, obj_index)
                self.pw_T_target_obj_opti_pose_lsit.append(opti_obj)
    
    #             model_pose, _ = self.ros_listener.listen_2_object_pose(self.object_name_list[obj_index])
    #             panda_pose = self.ros_listener.listen_2_robot_pose()
                
    #             gazebo_T_obj_pos = model_pose[0]
    #             gazebo_T_obj_ori = model_pose[1]
    #             gazebo_T_rob_pos = panda_pose[0]
    #             gazebo_T_rob_ori = panda_pose[1]
                
    #             pw_T_rob_sim_4_4 = self.pw_T_rob_sim_pose_list[0].trans_matrix
                
    #             rob_T_obj_opti_4_4 = self.compute_transformation_matrix(gazebo_T_rob_pos, gazebo_T_rob_ori, gazebo_T_obj_pos, gazebo_T_obj_ori)
    # #                rob_T_obj_opti_4_4 = np.dot(robpw_T_robga_4_4, rob_T_obj_opti_4_4)
    #             pw_T_obj_opti = np.dot(pw_T_rob_sim_4_4, rob_T_obj_opti_4_4)
    #             pw_T_obj_opti_pos = [pw_T_obj_opti[0][3], pw_T_obj_opti[1][3], pw_T_obj_opti[2][3]]
    #             pw_T_obj_opti_ori = transformations.quaternion_from_matrix(pw_T_obj_opti)
    #             opti_obj = Object_Pose(self.object_name_list[obj_index], 0, pw_T_obj_opti_pos, pw_T_obj_opti_ori, obj_index)
    #             self.pw_T_target_obj_opti_pose_lsit.append(opti_obj)
                self.trans_gt_list.append(trans_gt)
                self.rot_gt_list.append(rot_gt)
                
            for obj_index in range(self.other_obj_num):
                self.pw_T_other_obj_opti_pose_list = []
                
            return self.pw_T_target_obj_opti_pose_lsit, self.pw_T_other_obj_opti_pose_list, trans_gt, rot_gt
        else:
            opti_T_rob_opti_pos = self.ros_listener.listen_2_robot_pose()[0]
            opti_T_rob_opti_ori = self.ros_listener.listen_2_robot_pose()[1]
            pw_T_rob_sim_4_4 = self.pw_T_rob_sim_pose_list[0].trans_matrix
            
            for obj_index in range(self.target_obj_num):
                obj_name = self.object_name_list[obj_index]
                # opti_T_obj_opti_pos_x = self.ros_listener.listen_2_object_pose(obj_name).pose.position.x
                # opti_T_obj_opti_pos_y = self.ros_listener.listen_2_object_pose(obj_name).pose.position.y
                # opti_T_obj_opti_pos_z = self.ros_listener.listen_2_object_pose(obj_name).pose.position.z
                # opti_T_obj_opti_pos = [opti_T_obj_opti_pos_x, opti_T_obj_opti_pos_y, opti_T_obj_opti_pos_z]
                # print("opti_T_obj_opti_pos")
                # print(opti_T_obj_opti_pos)
                # opti_T_obj_opti_ori_x = self.ros_listener.listen_2_object_pose(obj_name).pose.orientation.x
                # opti_T_obj_opti_ori_y = self.ros_listener.listen_2_object_pose(obj_name).pose.orientation.y
                # opti_T_obj_opti_ori_z = self.ros_listener.listen_2_object_pose(obj_name).pose.orientation.z
                # opti_T_obj_opti_ori_w = self.ros_listener.listen_2_object_pose(obj_name).pose.orientation.w
                # opti_T_obj_opti_ori = [opti_T_obj_opti_ori_x, opti_T_obj_opti_ori_y, opti_T_obj_opti_ori_z, opti_T_obj_opti_ori_w]
                # print("opti_T_obj_opti_ori")
                # print(opti_T_obj_opti_ori)
                opti_T_obj_opti_pos = self.ros_listener.listen_2_object_pose(obj_name)[0]
                opti_T_obj_opti_ori = self.ros_listener.listen_2_object_pose(obj_name)[1]

                rob_T_obj_opti_4_4 = self.compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, opti_T_obj_opti_pos, opti_T_obj_opti_ori)
                pw_T_obj_opti_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_obj_opti_4_4)
                pw_T_obj_opti_pos = [pw_T_obj_opti_4_4[0][3], pw_T_obj_opti_4_4[1][3], pw_T_obj_opti_4_4[2][3]]
                pw_T_obj_opti_ori = transformations.quaternion_from_matrix(pw_T_obj_opti_4_4)
                opti_obj = Object_Pose(self.object_name_list[obj_index], 0, pw_T_obj_opti_pos, pw_T_obj_opti_ori, obj_index)
                self.pw_T_target_obj_opti_pose_lsit.append(opti_obj)
            
            for obj_index in range(self.other_obj_num):
                base_of_cheezit_pos = self.ros_listener.listen_2_object_pose("base")[0]
                base_of_cheezit_ori = self.ros_listener.listen_2_object_pose("base")[1]
                print(base_of_cheezit_pos)
                print(base_of_cheezit_ori)
                robot_T_base = self.compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, base_of_cheezit_pos, base_of_cheezit_ori)
                pw_T_base = np.dot(pw_T_rob_sim_4_4, robot_T_base)
                pw_T_base_pos = [pw_T_base[0][3], pw_T_base[1][3], pw_T_base[2][3]]
                pw_T_base_ori = transformations.quaternion_from_matrix(pw_T_base)
                opti_obj = Object_Pose(self.object_name_list[obj_index], 0, pw_T_base_pos, pw_T_base_ori, obj_index)
                self.pw_T_other_obj_opti_pose_list.append(opti_obj)
            trans_gt = 0
            rot_gt = 0
            return self.pw_T_target_obj_opti_pose_lsit, self.pw_T_other_obj_opti_pose_list, trans_gt, rot_gt
        
        
        
    
    def rotation_4_4_to_transformation_4_4(self, rotation_4_4, pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4
    
    def compute_transformation_matrix(self, a_pos, a_ori, b_pos, b_ori):
        ow_T_a_3_3 = transformations.quaternion_matrix(a_ori)
        ow_T_a_4_4 = self.rotation_4_4_to_transformation_4_4(ow_T_a_3_3, a_pos)
        ow_T_b_3_3 = transformations.quaternion_matrix(b_ori)
        ow_T_b_4_4 = self.rotation_4_4_to_transformation_4_4(ow_T_b_3_3, b_pos)
        a_T_ow_4_4 = np.linalg.inv(ow_T_a_4_4)
        a_T_b_4_4 = np.dot(a_T_ow_4_4,ow_T_b_4_4)
        return a_T_b_4_4

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
        mean = cur_angle
        ang_noise_sigma = 0.1
        sigma = ang_noise_sigma
        new_angle_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_angle_is_added_noise
    
    def take_easy_gaussian_value(self, mean, sigma):
        normal = random.normalvariate(mean, sigma)
        return normal
