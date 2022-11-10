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
from Ros_Listener import Ros_Listener
from Object_Pose import Object_Pose
from Robot_Pose import Robot_Pose
#Class of initialize the real world model
class Create_Scene():
    def __init__(self, target_obj_num=0, rob_num=0, other_obj_num=0):
        self.target_obj_num = target_obj_num
        self.rob_num = rob_num
        self.other_obj_num = other_obj_num
        self.pw_T_target_obj_opti_pose_lsit = []
        self.pw_T_target_obj_obse_pose_lsit = []
        self.pw_T_rob_sim_pose_list = []
        self.pw_T_other_obj_opti_pose_list = []
        self.ros_listener = Ros_Listener(True, "cracker")
        self.listener = tf.TransformListener()
        
    def initialize_object(self):
        objects_name_list = ["cracker", "soup"]
        for obj_index in range(self.target_obj_num):
            pw_T_rob_sim_pos = self.pw_T_rob_sim_pose_list[0].pos
            pw_T_rob_sim_ori = self.pw_T_rob_sim_pose_list[0].ori
            pw_T_rob_sim_3_3 = transformations.quaternion_matrix(pw_T_rob_sim_ori)
            pw_T_rob_sim_4_4 = self.rotation_4_4_to_transformation_4_4(pw_T_rob_sim_3_3, pw_T_rob_sim_pos)
            self.pw_T_rob_sim_pose_list[0].trans_matrix = pw_T_rob_sim_4_4
            while True:
                try:
                    (trans,rot) = self.listener.lookupTransform('/panda_link0', '/'+objects_name_list[obj_index], rospy.Time(0))
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            rob_T_obj_obse_pos = list(trans)
            rob_T_obj_obse_ori = list(rot)
            rob_T_obj_obse_3_3 = transformations.quaternion_matrix(rob_T_obj_obse_ori)
            rob_T_obj_obse_4_4 = self.rotation_4_4_to_transformation_4_4(rob_T_obj_obse_3_3, rob_T_obj_obse_pos)
            
            pw_T_obj_obse = np.dot(pw_T_rob_sim_4_4, rob_T_obj_obse_4_4)
            pw_T_obj_obse_pos = [pw_T_obj_obse[0][3], pw_T_obj_obse[1][3], pw_T_obj_obse[2][3]]
            pw_T_obj_obse_ori = transformations.quaternion_from_matrix(pw_T_obj_obse)
            obse_obj = Object_Pose(objects_name_list[obj_index], 0, pw_T_obj_obse_pos, pw_T_obj_obse_ori, obj_index)
            self.pw_T_target_obj_obse_pose_lsit.append(obse_obj)
        
        opti_T_rob_opti_pos = self.ros_listener.listen_2_robot_pose()[0]
        opti_T_rob_opti_ori = self.ros_listener.listen_2_robot_pose()[1]

        for obj_index in range(self.other_obj_num):
            base_of_cheezit_pos = self.ros_listener.listen_2_object_pose("base")[0]
            base_of_cheezit_ori = self.ros_listener.listen_2_object_pose("base")[1]
            robot_T_base = self.compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, base_of_cheezit_pos, base_of_cheezit_ori)
            pw_T_base = np.dot(pw_T_rob_sim_4_4, robot_T_base)
            pw_T_base_pos = [pw_T_base[0][3], pw_T_base[1][3], pw_T_base[2][3]]
            pw_T_base_ori = transformations.quaternion_from_matrix(pw_T_base)
            opti_obj = Object_Pose(objects_name_list[obj_index], 0, pw_T_base_pos, pw_T_base_ori, obj_index)
            self.pw_T_other_obj_opti_pose_list.append(opti_obj)
            
        for obj_index in range(self.target_obj_num):
            obj_name = objects_name_list[obj_index]
            opti_T_obj_opti_pos = self.ros_listener.listen_2_object_pose(obj_name)[0]
            opti_T_obj_opti_ori = self.ros_listener.listen_2_object_pose(obj_name)[1]
            rob_T_obj_opti_4_4 = self.compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, opti_T_obj_opti_pos, opti_T_obj_opti_ori)
            pw_T_obj_opti_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_obj_opti_4_4)
            pw_T_obj_opti_pos = [pw_T_obj_opti_4_4[0][3], pw_T_obj_opti_4_4[1][3], pw_T_obj_opti_4_4[2][3]]
            pw_T_obj_opti_ori = transformations.quaternion_from_matrix(pw_T_obj_opti_4_4)
            opti_obj = Object_Pose(objects_name_list[obj_index], 0, pw_T_obj_opti_pos, pw_T_obj_opti_ori, obj_index)
            self.pw_T_target_obj_opti_pose_lsit.append(opti_obj)
        return self.pw_T_target_obj_obse_pose_lsit, self.pw_T_target_obj_opti_pose_lsit, self.pw_T_other_obj_opti_pose_list
            
    def initialize_robot(self):
        for rob_index in range(self.rob_num):
            pw_T_rob_sim_pos = [0.0, 0.0, 0.026]
            pw_T_rob_sim_ori = [0, 0, 0, 1]
            rob_pose = Robot_Pose("pandaRobot", 0, pw_T_rob_sim_pos, pw_T_rob_sim_ori, 0, 0, rob_index)
            self.pw_T_rob_sim_pose_list.append(rob_pose)
        return self.pw_T_rob_sim_pose_list, self.listener
    
    def rotation_4_4_to_transformation_4_4(self, rotation_4_4, pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4
    
    def compute_transformation_matrix(self, a_pos, a_ori, b_pos, b_ori):
        ow_T_a_3_3 = transformations.quaternion_matrix(a_ori)
        ow_T_a_4_4 = self.rotation_4_4_to_transformation_4_4(ow_T_a_3_3,a_pos)
        ow_T_b_3_3 = transformations.quaternion_matrix(b_ori)
        ow_T_b_4_4 = self.rotation_4_4_to_transformation_4_4(ow_T_b_3_3,b_pos)
        a_T_ow_4_4 = np.linalg.inv(ow_T_a_4_4)
        a_T_b_4_4 = np.dot(a_T_ow_4_4,ow_T_b_4_4)
        return a_T_b_4_4