#!/usr/bin/python3
#Class of particle's structure
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
from gazebo_msgs.msg import ModelStates
from PBPF.msg import estimated_obj_pose, object_pose, particle_list, particle_pose
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
#class in other files
from Franka_robot import Franka_robot
from Ros_Listener import Ros_Listener
from Particle import Particle
from InitialSimulationModel import InitialSimulationModel
from Realworld import Realworld
from Visualisation_World import Visualisation_World
from Create_Scene import Create_Scene
from Object_Pose import Object_Pose
import yaml

class Ros_Listener():
    def __init__(self):
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)
        self.model_states = ModelStates()
        rospy.Subscriber('/joint_states', JointState, self.joint_values_callback, queue_size=1)
        self.joint_subscriber = JointState()
        rospy.spin
        
    def model_states_callback(self, model_states):
        self.model_name = model_states.name[5]
        self.model_pos = model_states.pose[5].position
        self.model_ori = model_states.pose[5].orientation
#        print(model_states.name[5])
#        print(model_states.pose[5].position.x)

    def joint_values_callback(self, msg):
        self.current_joint_values = list(msg.position) 
#        print(self.current_joint_values)

#    def pose_from_gazebo(self, pose_list):
#        self.model_pose = pose_list
#        return self.model_pose
#    
#    def listen_2_robot_joint(self):
#        return self.current_joint_values
#    
#    def joint_values_callback(self, msg):
#        self.current_joint_values = list(msg.position)  
#        
#        
#rospy.init_node('listener')
#listener = Ros_Listener()
#print(listener.listen_2_robot_joint())
#def callback(data):
#    rospy.loginfo(data.position)
#
#def callback_pose(data):
#    print(data.name[5])
#    print(data.pose[5].position.x)
##    print(data.pose.orientation[5])
##    print(data.twist.linear[5])
##    print(data.twist.angular[5])
#    # rospy.loginfo(data)
#
#def listener():
#    rospy.init_node('listener', anonymous=True)
#    rospy.Subscriber('/joint_states', JointState, callback, queue_size=1)
#    rospy.spin()
#
#
#def pose_model():
#    rospy.init_node('listener', anonymous=True)
#    rospy.Subscriber('/gazebo/model_states', ModelStates, callback_pose, queue_size=1)
#    rospy.spin()

# add position into transformation matrix
def rotation_4_4_to_transformation_4_4(rotation_4_4, pos):
    rotation_4_4[0][3] = pos[0]
    rotation_4_4[1][3] = pos[1]
    rotation_4_4[2][3] = pos[2]
    return rotation_4_4

# add noise
def add_noise_pose(sim_par_cur_pos, sim_par_cur_ori):
    normal_x = add_noise_2_par(sim_par_cur_pos[0], "x")
    normal_y = add_noise_2_par(sim_par_cur_pos[1], "y")
    normal_z = add_noise_2_par(sim_par_cur_pos[2], "z")
    pos_added_noise = [normal_x, normal_y, normal_z]
    # add noise on ang of each particle
    quat = copy.deepcopy(sim_par_cur_ori)# x,y,z,w
    quat_QuatStyle = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])# w,x,y,z
    random_dir = random.uniform(0, 2*math.pi)
    z_axis = random.uniform(-1,1)
    x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
    y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
    angle_noise = add_noise_2_ang(0)
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

def add_noise_2_par(current_pos, axis):
    mean = current_pos
    pos_noise_sigma = 0.02
    if axis == "z":
        pos_noise_sigma = 0.001
    sigma = pos_noise_sigma
    new_pos_is_added_noise = take_easy_gaussian_value(mean, sigma)
    return new_pos_is_added_noise

def add_noise_2_ang(cur_angle):
    mean = cur_angle
    ang_noise_sigma = 0.13
    sigma = ang_noise_sigma
    new_angle_is_added_noise = take_easy_gaussian_value(mean, sigma)
    return new_angle_is_added_noise

# random values generated from a Gaussian distribution
def take_easy_gaussian_value(mean, sigma):
    normal = random.normalvariate(mean, sigma)
    return normal

def convert_dope_to_tf(position, orientation, object_name):
    br.sendTransform(position, orientation, rospy.Time.now(), child=object_name, parent='panda_link0')

# ctrl-c write down the error file
def signal_handler(sig, frame):
    sys.exit()
    
if __name__ == '__main__':
    rospy.init_node('add_noise', anonymous=True)
    listener_tf = tf.TransformListener()
    br = tf.TransformBroadcaster()
    signal.signal(signal.SIGINT, signal_handler) # interrupt judgment
    time.sleep(0.5)
    with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
        parameter_info = yaml.safe_load(file)
    
    gazebo_flag = parameter_info['gazebo_flag']
    # which algorithm to run
    run_alg_flag = parameter_info['run_alg_flag'] # PBPF/CVPF
    # update mode (pose/time)
    update_style_flag = parameter_info['update_style_flag'] # time/pose
    object_num = parameter_info['object_num']
    object_name_list = parameter_info['object_name_list']
    particle_num = parameter_info['particle_num']
    
    first_run_flag = 0
    
#    for obj_index in range(object_num):
#        rospy.Subscriber('/dope/pose_' + object_name_list[obj_index], PoseStamped, convert_dope_to_tf, object_name, queue_size=1)
        
    while True:
        for obj_index in range(object_num):
            gt_name = "_gt"
            if first_run_flag == 0:
                if obj_index == object_num - 1:
                    first_run_flag = 1
                    
                while True:
                    try:
                        (trans_ob,rot_ob) = listener_tf.lookupTransform('/panda_link0', '/'+object_name_list[obj_index]+gt_name, rospy.Time(0))
                        break
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        continue
            obse_is_fresh = True
            try:
                latest_obse_time = listener_tf.getLatestCommonTime('/panda_link0', '/'+object_name_list[obj_index]+gt_name)
                if (rospy.get_time() - latest_obse_time.to_sec()) < 0.1:
                    (trans_ob,rot_ob) = listener_tf.lookupTransform('/panda_link0', '/'+object_name_list[obj_index]+gt_name, rospy.Time(0))
                    obse_is_fresh = True
                    # print("obse is FRESH")
                else:
                    # obse has not been updating for a while
                    obse_is_fresh = False
                    print("obse is NOT fresh")
                # break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("can not find tf")
                    
            rob_T_obj_obse_pos = list(trans_ob)
            rob_T_obj_obse_ori = list(rot_ob)
            rob_T_obj_obse_3_3 = transformations.quaternion_matrix(rob_T_obj_obse_ori)
            rob_T_obj_obse_4_4 = rotation_4_4_to_transformation_4_4(rob_T_obj_obse_3_3, rob_T_obj_obse_pos)
            
            object_name = object_name_list[obj_index]+'_noise'
            pos_added_noise, ori_added_noise = add_noise_pose(rob_T_obj_obse_pos, rob_T_obj_obse_ori)
            convert_dope_to_tf(pos_added_noise, ori_added_noise, object_name)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
