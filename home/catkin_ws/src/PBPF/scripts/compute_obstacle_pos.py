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
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3
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
import yaml
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
#Class of franka robot listen to info from ROS
class Ros_Listener():
    def __init__(self):
        with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
            self.parameter_info = yaml.safe_load(file)
        self.gazebo_flag = self.parameter_info['gazebo_flag']
        
        rospy.Subscriber('/joint_states', JointState, self.joint_values_callback, queue_size=1)
        self.joint_subscriber = JointState()
        
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)
        self.model_states = ModelStates()
        
        rospy.Subscriber('/mocap/rigid_bodies/pandaRobot/pose', PoseStamped, self.robot_pose_callback, queue_size=1)
        self.robot_pose = PoseStamped()
        
        rospy.Subscriber('/mocap/rigid_bodies/cheezit/pose', PoseStamped, self.object_pose_callback_cracker, queue_size=1)
        self.object_cracker_pose = PoseStamped()
        
        rospy.Subscriber('/mocap/rigid_bodies/zisongsoup/pose', PoseStamped, self.object_pose_callback_soup, queue_size=1)
        self.object_soup_pose = PoseStamped()
        
        rospy.Subscriber('/mocap/rigid_bodies/baseofcheezit/pose', PoseStamped, self.base_of_cheezit_callback, queue_size=1)
        self.base_pose = PoseStamped()

        rospy.Subscriber('/mocap/rigid_bodies/smallObstacle/pose', PoseStamped, self.smallObstacle_callback, queue_size=1)
        self.smallObstacle = PoseStamped()
        
        rospy.Subscriber('/mocap/rigid_bodies/bigObstacle/pose', PoseStamped, self.bigObstacle_callback, queue_size=1)
        self.bigObstacle = PoseStamped()

        rospy.Subscriber('/Opti_pose', PoseStamped, self.fake_optipose_callback, queue_size=10)
        self.fake_opti_pose = PoseStamped()
        
        rospy.Subscriber('/esti_obj_list', estimated_obj_pose, self.esti_obj_states_callback, queue_size=10)
        self.esti_obj_states_list = estimated_obj_pose()
        
        rospy.Subscriber('/par_list', particle_list, self.particles_states_callback, queue_size=10)
        self.particles_states_list = particle_list()
        
        self.pos_added_noise = []
        self.ori_added_noise = []
        self.model_pose_added_noise = []
        self.rob_T_obj_obse_4_4 = []
        
        rospy.spin
    
    def model_states_callback(self, model_states):
        gzb_T_obj_obse_4_4 = [[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]]
        gzb_T_obj_obse_4_4 = np.array(gzb_T_obj_obse_4_4)  
        gzb_T_rob_obse_4_4 = [[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]]
        gzb_T_rob_obse_4_4 = np.array(gzb_T_rob_obse_4_4)  
        name_lenght = len(model_states.name)
        for name_index in range(name_lenght):
            if model_states.name[name_index] == "cracker":
                model_name = model_states.name[name_index]
                model_pos = model_states.pose[name_index].position
                model_ori = model_states.pose[name_index].orientation
                self.model_pos = [model_pos.x, model_pos.y, model_pos.z]
                self.model_ori = [model_ori.x, model_ori.y, model_ori.z, model_ori.w]
                self.model_pose = [self.model_pos, self.model_ori]
                
                gzb_T_obj_obse_3_3 = transformations.quaternion_matrix(self.model_ori)
                gzb_T_obj_obse_4_4 = self.rotation_4_4_to_transformation_4_4(gzb_T_obj_obse_3_3, self.model_pos)
#                gzb_T_obj_opti_4_4 = np.dot(robpw_T_robga_4_4, rob_T_obj_opti_4_4)
            
            if model_states.name[name_index] == "panda":
#                self.pos_added_noise, self.ori_added_noise = self.add_noise_pose(self.model_pos, self.model_ori)
#                self.model_pose_added_noise = [self.pos_added_noise, self.ori_added_noise]
                panda_name = model_states.name[name_index]
                panda_pos = model_states.pose[name_index].position
                panda_ori = model_states.pose[name_index].orientation
                self.panda_pos = [panda_pos.x, panda_pos.y, panda_pos.z]
                self.panda_ori = [panda_ori.x, panda_ori.y, panda_ori.z, panda_ori.w]
                self.panda_pose = [self.panda_pos, self.panda_ori]
                
                gzb_T_rob_obse_3_3 = transformations.quaternion_matrix(self.panda_ori)
                gzb_T_rob_obse_4_4 = self.rotation_4_4_to_transformation_4_4(gzb_T_rob_obse_3_3, self.panda_ori)
                robpos_T_pandalink0_4_4 = [[1., 0., 0.,    0.],
                                           [0., 1., 0.,    0.],
                                           [0., 0., 1.,  0.06],
                                           [0., 0., 0.,    1.]]
                robpos_T_pandalink0_4_4 = np.array(robpos_T_pandalink0_4_4)                
                gazebo_T_pandalink0_opti_4_4 = np.dot(gzb_T_rob_obse_4_4, robpos_T_pandalink0_4_4)        
                
            if model_states.name[name_index] == "fish_can":
                model_name = model_states.name[name_index]
                model_pos = model_states.pose[name_index].position
                model_ori = model_states.pose[name_index].orientation
                self.model_pos = [model_pos.x, model_pos.y, model_pos.z]
                self.model_ori = [model_ori.x, model_ori.y, model_ori.z, model_ori.w]
                self.model_pose = [self.model_pos, self.model_ori]
                
                gzb_T_fish_obse_3_3 = transformations.quaternion_matrix(self.model_ori)
                gzb_T_fish_obse_4_4 = self.rotation_4_4_to_transformation_4_4(gzb_T_fish_obse_3_3, self.model_pos)
                
#        pandalink0_T_gzb_obse_4_4 = np.linalg.inv(gzb_T_rob_obse_4_4)
        pandalink0_T_gzb_obse_4_4 = np.linalg.inv(gazebo_T_pandalink0_opti_4_4)
        pandalink0_T_obj_obse_4_4 = np.dot(pandalink0_T_gzb_obse_4_4, gzb_T_obj_obse_4_4)
        pandalink0_T_fish_obse_4_4 = np.dot(pandalink0_T_gzb_obse_4_4, gzb_T_fish_obse_4_4)
        self.rob_T_obj_obse_4_4 = pandalink0_T_obj_obse_4_4
#        self.rob_T_obj_obse_4_4 = pandalink0_T_fish_obse_4_4

    def listen_2_test_matrix(self):
        return self.rob_T_obj_obse_4_4

    def listen_2_object_pose(self, object_flag):
        if object_flag == "cracker":
            if self.gazebo_flag == True:
                return self.model_pose, self.model_pose_added_noise
            return self.object_cracker_pose
        elif object_flag == "soup":
            return self.object_soup_pose
        elif object_flag == "base":
            return self.base_pose
        elif object_flag == "smallobstacle":
            return self.smallObstacle_pose
        elif object_flag == "bigobstacle":
            return self.bigObstacle_pose
        elif object_flag == "panda_robot":
            return self.robot_pose
    
    def listen_2_pars_states(self):
        return self.particles_states_list
    
    def listen_2_estis_states(self):
        return self.esti_obj_states_list
    
    def listen_2_gazebo_robot_pose(self):
        return self.panda_pose
    
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
        self.robot_pose = [self.robot_pos, self.robot_ori]
        
    def object_pose_callback_cracker(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_cracker_pose = [self.object_pos, self.object_ori]
        
    def object_pose_callback_soup(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_soup_pose = [self.object_pos, self.object_ori]
        
    def base_of_cheezit_callback(self, data):
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
        self.base_pose = [self.base_pos, self.base_ori]
        # print("self.base_pose:", self.base_pose)

    def smallObstacle_callback(self, data):
        # pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.smallObstacle_pos = [x_pos, y_pos, z_pos]
        # ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.smallObstacle_ori = [x_ori, y_ori, z_ori, w_ori]
        self.smallObstacle_pose = [self.smallObstacle_pos, self.smallObstacle_ori]
        # print("smallObstacle_pose:",self.smallObstacle_pose)
        # print("self.smallObstacle_pose:", self.smallObstacle_pose)

    def bigObstacle_callback(self, data):
        # pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.bigObstacle_pos = [x_pos, y_pos, z_pos]
        # ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.bigObstacle_ori = [x_ori, y_ori, z_ori, w_ori]
        self.bigObstacle_pose = [self.bigObstacle_pos, self.bigObstacle_ori]
        # print("self.bigObstacle_pose:", self.bigObstacle_pose)
        
    def fake_optipose_callback(self, data):
        # pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.fake_opti_pos = [x_pos, y_pos, z_pos]
        # ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.fake_opti_ori = [x_ori, y_ori, z_ori, w_ori]
        self.fake_opti_pose = [self.fake_opti_pos, self.fake_opti_ori]

    def particles_states_callback(self, pars_states_list):
        self.particles_states_list = pars_states_list
    
    def esti_obj_states_callback(self, esti_objs_list):
        self.esti_obj_states_list = esti_objs_list
        
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
    
    def rotation_4_4_to_transformation_4_4(self, rotation_4_4, pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4

if __name__ == '__main__':
    rospy.init_node('compute_obstacle_pos_node') # ros node
    ros_listener = Ros_Listener()
    robot_pose = ros_listener.robot_pose
    # obstacle_pose = ros_listener.obstacle_pose
    time.sleep(0.5)

    p_visualisation = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT, GUI_SERVER
    p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
    p_visualisation.setGravity(0, 0, -9.81)
    p_visualisation.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0.3,0.1,0.2])      
    # p_visualisation.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85, cameraTargetPosition=[0.3,0.1,0.2])    
    plane_id = p_visualisation.loadURDF("plane.urdf")

    obstacle_pose = ros_listener.listen_2_object_pose("smallobstacle")
    # obstacle_pose = ros_listener.listen_2_object_pose("smallobstacle")
    robot_pose = ros_listener.listen_2_object_pose("panda_robot")
    print("obstacle_pose:")
    print(obstacle_pose)
    print("robot_pose:")
    print(robot_pose)
    opti_T_rob_pose = robot_pose
    opti_T_rob_pos = copy.deepcopy(robot_pose[0])
    opti_T_rob_ori = copy.deepcopy(robot_pose[1]) # x,y,z,w
    opti_T_rob_sim_3_3 = transformations.quaternion_matrix(opti_T_rob_ori)
    opti_T_rob_sim_4_4 = ros_listener.rotation_4_4_to_transformation_4_4(opti_T_rob_sim_3_3, opti_T_rob_pos)

    opti_T_obst_pose = obstacle_pose
    opti_T_obst_pos = copy.deepcopy(obstacle_pose[0])
    opti_T_obst_ori = copy.deepcopy(obstacle_pose[1]) # x,y,z,w
    opti_T_obst_sim_3_3 = transformations.quaternion_matrix(opti_T_obst_ori)
    opti_T_obst_sim_4_4 = ros_listener.rotation_4_4_to_transformation_4_4(opti_T_obst_sim_3_3, opti_T_obst_pos)

    pw_T_rob_sim_pos = [0.0, 0.0, 0.026]
    pw_T_rob_sim_pos = [0.0, 0.0, 0.02]
    pw_T_rob_sim_ori = [0, 0, 0, 1]
    pw_T_rob_sim_3_3 = transformations.quaternion_matrix(pw_T_rob_sim_ori)
    pw_T_rob_sim_4_4 = ros_listener.rotation_4_4_to_transformation_4_4(pw_T_rob_sim_3_3, pw_T_rob_sim_pos)

    rob_T_opti_sim_4_4 = np.linalg.inv(opti_T_rob_sim_4_4)
    rob_T_obst_opti_4_4 = np.dot(rob_T_opti_sim_4_4, opti_T_obst_sim_4_4)
    pw_T_obst_opti_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_obst_opti_4_4)

    pw_T_obst_opti_pos = [pw_T_obst_opti_4_4[0][3],pw_T_obst_opti_4_4[1][3],pw_T_obst_opti_4_4[2][3]]
    pw_T_obst_opti_ori = transformations.quaternion_from_matrix(pw_T_obst_opti_4_4)

    print(pw_T_obst_opti_pos)
    print(pw_T_obst_opti_ori)
    track_fk_rob_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_obstacle_small.urdf"),
                                              pw_T_obst_opti_pos,
                                              pw_T_obst_opti_ori,
                                              useFixedBase=1)
        
    while True:
        p_visualisation.stepSimulation()