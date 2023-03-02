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
# CVPF Pose list (motion model)

def rotation_4_4_to_transformation_4_4(rotation_4_4, pos):
    rotation_4_4[0][3] = pos[0]
    rotation_4_4[1][3] = pos[1]
    rotation_4_4[2][3] = pos[2]
    return rotation_4_4

p_visualisation = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT, GUI_SERVER
p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
p_visualisation.setGravity(0, 0, -9.81)
p_visualisation.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0.3,0.1,0.2])      
# p_visualisation.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85, cameraTargetPosition=[0.3,0.1,0.2])    
plane_id = p_visualisation.loadURDF("plane.urdf")
other_obj_pos = [0., 0., 0.0841]
other_obj_ori = [0,0,0,1]
test1_objID = p_visualisation.loadURDF(os.path.expanduser("~/project/object/base_of_cracker/base_of_cracker.urdf"),
                                      other_obj_pos,
                                      other_obj_ori)
other_obj_pos = [0., 0., 0.5841]
other_obj_ori = [0,0,0,1]
test2_objID = p_visualisation.loadURDF(os.path.expanduser("~/project/object/base_of_cracker/base_of_cracker.urdf"),
                                      other_obj_pos,
                                      other_obj_ori)


x_w = 0.159
y_l = 0.21243700408935547
z_h = 0.06
pw_T_center_pos = [0, 0, 0]
pw_T_center_ori = [0, 0, 0, 1] # x ,y, z, w
pw_T_center_3_3 = transformations.quaternion_matrix(pw_T_center_ori)
pw_T_center_4_4 = rotation_4_4_to_transformation_4_4(pw_T_center_3_3, pw_T_center_pos)

# 1 means positive
# -1 means negative
def generate_point_for_ray(pw_T_c_pos, pw_T_parC_4_4):
    vector_list = [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
                   [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
                   [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
                   [1,0.5,0.5], [1,0.5,-0.5], [1,-0.5,0.5], [1,-0.5,-0.5],
                   [-1,0.5,0.5], [-1,0.5,-0.5], [-1,-0.5,0.5], [-1,-0.5,-0.5],
                   [0.5,1,0.5], [0.5,1,-0.5], [-0.5,1,0.5], [-0.5,1,-0.5],
                   [0.5,-1,0.5], [0.5,-1,-0.5], [-0.5,-1,0.5], [-0.5,-1,-0.5],
                   [0.5,0.5,1], [0.5,-0.5,1], [-0.5,0.5,1], [-0.5,-0.5,1],
                   [0.5,0.5,-1], [0.5,-0.5,-1], [-0.5,0.5,-1], [-0.5,-0.5,-1]]
    point_list = []
    point_pos_list = []
    for index in range(38):
        x_new = pw_T_c_pos[0] + vector_list[index][0] * x_w/2
        y_new = pw_T_c_pos[1] + vector_list[index][1] * y_l/2
        z_new = pw_T_c_pos[2] + vector_list[index][2] * z_h/2
        parC_T_p_pos = [x_new, y_new, z_new]
        parC_T_p_ori = [0, 0, 0, 1] # x, y, z, w
        parC_T_p_3_3 = transformations.quaternion_matrix(parC_T_p_ori)
        parC_T_p_4_4 = rotation_4_4_to_transformation_4_4(parC_T_p_3_3, parC_T_p_pos)
        pw_T_p_4_4 = np.dot(pw_T_parC_4_4, parC_T_p_4_4)
        pw_T_p_pos = [pw_T_p_4_4[0][3], pw_T_p_4_4[1][3], pw_T_p_4_4[2][3]]
        pw_T_p_ori = transformations.quaternion_from_matrix(pw_T_p_4_4)
        pw_T_p_pose = Center_T_Point_for_Ray(pw_T_p_pos, pw_T_p_ori, parC_T_p_4_4, index)
        point_list.append(pw_T_p_pose)
        point_pos_list.append(pw_T_p_pos)
    return point_list, point_pos_list

pw_T_parC_4_4 = [[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]]

generate_point_for_ray(pw_T_center_pos, pw_T_parC_4_4)
input("stop")

#center_T_p2_pos = 
#center_T_p2_ori = 
#center_T_p3_pos = 
#center_T_p3_ori = 
#center_T_p4_pos = 
#center_T_p4_ori = 
#center_T_p5_pos = 
#center_T_p5_ori = 
#center_T_p6_pos = 
#center_T_p6_ori = 
#center_T_p7_pos = 
#center_T_p7_ori = 
#center_T_p8_pos = 
#center_T_p8_ori = 


print(pw_T_center_4_4)

test_point1 = [0., 0.3, 0.0841]
test_point11 = [0., 0.3, 0.241]
test_point2 = [0.,-0.3, 0.0841]
test_point22 = [0.,-0.3, 0.241]
test_point3 = [0.,-0.3,-0.0841]
#test_point2 = [0., 0., 0.7]

test_list1 = [test_point1, test_point11]
test_list2 = [test_point2, test_point22]
p_visualisation.stepSimulation()

callback = p_visualisation.rayTest(test_point3, test_point2)

callback1 = p_visualisation.rayTestBatch(test_list1, test_list2)


contact_obj_ID = callback[0][0]

print("plane_id:", plane_id)
print(callback)

if contact_obj_ID == test1_objID:
    print("test1_objID")
if contact_obj_ID == test2_objID:
    print("test2_objID")
while True:
    for i in range(240):
        p_visualisation.stepSimulation()
        time.sleep(1/240)
#    callback = p_visualisation.rayTest(test_point3, test_point2)
#    contact_obj_ID = callback[0][0]
#    if contact_obj_ID == test1_objID:
#        print("test1_objID")
    continue



































    


