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
# CVPF Pose list (motion model)


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

test_point1 = [0., 0.3, 0.0841]
test_point2 = [0.,-0.3, 0.0841]
#test_point2 = [0., 0., 0.7]

callback = p_visualisation.rayTest(test_point1, test_point2)

contact_obj_ID = callback[0][0]

print(callback)

if contact_obj_ID == test1_objID:
    print("test1_objID")
if contact_obj_ID == test2_objID:
    print("test2_objID")
while True:
    continue



































    


