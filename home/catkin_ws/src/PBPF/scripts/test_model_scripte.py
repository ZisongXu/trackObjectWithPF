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
from Create_Scene import Create_Scene
from Ros_Listener import Ros_Listener
import yaml

# (Basic Setting

p_visualisation = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT, GUI_SERVER


p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
# p_visualisation.configureDebugVisualizer(p.COV_WINDOW_TITLE, "Show PBPF")
p_visualisation.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p_visualisation.setGravity(0, 0, -9.81)
# p_visualisation.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0.1,0.1,0.1])      
p_visualisation.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85, cameraTargetPosition=[0.3,0.1,0.1])    
plane_id = p_visualisation.loadURDF("plane.urdf")

# pw_T_obst_opti_pos_small = [0.852134144216095, 0.14043691336334274, 0.10014295215002848]
# pw_T_obst_opti_ori_small = [0.00356749, -0.00269526, 0.28837681, 0.95750657]
pw_T_obst_opti_pos_big = [0.20, 0.20, 0.70]
pw_T_obst_opti_ori_big = [0, 0, 0, 1]
track_fk_obst_big_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/shelves.urdf"),
                                                pw_T_obst_opti_pos_big,
                                                pw_T_obst_opti_ori_big)

while True:
    a = 1
    p_visualisation.stepSimulation()
    time.sleep(1./240.)
    # time.sleep(1)
    
# for i in range(240):
#     p_visualisation.stepSimulation()