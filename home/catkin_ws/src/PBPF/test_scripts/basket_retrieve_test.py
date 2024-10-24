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
import yaml



p_env = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT,GUI_SERVER
p_env.resetDebugVisualizerCamera(cameraDistance=1., cameraYaw=90, cameraPitch=-50, cameraTargetPosition=[0.1,0.15,0.35])  
p_env.setAdditionalSearchPath(pybullet_data.getDataPath())
p_env.setGravity(0, 0, -9.81)
p_env.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)

plane_id = p_env.loadURDF("plane.urdf")

table_pos_1 = [0.46, -0.01, 0.710]
table_ori_1 = p_env.getQuaternionFromEuler([0,0,0])
table_id_1 = p_env.loadURDF(os.path.expanduser("~/project/object/others/table.urdf"), table_pos_1, table_ori_1, useFixedBase=True)

basket_pos_1 = [0.46, -0.01, 1.2]
basket_ori_1 = p_env.getQuaternionFromEuler([0,0,0])
basket_id_1 = p_env.loadURDF(os.path.expanduser("~/project/object/others/basket.urdf"), basket_pos_1, basket_ori_1)


while True:
    p_env.stepSimulation()
    time.sleep(1./240.)
