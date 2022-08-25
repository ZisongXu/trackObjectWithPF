# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:57:49 2021

@author: 12106
"""
#ROS
import itertools
import os.path

import rospy
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
import pybullet as p
import time
import pybullet_data
from pybullet_utils import bullet_client as bc
import numpy as np
import math
import random
import copy
import os

def print_(b):
    print(b)
    
rospy.init_node('rate_test')
rate = rospy.Rate(1)
a0 = time.time()
while True:
    print("test")
    a = time.time()
    print(a - a0)
    while True:
        b = time.time()
        print(b - a0)
        rate.sleep()
        break
    