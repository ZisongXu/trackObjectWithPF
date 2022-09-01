# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:57:49 2021

@author: 12106
"""
#ROS
#ROS
from concurrent.futures.process import _threads_wakeups
import itertools
import os.path
from ssl import ALERT_DESCRIPTION_ILLEGAL_PARAMETER

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
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
#from sksurgerycore.algorithms.averagequaternions import average_quaternions
from quaternion_averaging import weightedAverageQuaternions

rospy.init_node('rate_test')
rate = rospy.Rate(10)
try:
    while not rospy.is_shutdown():
        print("rate")
        rate.sleep()
except KeyboardInterrupt:
    print('stopping.....')

'''
def print_(b):
    print(b)
    
rospy.init_node('rate_test')
rate = rospy.Rate(10)
a0 = time.time()
while True:
    print("test")
    a = time.time()
    print(a - a0)
    rate.sleep()
    
    while True:
        b = time.time()
        print(b - a0)
        # rate.sleep()
        # break
'''
          
    
        
'''
rospy.init_node('rate_test')
rate = rospy.Rate(1/2)
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
    
    if rospy.is_shutdown():
        print("I am here")
        break
'''    