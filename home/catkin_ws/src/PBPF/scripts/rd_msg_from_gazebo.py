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
from PBPF.msg import particle_state, particles_states
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
    

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    Listener = Ros_Listener()
    time.sleep(0.5)
    for i in range(10):
        print(i)
        print(Listener.current_joint_values)
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
