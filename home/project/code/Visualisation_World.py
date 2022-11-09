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

#Class of initialize the real world model
class Visualisation_World():
    def __init__(self, object_num=0, rob_num=1, other_obj_num=0, visualisation_all=True):
        self.object_num = object_num
        self.rob_num = rob_num
        self.other_obj_num = other_obj_num
        self.visualisation_all = visualisation_all
        self.p_visualisation = 0
        self.create_scene = Create_Scene(object_num, rob_num, other_obj_num)
        self.ros_listener = Ros_Listener(True, "cracker")
        
        
    def initialize_visual_world_pybullet_env(self, task_flag):
        objects_name_list = ["cracker", "soup"]
        pw_T_rob_sim_pose_list, listener_tf = self.create_scene.initialize_robot()
        pw_T_target_obj_obse_pose_lsit, pw_T_target_obj_opti_pose_lsit, pw_T_other_obj_opti_pose_list = self.create_scene.initialize_object()
        
        if self.visualisation_all == True:
            p_visualisation = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT, GUI_SERVER
        elif self.visualisation_all == False:
            p_visualisation = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT, GUI_SERVER
        self.p_visualisation = p_visualisation
        p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_visualisation.setGravity(0, 0, -9.81)
        p_visualisation.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85, cameraTargetPosition=[0.3,0.1,0.2])        
        plane_id = p_visualisation.loadURDF("plane.urdf")
        
        for obj_index in range(self.object_num):
            obse_obj_name = pw_T_target_obj_obse_pose_lsit[obj_index].obj_name
            obse_obj_pos = pw_T_target_obj_obse_pose_lsit[obj_index].pos
            obse_obj_ori = pw_T_target_obj_obse_pose_lsit[obj_index].ori
            obse_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+obse_obj_name+"/"+obse_obj_name+"_obse_obj_with_visual_hor.urdf"),
                                                      obse_obj_pos,
                                                      obse_obj_ori)
            pw_T_target_obj_obse_pose_lsit[obj_index].obj_id = obse_object_id
            opti_obj_name = pw_T_target_obj_opti_pose_lsit[obj_index].obj_name
            opti_obj_pos = pw_T_target_obj_opti_pose_lsit[obj_index].pos
            opti_obj_ori = pw_T_target_obj_opti_pose_lsit[obj_index].ori
            opti_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+opti_obj_name+"/"+opti_obj_name+"_real_obj_with_visual_hor.urdf"),
                                                      opti_obj_pos,
                                                      opti_obj_ori)
            pw_T_target_obj_opti_pose_lsit[obj_index].obj_id = opti_object_id
           
        for obj_index in range(self.other_obj_num):
            other_obj_name = pw_T_other_obj_opti_pose_list[obj_index].obj_name
            other_obj_pos = pw_T_other_obj_opti_pose_list[obj_index].pos
            other_obj_ori = pw_T_other_obj_opti_pose_list[obj_index].ori
            optitrack_base_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+other_obj_name+"/base_of_cracker.urdf"),
                                                         other_obj_pos,
                                                         other_obj_ori)
            pw_T_other_obj_opti_pose_list[obj_index].obj_id = optitrack_base_id
            
        for rob_index in range(self.rob_num):
            pw_T_rob_sim_pose_list
            # rob_name = pw_T_rob_sim_pose_list[obj_index].obj_name
            rob_pos = pw_T_rob_sim_pose_list[rob_index].pos
            rob_ori = pw_T_rob_sim_pose_list[rob_index].ori
            real_robot_id = p_visualisation.loadURDF(os.path.expanduser("~/project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                                     rob_pos,
                                                     rob_ori,
                                                     useFixedBase=1)
            joint_pos = self.ros_listener.current_joint_values
            self.set_real_robot_JointPosition(p_visualisation, real_robot_id, joint_pos)
            pw_T_rob_sim_pose_list[rob_index].obj_id = real_robot_id
            pw_T_rob_sim_pose_list[rob_index].joints = joint_pos
        for i in range(240):
            p_visualisation.stepSimulation()
        return listener_tf, p_visualisation, pw_T_rob_sim_pose_list, pw_T_target_obj_obse_pose_lsit, pw_T_target_obj_opti_pose_lsit, pw_T_other_obj_opti_pose_list
    
    
    def initial_contact_object(self, objects_pose_list):
        contact_obj_id_list = []
        for i in range(self.object_num):
            object_name = objects_pose_list[i].obj_name
            object_pos = objects_pose_list[i].pos
            object_ori = objects_pose_list[i].ori
            contact_obj_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+object_name+"/"+object_name+"_contact_test_obj_hor.urdf"),
                                                           object_pos,
                                                           object_ori)
            self.p_visualisation.changeDynamics(contact_obj_id, -1, mass=0.380, lateralFriction=0.5)
            contact_obj_id_list.append(contact_obj_id)
        return contact_obj_id_list
    
    
    def set_real_robot_JointPosition(self, pybullet_simulation_env, robot, position):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                pybullet_simulation_env.setJointMotorControl2(robot,
                                                              joint_index+2,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])
            else:
                pybullet_simulation_env.setJointMotorControl2(robot,
                                                              joint_index,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        