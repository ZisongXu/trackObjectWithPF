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
from Center_T_Point_for_Ray import Center_T_Point_for_Ray
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


# add position into transformation matrix
def rotation_4_4_to_transformation_4_4(rotation_4_4, pos):
    rotation_4_4[0][3] = pos[0]
    rotation_4_4[1][3] = pos[1]
    rotation_4_4[2][3] = pos[2]
    return rotation_4_4
        
# compute the transformation matrix represent that the pose of object in the robot world
def compute_transformation_matrix(a_pos, a_ori, b_pos, b_ori):
    ow_T_a_3_3 = transformations.quaternion_matrix(a_ori)
    ow_T_a_4_4 = rotation_4_4_to_transformation_4_4(ow_T_a_3_3,a_pos)
    ow_T_b_3_3 = transformations.quaternion_matrix(b_ori)
    ow_T_b_4_4 = rotation_4_4_to_transformation_4_4(ow_T_b_3_3,b_pos)
    a_T_ow_4_4 = np.linalg.inv(ow_T_a_4_4)
    a_T_b_4_4 = np.dot(a_T_ow_4_4,ow_T_b_4_4)
    return a_T_b_4_4 

# robot arm move
def set_real_robot_JointPosition(pybullet_env, robot_id, joint_states):
    num_joints = 9
    for joint_index in range(num_joints):
        if joint_index == 7 or joint_index == 8:
            pybullet_env.setJointMotorControl2(robot_id,
                                                joint_index+2,
                                                pybullet_env.POSITION_CONTROL,
                                                targetPosition=joint_states[joint_index])
        else:
            pybullet_env.setJointMotorControl2(robot_id,
                                                joint_index,
                                                pybullet_env.POSITION_CONTROL,
                                                targetPosition=joint_states[joint_index])

# compute debug line point set
def cracker_debug_line_point_set(cracker_shape_size, pw_T_parC_4_4, vector_list):
    point_list = []
    point_pos_list = []
    vector_length = len(vector_list)
    for index in range(vector_length):
        parC_T_p_x_new = vector_list[index][0] * cracker_shape_size[0]/2
        parC_T_p_y_new = vector_list[index][1] * cracker_shape_size[1]/2
        parC_T_p_z_new = vector_list[index][2] * cracker_shape_size[2]/2
        parC_T_p_pos = [parC_T_p_x_new, parC_T_p_y_new, parC_T_p_z_new]
        parC_T_p_ori = [0, 0, 0, 1] # x, y, z, w
        parC_T_p_3_3 = transformations.quaternion_matrix(parC_T_p_ori)
        parC_T_p_4_4 = rotation_4_4_to_transformation_4_4(parC_T_p_3_3, parC_T_p_pos)
        pw_T_p_4_4 = np.dot(pw_T_parC_4_4, parC_T_p_4_4)
        pw_T_p_pos = [pw_T_p_4_4[0][3], pw_T_p_4_4[1][3], pw_T_p_4_4[2][3]]
        pw_T_p_ori = transformations.quaternion_from_matrix(pw_T_p_4_4)
        pw_T_p_pose = Center_T_Point_for_Ray(pw_T_p_pos, pw_T_p_ori, parC_T_p_4_4, index)
        point_list.append(pw_T_p_pose)
        point_pos_list.append(pw_T_p_pos)
    return point_pos_list

# add debug line
def add_debug_line(vector_length, a_start, b_end_list, color):
    for index in range(vector_length):
        p_visualisation.addUserDebugLine(a_start, b_end_list[index], lineColorRGB=color, lineWidth=1)

opt_T_she_pos = [0.9005279541015625, 0.36096227169036865, 0.8243836760520935]
opt_T_she_ori = [0, 0, 0, 1]

opt_T_rob_pos = [0.1412336677312851, 0.02477111667394638, 0.5779450535774231]
opt_T_rob_ori = [0.7075536251068115, -0.00017867401766125113, 0.0019650713074952364, -0.7066569924354553]

# opt_T_she_pos = list(opt_T_she_pos)
# opt_T_she_ori = list(opt_T_she_ori)
# opt_T_she_3_3 = transformations.quaternion_matrix(opt_T_she_ori)
# opt_T_she_4_4 = rotation_4_4_to_transformation_4_4(opt_T_she_3_3, opt_T_she_pos)

rob_T_she_4_4 = compute_transformation_matrix(opt_T_rob_pos, opt_T_rob_ori, opt_T_she_pos, opt_T_she_ori)

print(rob_T_she_4_4)

# pw_T_obst_opti_pos_small = [0.852134144216095, 0.14043691336334274, 0.10014295215002848]
# pw_T_obst_opti_ori_small = [0.00356749, -0.00269526, 0.28837681, 0.95750657]
# pw_T_obst_opti_pos_big = [0.75889274, -0.24494845, 0.33818097+0.02]
# pw_T_obst_opti_ori_big = [0, 0, 0, 1]
# track_fk_obst_big_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/shelves.urdf"),
#                                                 pw_T_obst_opti_pos_big,
#                                                 pw_T_obst_opti_ori_big)

rob_pos = [0, 0, 0.02]
rob_ori = [0, 0, 0, 1]
real_robot_id = p_visualisation.loadURDF(os.path.expanduser("~/project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                         rob_pos,
                                         rob_ori,
                                         useFixedBase=1)

joint_states = [-0.416393778717333,
                 0.8254077830686731,
                -0.07092072120488697,
                -2.1336947324364215,
                 1.0840709206509551,
                 1.4970466512368048,
                 0.9383130510987506,
                 0,
                 0]
set_real_robot_JointPosition(p_visualisation, real_robot_id, joint_states)

# cracker_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+gazebo_contain+obj_obse_name+"/"+gazebo_contain+obj_obse_name+"_par_no_visual_hor.urdf"),
cracker1_pos = [0.25, -0.05, 0.085]
cracker1_ori = [0, 1, 0, 1] # (0.690527050730201, 0.15228119684895902, 0.15227363064756175, 0.6905038527810753)
pw_T_cracker1_3_3 = transformations.quaternion_matrix(cracker1_ori)
pw_T_cracker1_4_4 = rotation_4_4_to_transformation_4_4(pw_T_cracker1_3_3, cracker1_pos)
cracker_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_par_no_visual_hor.urdf"),
                                                         cracker1_pos,
                                                         cracker1_ori)

cracker2_pos = [0.37724945986529196, 0.20917670309857364, 0.085]
cracker2_ori = [-0.2130909190791487, -0.6742164207435634, -0.2130985610453765, 0.6742503107052543] # (0.690527050730201, 0.15228119684895902, 0.15227363064756175, 0.6905038527810753)
pw_T_cracker2_3_3 = transformations.quaternion_matrix(cracker2_ori)
pw_T_cracker2_4_4 = rotation_4_4_to_transformation_4_4(pw_T_cracker2_3_3, cracker2_pos)
cracker2_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_par_no_visual_hor.urdf"),
                                                         cracker2_pos,
                                                         cracker2_ori)


obstacle_pos = [0.7390681677092075, -0.05127685137888386, 0.299/2]
obstacle_ori = [0.690527050730201, 0.15228119684895902, 0.15227363064756175, 0.6905038527810753]
pw_T_obstacle_3_3 = transformations.quaternion_matrix(obstacle_ori)
pw_T_obstacle_4_4 = rotation_4_4_to_transformation_4_4(pw_T_obstacle_3_3, obstacle_pos)
obstacle_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_obstacle_big.urdf"),
                                                          obstacle_pos,
                                                          obstacle_ori)


camera_pos = [1, 0.0, 0.3]
camera_ori = [0, 0, 0, 1]
camera_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/camera_model.urdf"),
                                                         camera_pos,
                                                         camera_ori,
                                                         useFixedBase=1)



camera_T_lens_pos = [0.0, 0.025, 0.0]
camera_T_lens_ori = [0, 0, 0, 1]
camera_T_lens_3_3 = transformations.quaternion_matrix(camera_T_lens_ori)
camera_T_lens_4_4 = rotation_4_4_to_transformation_4_4(camera_T_lens_3_3, camera_T_lens_pos)
camera_pos = [1, 0.0, 0.3]
camera_ori = [0, 0, 0, 1]
pw_T_camera_3_3 = transformations.quaternion_matrix(camera_ori)
pw_T_camera_4_4 = rotation_4_4_to_transformation_4_4(pw_T_camera_3_3, camera_pos)
pw_T_lens_4_4 = np.dot(pw_T_camera_4_4, camera_T_lens_4_4)
pw_T_lens_pos = [pw_T_lens_4_4[0][3], pw_T_lens_4_4[1][3], pw_T_lens_4_4[2][3]]
pw_T_lens_ori = transformations.quaternion_from_matrix(pw_T_lens_4_4)

vector_list = [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
               [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1],
               [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
               [1,0.5,0.5], [1,0.5,-0.5], [1,-0.5,0.5], [1,-0.5,-0.5],
               [-1,0.5,0.5], [-1,0.5,-0.5], [-1,-0.5,0.5], [-1,-0.5,-0.5],
               [0.5,1,0.5], [0.5,1,-0.5], [-0.5,1,0.5], [-0.5,1,-0.5],
               [0.5,-1,0.5], [0.5,-1,-0.5], [-0.5,-1,0.5], [-0.5,-1,-0.5],
               [0.5,0.5,1], [0.5,-0.5,1], [-0.5,0.5,1], [-0.5,-0.5,1],
               [0.5,0.5,-1], [0.5,-0.5,-1], [-0.5,0.5,-1], [-0.5,-0.5,-1]]
vector_length = len(vector_list)
x_w = 0.16 
y_l = 0.21343700408935547 
z_h = 0.061
cracker_shape_size = [x_w, y_l, z_h]

pw_T_parC1_4_4 = pw_T_cracker1_4_4
point_pos_list1 = cracker_debug_line_point_set(cracker_shape_size, pw_T_parC1_4_4, vector_list)
color1 = [1, 0.30196078, 0.25098039]
add_debug_line(vector_length, pw_T_lens_pos, point_pos_list1, color1)

pw_T_parC2_4_4 = pw_T_cracker2_4_4
point_pos_list2 = cracker_debug_line_point_set(cracker_shape_size, pw_T_parC2_4_4, vector_list)
# color2 = [0.56862745, 0.62352941, 0.83921569] # purple
color2 = [0.02352941, 0.6627451, 0.61568627]
add_debug_line(vector_length, pw_T_lens_pos, point_pos_list2, color2)


while True:
# for i in range(240):
    a = 1
    p_visualisation.stepSimulation()
    time.sleep(1./240.)
    obstacle_info = p_visualisation.getBasePositionAndOrientation(cracker2_id)
    print("obstacle_info:")
    print(obstacle_info)
    # time.sleep(1)
    
# for i in range(240):
#     p_visualisation.stepSimulation()