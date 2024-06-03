#!/usr/bin/python3
import pybullet as p

from rosbag.bag import Bag
import roslib;   #roslib.load_manifest(PKG)
import rosbag
import rospy
import tf2_msgs.msg
from tf2_msgs.msg import TFMessage

import cv2
import numpy as np
import pandas as pd
import argparse
import os
import time
import sys
import yaml
import copy
import math

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import tf
import tf.transformations as transformations
import tf2_ros
from tf2_geometry_msgs import tf2_geometry_msgs


def ADDMatrixBtTwoObjects(obj_name, pos1, ori1, pos2, ori2, task_flag):
    center_T_points_pose_4_4_list = getCenterTPointsList(obj_name)
    # if obj_name == "Ketchup":
    #     if pos2[2] > 0.85:
    #         pos_ = [0,0,0]
    #         ori_ = copy.deepcopy(ori2)
    #         m_3_3 = np.array(p.getMatrixFromQuaternion(ori_)).reshape(3, 3)
    #         m_3_4 = np.c_[m_3_3, pos_]  # Add position to create 3x4 matrix
    #         m_4_4 = np.r_[m_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
    #         rotation_matrix = [[-1, 0, 0, 0],
    #                            [ 0,-1, 0, 0],
    #                            [ 0, 0, 1, 0],
    #                            [ 0, 0, 0, 1]]
    #         new = np.dot(m_4_4, rotation_matrix)
    #         new_ori = transformations.quaternion_from_matrix(new)
    #         ori2 = new_ori
    #         pos2[2] = pos2[2] - 0.145
    #     if ori2[3] < 0:
    #         ori2[0] = -ori2[0]
    #         ori2[1] = -ori2[1]
    #         ori2[2] = -ori2[2]
    #         ori2[3] = -ori2[3]
    # if obj_name == "soup" and task_flag == "scene2" and run_alg_flag == "PBPF":
    #     if ori1[3] < 0:
    #         ori1[0] = -ori1[0]
    #         ori1[1] = -ori1[1]
    #         ori1[2] = -ori1[2]
    #         ori1[3] = -ori1[3]
    # if obj_name == "Parmesan" and task_flag == "scene2":
    #     if ori2[3] > 0:
    #         ori2[0] = -ori2[0]
    #         ori2[1] = -ori2[1]
    #         ori2[2] = -ori2[2]
    #         ori2[3] = -ori2[3]
    # if obj_name == "Mustard" and task_flag == "scene1":
    #     if ori2[3] > 0:
    #         ori2[0] = -ori2[0]
    #         ori2[1] = -ori2[1]
    #         ori2[2] = -ori2[2]
    #         ori2[3] = -ori2[3]
    # if obj_name == "SaladDressing" and task_flag == "scene1":
    #     if ori2[3] < 0:
    #         ori2[0] = -ori2[0]
    #         ori2[1] = -ori2[1]
    #         ori2[2] = -ori2[2]
    #         ori2[3] = -ori2[3]

    # mark
    # if obj_name == "soup":
    #     pw_T_parC_ang = list(p.getEulerFromQuaternion(pw_T_parC_ori))
    #     pw_T_parC_ang[0] = pw_T_parC_ang[0] + 1.5707963
    #     pw_T_parC_ori = p.getQuaternionFromEuler(pw_T_parC_ang)

    pw_T_points_pose_4_4_list_1 = getPwTPointsList(center_T_points_pose_4_4_list, pos1, ori1)
    pw_T_points_pose_4_4_list_2 = getPwTPointsList(center_T_points_pose_4_4_list, pos2, ori2)
    if ang_and_pos == "ADD":
        err_distance = computeADD(pw_T_points_pose_4_4_list_1, pw_T_points_pose_4_4_list_2)
    elif ang_and_pos == "ADDS":
        err_distance = computeADDS(pw_T_points_pose_4_4_list_1, pw_T_points_pose_4_4_list_2)
    return err_distance

def getCenterTPointsList(object_name):
    center_T_points_pose_4_4_list = []
    # if object_name == "cracker" or object_name == "gelatin":
    if object_name != "soup":
        if object_name == "cracker":
            x_w = 0.159
            y_l = 0.21243700408935547
            z_h = 0.06
        elif object_name == "Ketchup":
            x_w = 0.145
            y_l = 0.042
            z_h = 0.061
        elif object_name == "Milk":
            x_w = 0.179934
            y_l = 0.0613
            z_h = 0.0613
        elif object_name == "Mustard":
            x_w = 0.14
            y_l = 0.038
            z_h = 0.055
        elif object_name == "Mayo":
            x_w = 0.1377716
            y_l = 0.0310130
            z_h = 0.054478
        elif object_name == "Parmesan":
            x_w = 0.0929022
            y_l = 0.0592842
            z_h = 0.0592842
        elif object_name == "SaladDressing":
            x_w = 0.1375274
            y_l = 0.036266
            z_h = 0.052722
        else:
            x_w = 0.0851
            y_l = 0.0737
            z_h = 0.0279
        vector_list = [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1], [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1], [1,0.5,0.5], [1,0.5,-0.5], [1,-0.5,0.5], [1,-0.5,-0.5], [-1,0.5,0.5], [-1,0.5,-0.5], [-1,-0.5,0.5], [-1,-0.5,-0.5], [0.5,1,0.5], [0.5,1,-0.5], [-0.5,1,0.5], [-0.5,1,-0.5], [0.5,-1,0.5], [0.5,-1,-0.5], [-0.5,-1,0.5], [-0.5,-1,-0.5], [0.5,0.5,1], [0.5,-0.5,1], [-0.5,0.5,1], [-0.5,-0.5,1], [0.5,0.5,-1], [0.5,-0.5,-1], [-0.5,0.5,-1], [-0.5,-0.5,-1]]
    else:
        x_w = 0.032829689025878906
        y_l = 0.032829689025878906
        z_h = 0.099
        r = math.sqrt(2)
        vector_list = [[0,0,1], [0,0,-1],
                       [r,0,1], [0,r,1], [-r,0,1], [0,-r,1], [r,r,1], [r,-r,1], [-r,r,1], [-r,-r,1],
                       [r,0,0.5], [0,r,0.5], [-r,0,0.5], [0,-r,0.5], [r,r,0.5], [r,-r,0.5], [-r,r,0.5], [-r,-r,0.5],
                       [r,0,0], [0,r,0], [-r,0,0], [0,-r,0], [r,r,0], [r,-r,0], [-r,r,0], [-r,-r,0],
                       [r,0,-0.5], [0,r,-0.5], [-r,0,-0.5], [0,-r,-0.5], [r,r,-0.5], [r,-r,-0.5], [-r,r,-0.5], [-r,-r,-0.5],
                       [r,0,-1], [0,r,-1], [-r,0,-1], [0,-r,-1], [r,r,-1], [r,-r,-1], [-r,r,-1], [-r,-r,-1]]
    for index in range(len(vector_list)):
        center_T_p_x_new = vector_list[index][0] * x_w/2
        center_T_p_y_new = vector_list[index][1] * y_l/2
        center_T_p_z_new = vector_list[index][2] * z_h/2
        center_T_p_pos = [center_T_p_x_new, center_T_p_y_new, center_T_p_z_new]
        center_T_p_ori = [0, 0, 0, 1] # x, y, z, w
        # center_T_p_3_3 = transformations.quaternion_matrix(center_T_p_ori)
        # center_T_p_4_4 = rotation_4_4_to_transformation_4_4(center_T_p_3_3, center_T_p_pos)
        center_T_p_3_3 = np.array(p.getMatrixFromQuaternion(center_T_p_ori)).reshape(3, 3)
        center_T_p_3_4 = np.c_[center_T_p_3_3, center_T_p_pos]  # Add position to create 3x4 matrix
        center_T_p_4_4 = np.r_[center_T_p_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
        
        center_T_points_pose_4_4_list.append(center_T_p_4_4)
    return center_T_points_pose_4_4_list

def getPwTPointsList(center_T_points_pose_4_4_list, pos, ori):
    pw_T_points_pose_4_4_list = []
    # pw_T_center_ori_3_3 = transformations.quaternion_matrix(ori)
    # pw_T_center_ori_4_4 = rotation_4_4_to_transformation_4_4(pw_T_center_ori_3_3, pos)
    pw_T_center_ori_3_3 = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    pw_T_center_ori_3_4 = np.c_[pw_T_center_ori_3_3, pos]  # Add position to create 3x4 matrix
    pw_T_center_ori_4_4 = np.r_[pw_T_center_ori_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
    # mark
    for index in range(len(center_T_points_pose_4_4_list)):
        center_T_p_4_4 = copy.deepcopy(center_T_points_pose_4_4_list[index])
        pw_T_p_4_4 = np.dot(pw_T_center_ori_4_4, center_T_p_4_4)
        pw_T_points_pose_4_4_list.append(pw_T_p_4_4)
    return pw_T_points_pose_4_4_list

def computeADD(pw_T_points_pose_4_4_list_1, pw_T_points_pose_4_4_list_2):
    dis_sum = 0
    points_num = len(pw_T_points_pose_4_4_list_1)
    for index in range(points_num):
        pw_T_p_pos1 = [pw_T_points_pose_4_4_list_1[index][0][3], pw_T_points_pose_4_4_list_1[index][1][3], pw_T_points_pose_4_4_list_1[index][2][3]]
        pw_T_p_pos2 = [pw_T_points_pose_4_4_list_2[index][0][3], pw_T_points_pose_4_4_list_2[index][1][3], pw_T_points_pose_4_4_list_2[index][2][3]]
        distance = compute_pos_err_bt_2_points(pw_T_p_pos1, pw_T_p_pos2)
        dis_sum = dis_sum + distance
    average_distance = 1.0 * dis_sum / points_num
    return average_distance

def computeADDS(pw_T_points_pose_4_4_list_1, pw_T_points_pose_4_4_list_2):
    dis_sum = 0
    points_num = len(pw_T_points_pose_4_4_list_1)
    for index1 in range(points_num):
        pw_T_p_pos1 = [pw_T_points_pose_4_4_list_1[index1][0][3], pw_T_points_pose_4_4_list_1[index1][1][3], pw_T_points_pose_4_4_list_1[index1][2][3]]
        single_point_distance_list = []
        for index2 in range(points_num):
            pw_T_p_pos2 = [pw_T_points_pose_4_4_list_2[index2][0][3], pw_T_points_pose_4_4_list_2[index2][1][3], pw_T_points_pose_4_4_list_2[index2][2][3]]
            distance = compute_pos_err_bt_2_points(pw_T_p_pos1, pw_T_p_pos2)
            single_point_distance_list.append(distance)
        min_distance = min(single_point_distance_list)
        dis_sum = dis_sum + min_distance
    average_distance = 1.0 * dis_sum / points_num
    return average_distance

def compute_pos_err_bt_2_points(pos1, pos2):
    x1=pos1[0]
    y1=pos1[1]
    z1=pos1[2]
    x2=pos2[0]
    y2=pos2[1]
    z2=pos2[2]
    x_d = x1-x2
    y_d = y1-y2
    z_d = z1-z2
    distance = math.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)
    return distance


with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
        parameter_info = yaml.safe_load(file)

gazebo_flag = parameter_info['gazebo_flag']
particle_num = parameter_info['particle_num']
# update mode (pose/time)
update_style_flag = parameter_info['update_style_flag'] # time/pose
# which algorithm to run
run_alg_flag = parameter_info['run_alg_flag'] # PBPF/CVPF
# scene
task_flag = parameter_info['task_flag'] # parameter_info['task_flag']
# rosbag_flag = "1"
err_file = parameter_info['err_file']

particle_num = sys.argv[1]
object_name = sys.argv[2]
task_flag = sys.argv[3] # "scene1"
rosbag_flag = sys.argv[4]
repeat_time = sys.argv[5]
run_alg_flag = sys.argv[6] # ("PBPF" "obse" "FOUD")
ang_and_pos = sys.argv[7] # pos/ang/ADD/ADDS
runVersion = sys.argv[8] # "PBPF_RGBD" "PBPF_RGB" "PBPF_D"

file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/results/")
# Time_aligned_10_scene1_rosbag1_repeat0_cracker_time_GT_pose_PBPF_RGBD.csv
# Time_aligned_10_scene1_rosbag1_repeat0_cracker_time_obse_pose_PBPF_RGBD.csv
# Time_aligned_10_scene1_rosbag1_repeat0_cracker_time_PBPF_pose_PBPF_RGBD.csv
file_name = "Time_aligned_"+str(particle_num)+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+object_name+'_'+update_style_flag+'_'+run_alg_flag+'_pose_'+runVersion+'.csv'
# GT_file_name = "Time_aligned_"+str(particle_num)+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+object_name+'_'+update_style_flag+'_GT_pose_'+runVersion+'.csv'
GT_file_name = "Time_aligned_"+str(particle_num)+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+object_name+'_'+update_style_flag+'_GT_pose_'+runVersion+'.csv'

columns_names = ['step','time','pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w','alg','obj','scene','particle_num','ray_type','obj_name']
data = pd.read_csv(file_path+file_name, names=columns_names, header=None)
data_GT = pd.read_csv(file_path+GT_file_name, names=columns_names, header=None)


columns_of_interest = ['pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w']
pos_ori_data = data[columns_of_interest]
pos_ori_data_GT = data_GT[columns_of_interest]

num_rows_data = pos_ori_data.shape[0]
num_rows_data_GT = pos_ori_data_GT.shape[0]
if num_rows_data != num_rows_data_GT:
    print("")
    print(" ------------------------------------------ ")
    print("|                                          |")
    print("|                                          |")
    print("|                                          |")
    print("|                  Error!                  |")
    print("|                                          |")
    print("|                                          |")
    print("|                                          |")
    print(" ------------------------------------------ ")


pos_columns = ['pos_x', 'pos_y', 'pos_z']
pos_data = pos_ori_data[pos_columns]
pos_data_GT = pos_ori_data_GT[pos_columns] 
pos_combined = [(row1, row2) for row1, row2 in zip(pos_data.values.tolist(), pos_data_GT.values.tolist())]

ori_columns = ['ori_x', 'ori_y', 'ori_z', 'ori_w']
ori_data = pos_ori_data[ori_columns]
ori_data_GT = pos_ori_data_GT[ori_columns] 
ori_combined = [(row1, row2) for row1, row2 in zip(ori_data.values.tolist(), ori_data_GT.values.tolist())]

existing_columns = ['step','time','alg','obj','scene','particle_num','ray_type','obj_name']
new_err_data = data[existing_columns].copy()
new_err_data[ang_and_pos] = pd.Series(dtype='float64')

file_name_error = 'based_on_time_'+str(particle_num)+'_'+task_flag+'_'+update_style_flag+'_'+object_name+'_'+ang_and_pos

for row_index in range(num_rows_data):
    pos_compare = pos_combined[row_index][0]
    pos______GT = pos_combined[row_index][1]
    ori_compare = ori_combined[row_index][0]
    ori______GT = ori_combined[row_index][1]
    err_distance = ADDMatrixBtTwoObjects(object_name, pos_compare, ori_compare, pos______GT, ori______GT, task_flag)
    new_err_data.loc[row_index, ang_and_pos] = err_distance
    print("Compute "+ang_and_pos+" error: "+file_name+" processing... ", row_index)

new_err_data.to_csv(file_path+file_name_error+".csv",index=0,header=0,mode='a')
if run_alg_flag == "FOUD" or run_alg_flag == "obse":
    new_err_data.to_csv(file_path+"particles/"+file_name_error+"_par_min.csv",index=0,header=0,mode='a')
    new_err_data.to_csv(file_path+"particles/"+file_name_error+"_par_avg.csv",index=0,header=0,mode='a')
print("Done")  