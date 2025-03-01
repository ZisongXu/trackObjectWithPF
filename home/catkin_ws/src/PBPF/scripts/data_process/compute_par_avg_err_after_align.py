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
from pyquaternion import Quaternion
from quaternion_averaging import weightedAverageQuaternions

def ADDMatrixBtTwoObjects(obj_name, pos1, ori1, pos2, ori2):
    center_T_points_pose_4_4_list = getCenterTPointsList(obj_name)
    if obj_name == "Ketchup":
        if pos2[2] > 0.85:
            pos_ = [0,0,0]
            ori_ = copy.deepcopy(ori2)
            m_3_3 = np.array(p.getMatrixFromQuaternion(ori_)).reshape(3, 3)
            m_3_4 = np.c_[m_3_3, pos_]  # Add position to create 3x4 matrix
            m_4_4 = np.r_[m_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
            rotation_matrix = [[-1, 0, 0, 0],
                               [ 0,-1, 0, 0],
                               [ 0, 0, 1, 0],
                               [ 0, 0, 0, 1]]
            new = np.dot(m_4_4, rotation_matrix)
            new_ori = transformations.quaternion_from_matrix(new)
            ori2 = new_ori
            pos2[2] = pos2[2] - 0.145
            
    # pos1: [0.4685696586646535, -0.2038274908013323, 0.7970063996498579]
    # pos2: [0.3304728561443379, -0.2181392347083696, 0.7753211406030377]
    # ori1: [-0.3066626133656215, 0.5752608330760987, 0.4145899811624447, 0.6349394955522012]
    # ori2: [-0.4857279655585651, 0.5239397674606893, 0.4965236874302953, 0.4929702743251346]

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
    if object_name != "soup" or object_name != "soup2" :
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

def _compute_estimate_pos_of_object_and_find_which_particle_is_closed(particle_cloud_pos_list, particle_cloud_ori_list):
    ## compute mean
    w = 1.0/float(particle_num)
    esti_objs_cloud = []
    # remenber after resampling weight of each particle is the same
    x_set = 0
    y_set = 0
    z_set = 0
    w_set = 0
    quaternions = []
    qws = []

    # for index, particle in enumerate(particle_cloud_pos_list):
    for index in range(int(particle_num)):
        x_set = x_set + particle_cloud_pos_list[index][0] * w
        y_set = y_set + particle_cloud_pos_list[index][1] * w
        z_set = z_set + particle_cloud_pos_list[index][2] * w
        q = quaternion_correction(particle_cloud_ori_list[index])

        qws.append(w)
        quaternions.append([q[0], q[1], q[2], q[3]])
        w_set = w_set + w
    q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
    ###################################
    esti_obj_pos_x = x_set/w_set
    esti_obj_pos_y = y_set/w_set
    esti_obj_pos_z = z_set/w_set
    esti_obj_pos = [esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z]
    esti_obj_ori_x = q[0]
    esti_obj_ori_y = q[1]
    esti_obj_ori_z = q[2]
    esti_obj_ori_w = q[3]
    esti_obj_ori = [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]
    ###################################
    est_obj_pose = [esti_obj_pos, esti_obj_ori]
    err_distance_list = []
    ## find the closest particle to the mean, and record
    for index in range(int(particle_num)):
        par_pos = [particle_cloud_pos_list[index][0], particle_cloud_pos_list[index][1], particle_cloud_pos_list[index][2]]
        par_ori = quaternion_correction(particle_cloud_ori_list[index])
        err_distance = ADDMatrixBtTwoObjects(object_name, par_pos, par_ori, esti_obj_pos, esti_obj_ori)
        err_distance_list.append(err_distance)
    ## find the min distance
    min_value = min(err_distance_list)
    min_index = err_distance_list.index(min_value)
    min_par_pose = [[particle_cloud_pos_list[min_index][0], particle_cloud_pos_list[min_index][1], particle_cloud_pos_list[min_index][2]], quaternion_correction(particle_cloud_ori_list[min_index])]
    return min_par_pose


# make sure all quaternions all between -pi and +pi
def quaternion_correction(quaternion): # x,y,z,w
    new_quat = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]) # w,x,y,z
    cos_theta_over_2 = new_quat.w
    sin_theta_over_2 = math.sqrt(new_quat.x ** 2 + new_quat.y ** 2 + new_quat.z ** 2)
    theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
    theta = theta_over_2 * 2.0
    while theta >= math.pi:
        theta = theta - 2.0*math.pi
    while theta <= -math.pi:
        theta = theta + 2.0*math.pi
    new_quaternion = [math.sin(theta/2.0)*(new_quat.x/sin_theta_over_2), math.sin(theta/2.0)*(new_quat.y/sin_theta_over_2), math.sin(theta/2.0)*(new_quat.z/sin_theta_over_2), math.cos(theta/2.0)]
    #if theta >= math.pi or theta <= -math.pi:
    #    new_quaternion = [-quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
    #    return new_quaternion
    #return quaternion # x,y,z,w
    return new_quaternion


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
run_alg_flag = sys.argv[6] # "obse" "PBPF" "GT"
ang_and_pos = sys.argv[7] # pos/ang/ADD/ADDS
runVersion = sys.argv[8] # "PBPF_RGBD" "PBPF_RGB" "PBPF_D"
MASS_marker = sys.argv[9] # obj_name
FRICTION_marker = sys.argv[10] # obj_name
MF_FLAG = sys.argv[11] # obj_name

file_path_par = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/results/particles/"+MF_FLAG+"/")
file_path_GT = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/results/"+MF_FLAG+"/")
# Time_aligned_5_scene1_rosbag1_repeat0_time_PBPF_pose_PBPF_RGBD_0_cracker.csv
# Time_aligned_70_scene1_rosbag1_repeat0_time_PBPF_pose_PBPF_RGBD_69_Mayo.csv
file_name_list = []
for par_index in range(int(particle_num)):
    par_index_name = par_index
    # Time_aligned_50_scene1_rosbag1_repeat9_time_PBPF_pose_PBPF_RGBD_mC_fA_49_Parmesan
    file_name = "Time_aligned_"+str(particle_num)+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+update_style_flag+'_'+run_alg_flag+'_pose_'+runVersion+'_'+MASS_marker+'_'+FRICTION_marker+'_'+str(par_index_name)+'_'+object_name+'.csv'
    file_name_list.append(file_name)

# Time_aligned_70_scene1_rosbag1_repeat0_Mayo_time_GT_pose_PBPF_D.csv
# GT_file_name = "Time_aligned_"+str(particle_num)+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+object_name+'_'+update_style_flag+'_GT_pose_'+runVersion+'.csv'
GT_file_name = "Time_aligned_"+str(particle_num)+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+object_name+'_'+update_style_flag+'_GT_pose_'+runVersion+'_'+MASS_marker+'_'+FRICTION_marker+'.csv'

# load data
columns_names = ['step','time','pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w','alg','obj','scene','particle_num','ray_type','obj_name','mass','friction']
data_list = []
for par_index in range(int(particle_num)):
    data = pd.read_csv(file_path_par+file_name_list[par_index], names=columns_names, header=None)
    data_list.append(data)
data_GT = pd.read_csv(file_path_GT+GT_file_name, names=columns_names, header=None)

# simple data
columns_of_interest = ['pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w']
pos_ori_data_list = []
for par_index in range(int(particle_num)):
    pos_ori_data = data_list[par_index][columns_of_interest]
    pos_ori_data_list.append(pos_ori_data)
pos_ori_data_GT = data_GT[columns_of_interest]

num_rows_data = pos_ori_data_list[0].shape[0]
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
pos_data_list = []
for par_index in range(int(particle_num)):
    pos_data = pos_ori_data_list[par_index][pos_columns]
    pos_data_list.append(pos_data)
pos_data_GT = pos_ori_data_GT[pos_columns] 
pos_combined_list = []
for par_index in range(int(particle_num)):
    pos_combined = [(row1, row2) for row1, row2 in zip(pos_data_list[par_index].values.tolist(), pos_data_GT.values.tolist())]
    pos_combined_list.append(pos_combined)

ori_columns = ['ori_x', 'ori_y', 'ori_z', 'ori_w']
ori_data_list = []
for par_index in range(int(particle_num)):
    ori_data = pos_ori_data_list[par_index][ori_columns]
    ori_data_list.append(ori_data)
ori_data_GT = pos_ori_data_GT[ori_columns] 
ori_combined_list = []
for par_index in range(int(particle_num)):
    ori_combined = [(row1, row2) for row1, row2 in zip(ori_data_list[par_index].values.tolist(), ori_data_GT.values.tolist())]
    ori_combined_list.append(ori_combined)

existing_columns = ['step','time','alg','obj','scene','particle_num','ray_type','obj_name','mass','friction']
new_err_data = data_list[0][existing_columns].copy()
new_err_data['alg'] = runVersion+'_par_avg'
new_err_data[ang_and_pos] = pd.Series(dtype='float64')


min_par_pose_list = []
# try to compute mean
for row_index in range(num_rows_data):
    alg_pos_combined_list = []
    alg_ori_combined_list = []
    for par_index in range(int(particle_num)):
        pos_compare = pos_combined_list[par_index][row_index][0] # pos1: [0.4685696586646535, -0.2038274908013323, 0.7970063996498579]
        ori_compare = ori_combined_list[par_index][row_index][0] # ori1: [-0.3066626133656215, 0.5752608330760987, 0.4145899811624447, 0.6349394955522012]
        alg_pos_combined_list.append(pos_compare)
        alg_ori_combined_list.append(ori_compare)
    min_par_pose = _compute_estimate_pos_of_object_and_find_which_particle_is_closed(alg_pos_combined_list, alg_ori_combined_list)
    print(task_flag, rosbag_flag, object_name, ang_and_pos,". Finish compute the ",row_index, " row min particle!")
    min_par_pose_list.append(min_par_pose)

# based_on_time_70_scene1_time_cracker_ADD
file_name_error = 'based_on_time_'+str(particle_num)+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_'+update_style_flag+'_'+object_name+'_'+ang_and_pos+'_par_avg.csv'

for row_index in range(num_rows_data):
    err_distance_list = []
    pos_par = min_par_pose_list[row_index][0]
    ori_par = min_par_pose_list[row_index][1]
    pos__GT = pos_combined_list[0][row_index][1]
    ori__GT = ori_combined_list[0][row_index][1]
    err_distance = ADDMatrixBtTwoObjects(object_name, pos_par, ori_par, pos__GT, ori__GT)
    new_err_data.loc[row_index, ang_and_pos] = err_distance
    print("Compute par avg "+ang_and_pos+" error: "+file_name_error+" processing... ", row_index, err_distance)


save_file_path_par = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/results/particles/"+MF_FLAG+"/")

new_err_data.to_csv(save_file_path_par+file_name_error,index=0,header=0,mode='a')
print("Done")  