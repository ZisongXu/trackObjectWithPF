# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""
# gazebo_flag: false
# object_name_list:
# - soup
# - fish_can
# object_num: 1
# other_obj_num: 0
# oto_name_list:
# - base_of_cracker
# - fish_can
# particle_num: 140
# robot_num: 1
# run_alg_flag: PBPF
# task_flag: '3'
# update_style_flag: time
import ssl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import yaml
import os
import sys
import math
import numpy as np
import pybullet as p
import tf.transformations as transformations

with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
        parameter_info = yaml.safe_load(file)
object_name = parameter_info['object_name_list'][0]
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
run_alg_flag = sys.argv[6] # PBPF
ang_and_pos = sys.argv[7] # pos/ang
runVersion = sys.argv[8] # multiray/ang
MASS_marker = sys.argv[9] # obj_name
FRICTION_marker = sys.argv[10] # obj_name

# 10_scene1_rosbag1_repeat0_cracker_time_GT_pose_PBPF_RGBD
# 10_scene1_rosbag1_repeat0_cracker_time_obse_pose_PBPF_RGBD
# 10_scene1_rosbag1_repeat0_cracker_time_PBPF_pose_PBPF_RGBD
file_name = str(particle_num)+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+object_name+'_'+update_style_flag+'_'+run_alg_flag+'_pose_'+runVersion+'_'+MASS_marker+'_'+FRICTION_marker


flag_pos = True
flag_ang = True
flag_CVPF = True
correct_time_flag = False

loop_flag = 10
prepare_time = 28 * 100
prepare_time = 129 * 100
prepare_time = 265 * 100
prepare_time = 1730 * 100
prepare_time = 55 * 100
rosbag_slowdown_rate = 1

if object_name == "cracker" and rosbag_flag == "1":
    prepare_time = 95 * 100
    rosbag_slowdown_rate = 1
if object_name == "Ketchup" and rosbag_flag == "1":
    prepare_time = 100 * 100
    rosbag_slowdown_rate = 1
if object_name == "Mayo" and rosbag_flag == "1":
    prepare_time = 100 * 100
    rosbag_slowdown_rate = 1
if object_name == "Milk" and rosbag_flag == "1":
    prepare_time = 100 * 100
    rosbag_slowdown_rate = 1
if object_name == "Mustard" and rosbag_flag == "1":
    prepare_time = 115 * 100
    rosbag_slowdown_rate = 1
if object_name == "Parmesan" and rosbag_flag == "1":
    prepare_time = 30 * 100
    rosbag_slowdown_rate = 1
if object_name == "SaladDressing" and rosbag_flag == "1":
    prepare_time = 95 * 100
    rosbag_slowdown_rate = 1
if object_name == "soup" and rosbag_flag == "1":
    prepare_time = 60 * 100
    rosbag_slowdown_rate = 1
if object_name == "soup2" and rosbag_flag == "1":
    prepare_time = 60 * 100
    rosbag_slowdown_rate = 1

# pw_T_cam_pose = np.array([[-0.17022463,  0.22072718, -0.96036612,  1.01227219],
#                           [ 0.98534948,  0.02775525, -0.16827375,  0.09076827],
#                           [-0.01048739, -0.97494059, -0.22221804,  0.93997983],
#                           [ 0.        ,  0.        ,  0.        ,  1.        ]])

# pw_T_cam_pose = np.array([[-0.044931,    0.23043063, -0.97205089,  1.20441085],
#                             [ 0.99824785,  0.04785813, -0.03479683,  0.0336495 ],
#                             [ 0.03850228, -0.97191117, -0.2321772 ,  1.0223675 ],
#                             [ 0.        ,  0.        ,  0.        ,  1.        ]])

# pw_T_cam_pose = np.array([[-8.95095741e-03,  2.17633877e-01, -9.75989434e-01,  1.21345200e+00],
#                             [ 9.99005319e-01,  4.45843495e-02,  7.79732597e-04, -6.73263951e-03],
#                             [ 4.36835502e-02, -9.75011657e-01, -2.17816474e-01,  1.00687529e+00],
#                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
# pw_T_cam_pose = np.array([[ 9.37735911e-02,  2.78147082e-01, -9.55950163e-01,  1.18120405e+00],
#                             [ 9.95593427e-01, -2.57238183e-02,  9.01776779e-02, -8.42059848e-02],
#                             [ 4.91969723e-04, -9.60193983e-01, -2.79333621e-01,  1.04590236e+00],
#                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
# pw_T_cam_pose = np.array([[ 0.04771702,  0.29078387, -0.95559815,  1.15756931],
#                             [ 0.99797884,  0.02631729,  0.05784149,  0.01792461],
#                             [ 0.04196812, -0.95642676, -0.28894036,  1.0088215 ],
#                             [ 0.        ,  0.        ,  0.        ,  1.        ]])    
pw_T_cam_pose = np.array([[ 0.07372993,  0.2208193 , -0.9725239 ,  1.10858728],
                            [ 0.99704642,  0.00470366,  0.07665706,  0.01198535],
                            [ 0.02150178, -0.9753034 , -0.21982029,  1.02230446],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]])        

# prepare_time = 250000

# save file
# save_file_name = tem_name+'_'+'based_on_time_'+str(particle_num)+'_'+object_name+'_'+task_flag+'_'+update_style_flag+'_'+ang_and_pos+'.csv'
# save_file_name = 'based_on_time_'+str(particle_num)+'_'+task_flag+'_'+update_style_flag+'_'+object_name+'_'+ang_and_pos+'.csv'
save_file_name = 'Time_aligned_'+file_name+'.csv'

# save_file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/error_file_diff_par_num/70/1_cracker_scene1/inter_data_"+ang_and_pos+"/")
# save_file_path = os.path.expanduser("~/catkin_ws/src/PBPF/scripts/"+err_file+"/")
save_file_path = os.path.expanduser('~/catkin_ws/src/PBPF/scripts/results/')

def correct_time(datasetcopy):
    print("Enter into the correct time function")
    for time_index in range(len(datasetcopy)):
        time = datasetcopy.loc[datasetcopy.index==time_index,'time']
        if datasetcopy.time[time_index] > 2:
            datasetcopy.loc[datasetcopy.index==time_index,'time'] = datasetcopy.loc[datasetcopy.index==time_index,'time'] - 16
    # print(datasetcopy.time)

def angle_correction(angle):
    if math.pi <= angle <= (3.0 * math.pi):
        angle = angle - 2 * math.pi
    elif -(3.0 * math.pi) <= angle <= -math.pi:
        angle = angle + 2 * math.pi
    angle = abs(angle)
    return angle

# print("Ready to integrate the data of "+ang_and_pos)
dataset = pd.read_csv(save_file_path+file_name+'.csv', header=None)
# dataset.columns=["index","time","error","alg","obj_scene","particle_num","ray_type"]
dataset.columns=['index','time','pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w','alg','obj','scene','particle_num','ray_type', 'obj_name', 'mass', 'friction']

for index, row in dataset.iterrows():
    pos_x = row['pos_x']
    pos_y = row['pos_y']
    pos_z = row['pos_z']
    ori_x = row['ori_x']
    ori_y = row['ori_y']
    ori_z = row['ori_z']
    ori_w = row['ori_w']

    # theta = np.pi
    # rotation_matrix = np.array(
    #     [
    #         [1, 0, 0, 0],
    #         [0, np.cos(theta), -np.sin(theta), 0],
    #         [0, np.sin(theta), np.cos(theta), 0],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # rotation_matrix_inv = np.linalg.inv(rotation_matrix)
    # print("rotation_matrix:")
    # print(rotation_matrix)
    # print("rotation_matrix_inv")
    # print(rotation_matrix_inv)
    cam_T_dfdp_tf_3_3 = np.array(p.getMatrixFromQuaternion([ori_x, ori_y, ori_z, ori_w])).reshape(3, 3)
    cam_T_dfdp_tf_3_4 = np.c_[cam_T_dfdp_tf_3_3, [pos_x, pos_y, pos_z]]  # Add position to create 3x4 matrix
    cam_T_dfdp_tf_4_4 = np.r_[cam_T_dfdp_tf_3_4, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
    pw_T_dfdp_tf_4_4 = np.dot(pw_T_cam_pose, cam_T_dfdp_tf_4_4)
    # pw_T_dfdp_tf_4_4_1 = np.dot(rotation_matrix, pw_T_dfdp_tf_4_4)
    # pw_T_dfdp_tf_4_4_2 = np.dot(rotation_matrix_inv, pw_T_dfdp_tf_4_4)
    # print("pw_T_dfdp_tf_4_4:")
    # print(pw_T_dfdp_tf_4_4)
    # print("pw_T_dfdp_tf_4_4_1:")
    # print(pw_T_dfdp_tf_4_4_1)
    # print("pw_T_dfdp_tf_4_4_2:")
    # print(pw_T_dfdp_tf_4_4_2)
    new_pos_x = pw_T_dfdp_tf_4_4[0][3]
    new_pos_y = pw_T_dfdp_tf_4_4[1][3]
    new_pos_z = pw_T_dfdp_tf_4_4[2][3]
    ori = transformations.quaternion_from_matrix(pw_T_dfdp_tf_4_4) # x, y, z, w
    new_ori_x = ori[0]
    new_ori_y = ori[1]
    new_ori_z = ori[2]
    new_ori_w = ori[3]

    dataset.at[index, 'pos_x'] = new_pos_x
    dataset.at[index, 'pos_y'] = new_pos_y
    dataset.at[index, 'pos_z'] = new_pos_z
    dataset.at[index, 'ori_x'] = new_ori_x
    dataset.at[index, 'ori_y'] = new_ori_y
    dataset.at[index, 'ori_z'] = new_ori_z
    dataset.at[index, 'ori_w'] = new_ori_w




# dataset.time = dataset.time - 4.3
datasetcopy = copy.deepcopy(dataset)
newdataset = pd.DataFrame(columns=['step','time','pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w','alg','obj','scene','particle_num','ray_type', 'obj_name', 'mass', 'friction'],index=[])
timestep_list = []
# for timestep in range(prepare_time):
for timestep in range(int(prepare_time/rosbag_slowdown_rate)):
    timestep_list.append(timestep/100.0 * rosbag_slowdown_rate)
if update_style_flag == "time" and task_flag == "2" and correct_time_flag == True:
    correct_time(datasetcopy)
timedf = datasetcopy['time']
# print(datasetcopy)
# datasetcopy.to_csv("test",index=0,header=0,mode='a')
for i in range(int(prepare_time/rosbag_slowdown_rate)):
    print("Align time: "+file_name+" processing... ", i)
    newdata = (timedf - timestep_list[int(i)]).abs()
    #print(newdata)
    #print(newdata.idxmin())
    if object_name == "gelatin" and ang_and_pos == "ang":
        datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[int(i)]
        newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                             datasetcopy.loc[newdata.idxmin(),'time'],
                             angle_correction(datasetcopy.loc[newdata.idxmin(),'error']+math.pi),
                             datasetcopy.loc[newdata.idxmin(),'alg'],
                             datasetcopy.loc[newdata.idxmin(),'obj_scene'],
                             datasetcopy.loc[newdata.idxmin(),'particle_num'],
                             datasetcopy.loc[newdata.idxmin(),'ray_type'],
                             datasetcopy.loc[newdata.idxmin(),'obj_name']]
    else:
        datasetcopy.loc[datasetcopy.index==newdata.idxmin(),'time'] = timestep_list[int(i)]
        newdataset.loc[i] = [datasetcopy.loc[newdata.idxmin(),'index'],
                             datasetcopy.loc[newdata.idxmin(),'time'],
                             datasetcopy.loc[newdata.idxmin(),'pos_x'],
                             datasetcopy.loc[newdata.idxmin(),'pos_y'],
                             datasetcopy.loc[newdata.idxmin(),'pos_z'],
                             datasetcopy.loc[newdata.idxmin(),'ori_x'],
                             datasetcopy.loc[newdata.idxmin(),'ori_y'],
                             datasetcopy.loc[newdata.idxmin(),'ori_z'],
                             datasetcopy.loc[newdata.idxmin(),'ori_w'],
                             datasetcopy.loc[newdata.idxmin(),'alg'],
                             datasetcopy.loc[newdata.idxmin(),'obj'],
                             datasetcopy.loc[newdata.idxmin(),'scene'],
                             datasetcopy.loc[newdata.idxmin(),'particle_num'],
                             datasetcopy.loc[newdata.idxmin(),'ray_type'],
                             datasetcopy.loc[newdata.idxmin(),'obj_name'],
                             datasetcopy.loc[newdata.idxmin(),'mass'],
                             datasetcopy.loc[newdata.idxmin(),'friction']]
# print(newdataset.time)
# print(str(particle_num)+'_'+object_name+'_'+task_flag+'_rosbag'+str(rosbag_flag)+'_repeat'+str(repeat_time)+'_'+run_alg_flag+'_'+ang_and_pos)
print("Done")
newdataset.to_csv(save_file_path+save_file_name, index=0, header=0, mode='w')

