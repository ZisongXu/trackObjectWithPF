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

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import tf
import tf.transformations as transformations
import tf2_ros
from tf2_geometry_msgs import tf2_geometry_msgs


with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
        parameter_info = yaml.safe_load(file)

gazebo_flag = parameter_info['gazebo_flag']
PARTICLE_NUM = parameter_info['particle_num']
# update mode (pose/time)
update_style_flag = parameter_info['update_style_flag'] # time/pose
# which algorithm to run
run_alg_flag = parameter_info['run_alg_flag'] # PBPF/CVPF
# scene
task_flag = parameter_info['task_flag'] # parameter_info['task_flag']
# rosbag_flag = "1"
err_file = parameter_info['err_file']
RUNNING_MODEL = parameter_info['running_model']
VERSION = parameter_info['version']
FRICTION_marker = parameter_info['FRICTION_marker']
MASS_marker = parameter_info['MASS_marker']


class ImageCreator():
    def __init__(self, bagfile, rgbpath, depthpath, rgbstamp, depthstamp, all_data_list):
        self.bridge = CvBridge()
        with rosbag.Bag(bagfile, 'r') as bag:
            first_get_time_flag = 0
            timestr_first = 0
            count = 0
            FOUN_panda_step = 0
            slow_down_rate = 1
            for topic, msg, t in bag.read_messages():              
                if topic == "/camera/color/image_raw":
                    if first_get_time_flag == 0:
                        first_get_time_flag = 1
                        timestr_first = "%.6f" %  msg.header.stamp.to_sec()
                    timestr = "%.6f" %  msg.header.stamp.to_sec()
                    time_d = float(timestr) - float(timestr_first)
                    time_d = time_d * slow_down_rate
                    print(FOUN_panda_step)
                    pw_T_obj_foudation_pose_4_4 = all_data_list[FOUN_panda_step]
                    pos_x = pw_T_obj_foudation_pose_4_4[0][3]
                    pos_y = pw_T_obj_foudation_pose_4_4[1][3]
                    pos_z = pw_T_obj_foudation_pose_4_4[2][3]
                    ori = transformations.quaternion_from_matrix(pw_T_obj_foudation_pose_4_4) # x, y, z, w
                    # if OBJ_NAME == "Parmesan" and SCENE_NAMES == "scene2":
                    #     if ori[3] > 0:
                    #         ori[0] = -ori[0]
                    #         ori[1] = -ori[1]
                    #         ori[2] = -ori[2]
                            # ori[3] = -ori[3]
                    if OBJ_NAME == "cracker" and SCENE_NAMES == "scene2":
                        if ori[1] > 0:
                            ori[1] = -ori[1]
                    # if OBJ_NAME == "soup" and SCENE_NAMES == "scene2":
                    #     if ori[3] > 0:
                    #         ori[0] = -ori[0]
                    #         ori[1] = -ori[1]
                    #         ori[2] = -ori[2]
                    #         ori[3] = -ori[3]
                    # if OBJ_NAME == "SaladDressing" and SCENE_NAMES == "scene1":
                    #     if ori[3] < 0:
                    #         ori[0] = -ori[0]
                    #         ori[1] = -ori[1]
                    #         ori[2] = -ori[2]
                    #         ori[3] = -ori[3]
                    obj = OBJ_NAME
                    scene = "scene"+task_flag
                    obj_scene = OBJ_NAME+"_scene"+task_flag
                    boss_FOUD_err_ADD_df.loc[FOUN_panda_step] = [FOUN_panda_step, time_d, pos_x, pos_y, pos_z, ori[0], ori[1], ori[2], ori[3], "FOUD", obj, scene, PARTICLE_NUM, VERSION, OBJ_NAME, MASS_marker, FRICTION_marker]
                    FOUN_panda_step = FOUN_panda_step + 1
                    # print(time_d)

        file_save_path = os.path.expanduser('~/catkin_ws/src/PBPF/scripts/results/')
        file_name_FOUD_ADD = str(PARTICLE_NUM)+"_scene"+task_flag+"_rosbag"+str(ROSBAG_TIME)+"_repeat"+str(REPEAT_TIME)+"_"+OBJ_NAME+"_"+update_style_flag+'_FOUD_pose_'+RUNNING_MODEL+'_'+MASS_marker+'_'+FRICTION_marker+'.csv'

        boss_FOUD_err_ADD_df.to_csv(file_save_path+file_name_FOUD_ADD,index=0,header=0,mode='w')
        print(file_save_path+file_name_FOUD_ADD)
        print("Done")


if __name__ == '__main__':

    OBJ_NAME = sys.argv[1] # ("cracker" "Ketchup" "Mayo" "Milk" "Mustard" "Parmesan" "SaladDressing" "soup")
    SCENE_NAMES = sys.argv[2] # scene1 scene2
    ROSBAG_TIME = sys.argv[3]
    REPEAT_TIME = sys.argv[4]
    
    SIM_REAL_WORLD_FLAG = True
    boss_FOUD_err_ADD_df_list = []
    boss_FOUD_err_ADD_df = pd.DataFrame(columns=['step','time','pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w','alg','obj','scene','particle_num','ray_type','obj_name','mass','friction'],index=[])
    # rospy.init_node('record_FOUD_error') # ros node

    time.sleep(0.5)
    


    # pw_T_cam_pose = np.array([[ 3.96235287e-04,  2.53526658e-01, -9.67328319e-01,  9.51147287e-01],
    #                           [ 9.99988399e-01, -4.74415418e-03, -8.33779851e-04,  6.27580045e-02],
    #                           [-4.80054010e-03, -9.67316767e-01, -2.53525596e-01,  9.49659315e-01],
    #                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

#     pw_T_cam_pose = np.array([[ 4.00994672e-02,  3.70849964e-01, -9.27826674e-01,  1.09106847e+00],
#  [ 9.99195687e-01, -1.49875652e-02,  3.71934518e-02, -2.02243172e-02],
#  [-1.12672521e-04, -9.28571848e-01, -3.71152678e-01,  1.08219147e+00],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # pw_T_cam_pose = np.array([[-0.044931,    0.23043063, -0.97205089,  1.20441085],
    #                           [ 0.99824785,  0.04785813, -0.03479683,  0.0336495 ],
    #                           [ 0.03850228, -0.97191117, -0.2321772 ,  1.0223675 ],
    #                           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # pw_T_cam_pose = np.array([[-8.95095741e-03,  2.17633877e-01, -9.75989434e-01,  1.21345200e+00],
    #                           [ 9.99005319e-01,  4.45843495e-02,  7.79732597e-04, -6.73263951e-03],
    #                           [ 4.36835502e-02, -9.75011657e-01, -2.17816474e-01,  1.00687529e+00],
    #                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # pw_T_cam_pose = np.array([[ 9.37735911e-02,  2.78147082e-01, -9.55950163e-01,  1.18120405e+00],
    #                           [ 9.95593427e-01, -2.57238183e-02,  9.01776779e-02, -8.42059848e-02],
    #                           [ 4.91969723e-04, -9.60193983e-01, -2.79333621e-01,  1.04590236e+00],
    #                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # pw_T_cam_pose = np.array([[ 0.04771702,  0.29078387, -0.95559815,  1.15756931],
    #                           [ 0.99797884,  0.02631729,  0.05784149,  0.01792461],
    #                           [ 0.04196812, -0.95642676, -0.28894036,  1.0088215 ],
    #                           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    pw_T_cam_pose = np.array([[ 0.07372993,  0.2208193 , -0.9725239 ,  1.10858728],
                              [ 0.99704642,  0.00470366,  0.07665706,  0.01198535],
                              [ 0.02150178, -0.9753034 , -0.21982029,  1.02230446],
                              [ 0.        ,  0.        ,  0.        ,  1.        ]])


    _all_data_list = []
    file_path = os.path.expanduser('~/catkin_ws/src/PBPF/scripts/results/ob_in_cam_'+OBJ_NAME+'/')
    txt_file_count = len([file for file in os.listdir(file_path) if file.endswith('.txt')])

    for i in range(1, txt_file_count+1):
        filename = f"{i:04}.txt"
        cam_T_obj_foudation_pose = np.loadtxt(file_path+filename)
        pw_T_obj_foudation_pose = np.dot(pw_T_cam_pose, cam_T_obj_foudation_pose)

        if OBJ_NAME == "cracker":
            rotation_matrix = np.array([[ 1, 0, 0, 0],
                                        [ 0,-1, 0, 0],
                                        [ 0, 0,-1, 0],
                                        [ 0, 0, 0, 1]])
            pw_T_obj_foudation_pose = np.dot(pw_T_obj_foudation_pose, rotation_matrix)
        # if OBJ_NAME == "Milk":
        #     rotation_matrix = np.array([[ 1, 0, 0, 0],
        #                                 [ 0,-1, 0, 0],
        #                                 [ 0, 0,-1, 0],
        #                                 [ 0, 0, 0, 1]])
        #     pw_T_obj_foudation_pose = np.dot(pw_T_obj_foudation_pose, rotation_matrix)
        #     pw_T_obj_foudation_pose = np.dot(pw_T_obj_foudation_pose, objF_T_objP_z)
        #     pw_T_obj_foudation_pose = np.dot(pw_T_obj_foudation_pose, objF_T_objP__x)
        # elif OBJ_NAME == "soup":
        #     objF_T_objP_x = np.array([[ 1, 0, 0, 0],
        #                               [ 0, 0,-1, 0],
        #                               [ 0, 1, 0, 0],
        #                               [ 0, 0, 0, 1]])
        #     objF_T_objP_y = np.array([[-1, 0, 0, 0],
        #                               [ 0, 1, 0, 0],
        #                               [ 0, 0,-1, 0],
        #                               [ 0, 0, 0, 1]])
        #     pw_T_obj_foudation_pose = np.dot(pw_T_obj_foudation_pose, objF_T_objP_x)
        #     pw_T_obj_foudation_pose = np.dot(pw_T_obj_foudation_pose, objF_T_objP_y)

        _all_data_list.append(pw_T_obj_foudation_pose)
    print(len(_all_data_list))
    _all_time_list = []
    # declare -a objectNames=("Ketchup" "Mayo" "Milk" "SaladDressing" "soup" "Parmesan" "Mustard")
    rosbag_file_path = os.path.expanduser('~/pyvkdepth/rosbag/')
    # ImageCreator(rosbag_file_path+'1_scene2_'+OBJ_NAME+'1.bag', "/home/zisongxu/catkin_ws/src/PBPF/scripts/rayTracing/ob_in_cam/000000000/", "/home/sc19zx/depth/", 1, 1, _all_data_list)
    ImageCreator(rosbag_file_path+'hit_object_obstacle.bag', "", "", 1, 1, _all_data_list)
    


