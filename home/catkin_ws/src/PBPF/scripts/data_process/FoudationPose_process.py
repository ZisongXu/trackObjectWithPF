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



class ImageCreator():
    def __init__(self, bagfile, rgbpath, depthpath, rgbstamp, depthstamp, all_data_list):
        self.bridge = CvBridge()
        with rosbag.Bag(bagfile, 'r') as bag:
            first_get_time_flag = 0
            timestr_first = 0
            count = 0
            FOUN_panda_step = 0
            for topic, msg, t in bag.read_messages():              
                if topic == "/camera/color/image_raw":
                    if first_get_time_flag == 0:
                        first_get_time_flag = 1
                        timestr_first = "%.6f" %  msg.header.stamp.to_sec()
                    timestr = "%.6f" %  msg.header.stamp.to_sec()
                    time_d = float(timestr) - float(timestr_first)
                    time_d = time_d * 20
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
                    boss_FOUD_err_ADD_df.loc[FOUN_panda_step] = [FOUN_panda_step, time_d, pos_x, pos_y, pos_z, ori[0], ori[1], ori[2], ori[3], "FOUD", obj, scene, PARTICLE_NUM, VERSION, OBJ_NAME]
                    FOUN_panda_step = FOUN_panda_step + 1
                    # print(time_d)

        file_save_path = os.path.expanduser('~/catkin_ws/src/PBPF/scripts/results/')
        file_name_FOUD_ADD = str(PARTICLE_NUM)+"_scene"+task_flag+"_rosbag"+str(ROSBAG_TIME)+"_repeat"+str(REPEAT_TIME)+"_"+OBJ_NAME+"_"+update_style_flag+'_FOUD_pose_'+RUNNING_MODEL+'.csv'

        boss_FOUD_err_ADD_df.to_csv(file_save_path+file_name_FOUD_ADD,index=0,header=0,mode='w')
        print(file_save_path+file_name_FOUD_ADD)
        print("Done")


if __name__ == '__main__':

    OBJ_NAME = sys.argv[1]
    SCENE_NAMES = sys.argv[2]
    ROSBAG_TIME = sys.argv[3]
    REPEAT_TIME = sys.argv[4]

    SIM_REAL_WORLD_FLAG = True
    boss_FOUD_err_ADD_df_list = []
    boss_FOUD_err_ADD_df = pd.DataFrame(columns=['step','time','pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w','alg','obj','scene','particle_num','ray_type','obj_name'],index=[])
    
    # rospy.init_node('record_FOUD_error') # ros node

    time.sleep(0.5)
    

    # pw_T_cam_pose = np.array([[-0.05090748, 0.27306657,-0.96064722, 0.98427103],
    #                           [ 0.99835346, 0.03937482,-0.04171324, 0.06905193],
    #                           [ 0.02643482,-0.96118899,-0.27462143, 0.93720667],
    #                           [ 0.        , 0.        , 0.        , 1.        ]])
    # pw_T_cam_pose = np.array([[-0.17022463,  0.22072718, -0.96036612,  1.01227219],
    #                           [ 0.98534948,  0.02775525, -0.16827375,  0.09076827],
    #                           [-0.01048739, -0.97494059, -0.22221804,  0.93997983],
    #                           [ 0.        ,  0.        ,  0.        ,  1.        ]])



    # pw_T_cam_pose = np.array([[-0.17256843,  0.21958502, -0.96020964,  1.0094713 ],
    #                           [ 0.98494236,  0.02815054, -0.17057579,  0.09233854],
    #                           [-0.01042547, -0.97518714, -0.22113648,  0.93911341],
    #                           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    # pw_T_cam_pose = np.array([[-0.11932691,  0.22814166, -0.96628798,  0.9232672 ],
    #                           [ 0.99279037,  0.01631176, -0.11874847,  0.10087403],
    #                           [-0.01132961, -0.9734913 , -0.22844328,  0.92533336],
    #                           [ 0.        ,  0.        ,  0.        ,  1.        ]])


    pw_T_cam_pose = np.array([[ 3.96235287e-04,  2.53526658e-01, -9.67328319e-01,  9.51147287e-01],
                              [ 9.99988399e-01, -4.74415418e-03, -8.33779851e-04,  6.27580045e-02],
                              [-4.80054010e-03, -9.67316767e-01, -2.53525596e-01,  9.49659315e-01],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])



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
    ImageCreator(rosbag_file_path+'1_scene1_Milk1.bag', "/home/zisongxu/catkin_ws/src/PBPF/scripts/rayTracing/ob_in_cam/000000000/", "/home/sc19zx/depth/", 1, 1, _all_data_list)
    

