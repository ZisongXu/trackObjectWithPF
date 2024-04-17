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
OBJECT_NAME_LIST = parameter_info['object_name_list']

OBJ_NAME = sys.argv[1]
ROSBAG_TIME = sys.argv[2]
REPEAT_TIME = sys.argv[3]

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
                    pw_T_obj_foudation_pose_4_4 = all_data_list[FOUN_panda_step]
                    pos_x = pw_T_obj_foudation_pose_4_4[0][3]
                    pos_y = pw_T_obj_foudation_pose_4_4[1][3]
                    pos_z = pw_T_obj_foudation_pose_4_4[2][3]
                    ori = transformations.quaternion_from_matrix(pw_T_obj_foudation_pose_4_4) # x, y, z, w

                    obj_scene = OBJ_NAME+"_scene"+task_flag
                    boss_FOUD_err_ADD_df.loc[FOUN_panda_step] = [FOUN_panda_step, time_d, pos_x, pos_y, pos_z, ori[0], ori[1], ori[2], ori[3], "FOUD", obj_scene, PARTICLE_NUM, VERSION, OBJ_NAME]
                    FOUN_panda_step = FOUN_panda_step + 1
                    # print(time_d)

        file_save_path = os.path.expanduser('~/catkin_ws/src/PBPF/scripts/results/')
        file_name_FOUD_ADD = str(PARTICLE_NUM)+"_scene"+task_flag+"_rosbag"+str(ROSBAG_TIME)+"_repeat"+str(REPEAT_TIME)+"_"+OBJ_NAME+"_"+update_style_flag+'_FOUD_pose_'+RUNNING_MODEL+'.csv'

        boss_FOUD_err_ADD_df.to_csv(file_save_path+file_name_FOUD_ADD,index=0,header=0,mode='w')
        print(file_save_path+file_name_FOUD_ADD)
        print("Done")


if __name__ == '__main__':

    
    SIM_REAL_WORLD_FLAG = True
    OBJECT_NUM = 2
    OBJECT_NAME_LIST = ["cracker", "soup"]
    boss_FOUD_err_ADD_df_list = []
    boss_FOUD_err_ADD_df = pd.DataFrame(columns=['step','time','pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
    
    # rospy.init_node('record_FOUD_error') # ros node

    time.sleep(0.5)
    

    pw_T_cam_pose = np.array([[-0.05090748, 0.27306657,-0.96064722, 0.98427103],
                              [ 0.99835346, 0.03937482,-0.04171324, 0.06905193],
                              [ 0.02643482,-0.96118899,-0.27462143, 0.93720667],
                              [ 0.        , 0.        , 0.        , 1.        ]])
    _all_data_list = []
    file_path = os.path.expanduser('~/catkin_ws/src/PBPF/scripts/ob_in_cam/')
    txt_file_count = len([file for file in os.listdir(file_path) if file.endswith('.txt')])

    for i in range(1, txt_file_count+1):
        filename = f"{i:04}.txt"
        cam_T_obj_foudation_pose = np.loadtxt(file_path+filename)
        pw_T_obj_foudation_pose = np.dot(pw_T_cam_pose, cam_T_obj_foudation_pose)
        _all_data_list.append(pw_T_obj_foudation_pose)
    # print(_all_data_list)
    _all_time_list = []

    rosbag_file_path = os.path.expanduser('~/pyvkdepth/rosbag/')
    ImageCreator(rosbag_file_path+'scene1_new_camera_CrackerSoup_forward3.bag', "/home/zisongxu/catkin_ws/src/PBPF/scripts/rayTracing/ob_in_cam/000000000/", "/home/sc19zx/depth/", 1, 1, _all_data_list)
    


