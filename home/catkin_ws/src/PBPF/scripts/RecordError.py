#!/usr/bin/python3
from gazebo_msgs.msg import ModelStates
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
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3
from PBPF.msg import estimated_obj_pose, object_pose, particle_list, particle_pose
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
import yaml
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
from Robot_Pose import Robot_Pose


with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
    parameter_info = yaml.safe_load(file)

compute_error_flag = True
update_style_flag = parameter_info['update_style_flag'] # time/pose
run_alg_flag = parameter_info['run_alg_flag'] # PBPF/CVPF
task_flag = parameter_info['task_flag'] # 1/2/3/4 parameter_info['task_flag']
file_name = sys.argv[1]
err_file = parameter_info['err_file']
DOPE_fresh_flag = False
# file_time = 11 # 1~10
# when optitrack does not work
write_opti_pose_flag = "False"
write_estPB_pose_flag = "False"
write_estDO_pose_flag = "False"
print("The "+str(file_name)+"th time")
# panda data frame to record the error and to compare them
# pos
if compute_error_flag == True:
    # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
    boss_opti_pos_x_df = pd.DataFrame(columns=['step','time','pos_x','alg'],index=[])
    boss_opti_pos_y_df = pd.DataFrame(columns=['step','time','pos_y','alg'],index=[])
    boss_opti_pos_z_df = pd.DataFrame(columns=['step','time','pos_z','alg'],index=[])
    boss_opti_ori_x_df = pd.DataFrame(columns=['step','time','ang_x','alg'],index=[])
    boss_opti_ori_y_df = pd.DataFrame(columns=['step','time','ang_y','alg'],index=[])
    boss_opti_ori_z_df = pd.DataFrame(columns=['step','time','ang_z','alg'],index=[])
    boss_opti_ori_w_df = pd.DataFrame(columns=['step','time','ang_w','alg'],index=[])
    boss_estPB_pos_x_df = pd.DataFrame(columns=['step','time','pos_x','alg'],index=[])
    boss_estPB_pos_y_df = pd.DataFrame(columns=['step','time','pos_y','alg'],index=[])
    boss_estPB_pos_z_df = pd.DataFrame(columns=['step','time','pos_z','alg'],index=[])
    boss_estPB_ori_x_df = pd.DataFrame(columns=['step','time','ang_x','alg'],index=[])
    boss_estPB_ori_y_df = pd.DataFrame(columns=['step','time','ang_y','alg'],index=[])
    boss_estPB_ori_z_df = pd.DataFrame(columns=['step','time','ang_z','alg'],index=[])
    boss_estPB_ori_w_df = pd.DataFrame(columns=['step','time','ang_w','alg'],index=[])
    boss_estDO_pos_x_df = pd.DataFrame(columns=['step','time','pos_x','alg'],index=[])
    boss_estDO_pos_y_df = pd.DataFrame(columns=['step','time','pos_y','alg'],index=[])
    boss_estDO_pos_z_df = pd.DataFrame(columns=['step','time','pos_z','alg'],index=[])
    boss_estDO_ori_x_df = pd.DataFrame(columns=['step','time','ang_x','alg'],index=[])
    boss_estDO_ori_y_df = pd.DataFrame(columns=['step','time','ang_y','alg'],index=[])
    boss_estDO_ori_z_df = pd.DataFrame(columns=['step','time','ang_z','alg'],index=[])
    boss_estDO_ori_w_df = pd.DataFrame(columns=['step','time','ang_w','alg'],index=[])

def rotation_4_4_to_transformation_4_4(rotation_4_4, pos):
    rotation_4_4[0][3] = pos[0]
    rotation_4_4[1][3] = pos[1]
    rotation_4_4[2][3] = pos[2]
    return rotation_4_4

# compute the position distance between two objects
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

# compute the angle distance between two objects
def compute_ang_err_bt_2_points(object1_ori, object2_ori):
    #[x, y, z, w]
    obj1_ori = copy.deepcopy(object1_ori)
    obj2_ori = copy.deepcopy(object2_ori)
    #[w, x, y, z]
    obj1_quat = Quaternion(x = obj1_ori[0], y = obj1_ori[1], z = obj1_ori[2], w = obj1_ori[3])
    obj2_quat = Quaternion(x = obj2_ori[0], y = obj2_ori[1], z = obj2_ori[2], w = obj2_ori[3])
    diff_bt_o1_o2 = obj2_quat * obj1_quat.inverse
    cos_theta_over_2 = diff_bt_o1_o2.w
    sin_theta_over_2 = math.sqrt(diff_bt_o1_o2.x ** 2 + diff_bt_o1_o2.y ** 2 + diff_bt_o1_o2.z ** 2)
    theta_over_2 = math.atan2(sin_theta_over_2, cos_theta_over_2)
    theta = theta_over_2 * 2
    return theta

# make sure all angles all between -pi and +pi
def angle_correction(angle):
    if math.pi <= angle <= (3.0 * math.pi):
        angle = angle - 2 * math.pi
    elif -(3.0 * math.pi) <= angle <= -math.pi:
        angle = angle + 2 * math.pi
    angle = abs(angle)
    return angle

# compute the transformation matrix represent that the pose of object in the robot world
def compute_transformation_matrix(a_pos, a_ori, b_pos, b_ori):
    ow_T_a_3_3 = transformations.quaternion_matrix(a_ori)
    ow_T_a_4_4 = rotation_4_4_to_transformation_4_4(ow_T_a_3_3,a_pos)
    ow_T_b_3_3 = transformations.quaternion_matrix(b_ori)
    ow_T_b_4_4 = rotation_4_4_to_transformation_4_4(ow_T_b_3_3,b_pos)
    a_T_ow_4_4 = np.linalg.inv(ow_T_a_4_4)
    a_T_b_4_4 = np.dot(a_T_ow_4_4,ow_T_b_4_4)
    return a_T_b_4_4        

def signal_handler(sig, frame):
    if run_alg_flag == "PBPF":
        for obj_index in range(object_num):
            obj_name = object_name_list[obj_index]
            # file_name_obse_pos = object_name_list[obj_index]+'_'+update_style_flag+'_scene'+task_flag+'_obse_err_pos.csv'
            # file_name_PBPF_pos = object_name_list[obj_index]+'_'+update_style_flag+'_scene'+task_flag+'_PBPF_err_pos.csv'
            # file_name_obse_ang = object_name_list[obj_index]+'_'+update_style_flag+'_scene'+task_flag+'_obse_err_ang.csv'
            # file_name_PBPF_ang = object_name_list[obj_index]+'_'+update_style_flag+'_scene'+task_flag+'_PBPF_err_ang.csv'

            file_name_obse_pos = update_style_flag+'_obse_err_pos.csv'
            file_name_PBPF_pos = update_style_flag+'_PBPFV_err_pos.csv'
            file_name_obse_ang = update_style_flag+'_obse_err_ang.csv'
            file_name_PBPF_ang = update_style_flag+'_PBPFV_err_ang.csv'

            boss_obse_err_pos_df_list[obj_index].to_csv('catkin_ws/src/PBPF/scripts/'+err_file+'/'+str(file_name)+obj_name+'_'+file_name_obse_pos,index=0,header=0,mode='a')
            boss_obse_err_ang_df_list[obj_index].to_csv('catkin_ws/src/PBPF/scripts/'+err_file+'/'+str(file_name)+obj_name+'_'+file_name_obse_ang,index=0,header=0,mode='a')
            print("write "+obj_name+" obser file")
            boss_PBPF_err_pos_df_list[obj_index].to_csv('catkin_ws/src/PBPF/scripts/'+err_file+'/'+str(file_name)+obj_name+'_'+file_name_PBPF_pos,index=0,header=0,mode='a')
            boss_PBPF_err_ang_df_list[obj_index].to_csv('catkin_ws/src/PBPF/scripts/'+err_file+'/'+str(file_name)+obj_name+'_'+file_name_PBPF_ang,index=0,header=0,mode='a')
            print("write "+obj_name+" PBPF file")

            
    if run_alg_flag == "CVPF":
        for obj_index in range(object_num):
            # file_name_CVPF_pos = object_name_list[obj_index]+'_'+update_style_flag+'_scene'+task_flag+'_CVPF_err_pos.csv'
            # file_name_CVPF_ang = object_name_list[obj_index]+'_'+update_style_flag+'_scene'+task_flag+'_CVPF_err_ang.csv'

            file_name_CVPF_pos = update_style_flag+'_CVPF_err_pos.csv'
            file_name_CVPF_ang = update_style_flag+'_CVPF_err_ang.csv'
            
            # boss_CVPF_err_pos_df_list[obj_index].to_csv('catkin_ws/src/PBPF/scripts/'+err_file+'/'+str(file_name)+file_name_CVPF_pos,index=0,header=0,mode='a')
            # boss_CVPF_err_ang_df_list[obj_index].to_csv('catkin_ws/src/PBPF/scripts/'+err_file+'/'+str(file_name)+file_name_CVPF_ang,index=0,header=0,mode='a')
            # print("write CVPF file")
    print("file_name:", file_name)

    sys.exit()

if __name__ == '__main__':
    rospy.init_node('record_error') # ros node
    ros_listener = Ros_Listener()
    listener_tf = tf.TransformListener()
    time.sleep(0.5)
    
    object_num = parameter_info['object_num']
    robot_num = 1
    check_dope_work_flag_init = 0
    other_obj_num = 0
    particle_num = parameter_info['particle_num']
    object_name_list = parameter_info['object_name_list']
    init_esti_flag = 0
    flag_record_obse = 0
    flag_record_PBPF = 0
    flag_record_CVPF = 0
    first_get_info_from_tf_flag = 0
    first_get_info_from_tf_flag_gt = 0
    gazebo_flag = parameter_info['gazebo_flag']
    task_flag = parameter_info['task_flag']
    version = parameter_info['version']

    # pos
    boss_obse_err_pos_df_list = []
    boss_PBPF_err_pos_df_list = []
    boss_CVPF_err_pos_df_list = []
    boss_err_pos_df_list = []
    boss_obse_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
    boss_PBPF_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
    boss_CVPF_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
    # boss_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
    # ang
    boss_obse_err_ang_df_list = []
    boss_PBPF_err_ang_df_list = []
    boss_CVPF_err_ang_df_list = []
    boss_err_ang_df_list = []
    boss_obse_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
    boss_PBPF_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
    boss_CVPF_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
    # boss_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])

    pw_T_rob_sim_pos = [0.0, 0.0, 0.026]
    pw_T_rob_sim_pos = [0.0, 0.0, 0.02]
    pw_T_rob_sim_ori = [0, 0, 0, 1]
    pw_T_rob_sim_3_3 = transformations.quaternion_matrix(pw_T_rob_sim_ori)
    pw_T_rob_sim_4_4 = rotation_4_4_to_transformation_4_4(pw_T_rob_sim_3_3, pw_T_rob_sim_pos)
    
    signal.signal(signal.SIGINT, signal_handler) # interrupt judgment
    esti_obj_list_not_pub = 2
    t_begin = 0
    t_before_record = 0

    trans_ob_list = []
    rot_ob_list = []
    trans_gt_list = []
    rot_gt_list = []

    for _ in range(object_num):
        trans_ob_list.append("pos_value")
        rot_ob_list.append("rad_value")
        trans_gt_list.append("pos_value")
        rot_gt_list.append("rad_value")

    while True:
        for obj_index in range(object_num):
            obj_name = object_name_list[obj_index]
            if first_get_info_from_tf_flag == 0:
                boss_obse_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
                boss_obse_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
                boss_PBPF_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
                boss_PBPF_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
                boss_CVPF_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
                boss_CVPF_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg','obj_scene','particle_num','ray_type','obj_name'],index=[])
                
                boss_obse_err_pos_df_list.append(boss_obse_err_pos_df)
                boss_obse_err_ang_df_list.append(boss_obse_err_ang_df)
                boss_PBPF_err_pos_df_list.append(boss_PBPF_err_pos_df)
                boss_PBPF_err_ang_df_list.append(boss_PBPF_err_ang_df)
                boss_CVPF_err_pos_df_list.append(boss_CVPF_err_pos_df)
                boss_CVPF_err_ang_df_list.append(boss_CVPF_err_ang_df)
                if obj_index == object_num - 1:
                    first_get_info_from_tf_flag = 1
                use_gazebo = ""
                if gazebo_flag == True:
                    use_gazebo = '_noise'
                while True:
                    try:
                        (trans_ob,rot_ob) = listener_tf.lookupTransform('/panda_link0', '/'+object_name_list[obj_index]+use_gazebo, rospy.Time(0))
                        trans_ob_list[obj_index] = trans_ob
                        rot_ob_list[obj_index] = rot_ob
                        break
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        continue
                while True:
                    try:
                        (trans_gt,rot_gt) = listener_tf.lookupTransform('/panda_link0', '/'+object_name_list[obj_index], rospy.Time(0))
                        trans_gt_list[obj_index] = trans_gt
                        rot_gt_list[obj_index] = rot_gt
                        break
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        continue
    
            # observation
            obse_is_fresh = True
            try:
                latest_obse_time = listener_tf.getLatestCommonTime('/panda_link0', '/'+object_name_list[obj_index]+use_gazebo)
                if check_dope_work_flag_init == 0:
                    check_dope_work_flag_init = 1
                    old_obse_time = latest_obse_time.to_sec()
                # if (rospy.get_time() - latest_obse_time.to_sec()) < 0.1:
                #     (trans_ob,rot_ob) = listener_tf.lookupTransform('/panda_link0', '/'+object_name_list[obj_index]+use_gazebo, rospy.Time(0))
                #     obse_is_fresh = True
                    # print("obse is FRESH")
                
                if (latest_obse_time.to_sec() > old_obse_time):
                    (trans_ob,rot_ob) = listener_tf.lookupTransform('/panda_link0', '/'+object_name_list[obj_index]+use_gazebo, rospy.Time(0))
                    # if object_name_list[obj_index]+use_gazebo == "cracker": # gelatin, cracker, soup
                    
                    trans_ob_list[obj_index] = trans_ob
                    rot_ob_list[obj_index] = rot_ob
                    
                    obse_is_fresh = True
                    # print(t_before_record - t_begin - 14)
                else:
                    # obse has not been updating for a while
                    obse_is_fresh = False
                    
                    # print("obse is NOT fresh")
                old_obse_time = latest_obse_time.to_sec()
                # break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("can not find tf")
            
            rob_T_obj_obse_pos = list(trans_ob_list[obj_index])
            rob_T_obj_obse_ori = list(rot_ob_list[obj_index])
            # print("---------------------------------------")
            # print("obj_name:", obj_name)
            # print("tf__name:", object_name_list[obj_index]+use_gazebo)
            # print("rob_T_obj_obse_pos:", rob_T_obj_obse_pos)
            # print("---------------------------------------")
            # print("=======================================")
            # print("obj_name:", obj_name)
            # print("tf__name:", object_name_list[obj_index]+use_gazebo)
            # print("rob_T_obj_obse_pos:", rob_T_obj_obse_pos)
            # print("=======================================")
            rob_T_obj_obse_3_3 = transformations.quaternion_matrix(rob_T_obj_obse_ori)
            rob_T_obj_obse_4_4 = rotation_4_4_to_transformation_4_4(rob_T_obj_obse_3_3, rob_T_obj_obse_pos)
            
            pw_T_obj_obse_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_obj_obse_4_4)
            pw_T_obj_obse_pos = [pw_T_obj_obse_4_4[0][3], pw_T_obj_obse_4_4[1][3], pw_T_obj_obse_4_4[2][3]]
            pw_T_obj_obse_ori = transformations.quaternion_from_matrix(pw_T_obj_obse_4_4)
            
            # ground truth pose information
            if gazebo_flag == True:
                obj_name = object_name_list[obj_index]
                obse_is_fresh = True
                try:
                    latest_obse_time = listener_tf.getLatestCommonTime('/panda_link0', '/'+object_name_list[obj_index])
                    if (rospy.get_time() - latest_obse_time.to_sec()) < 0.1:
                        (trans_gt,rot_gt) = listener_tf.lookupTransform('/panda_link0', '/'+object_name_list[obj_index], rospy.Time(0))
                        trans_gt_list[obj_index] = trans_gt
                        rot_gt_list[obj_index] = rot_gt
                        obse_is_fresh = True
                        # print("obse is FRESH")
                    else:
                        # obse has not been updating for a while
                        obse_is_fresh = False
                        print("obse is NOT fresh")
                    # break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print("can not find tf")
                    
                rob_T_obj_opti_pos = list(trans_gt_list[obj_index])
                rob_T_obj_opti_ori = list(rot_gt_list[obj_index])                        
                # rob_T_obj_opti_pos = copy.deepcopy(rob_T_obj_obse_pos)
                # rob_T_obj_opti_ori = copy.deepcopy(rob_T_obj_obse_ori)
                rob_T_obj_opti_3_3 = transformations.quaternion_matrix(rob_T_obj_opti_ori)
                rob_T_obj_opti_4_4 = rotation_4_4_to_transformation_4_4(rob_T_obj_opti_3_3, rob_T_obj_opti_pos)
                
                pw_T_obj_opti_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_obj_opti_4_4)
                pw_T_obj_opti_pos = [pw_T_obj_opti_4_4[0][3], pw_T_obj_opti_4_4[1][3], pw_T_obj_opti_4_4[2][3]]
                pw_T_obj_opti_ori = transformations.quaternion_from_matrix(pw_T_obj_opti_4_4)
            else:
                obj_name = object_name_list[obj_index]
                
                # pose of robot in OptiTrack coordinate frame
                opti_T_rob_opti_pos = ros_listener.listen_2_robot_pose()[0]
                opti_T_rob_opti_ori = ros_listener.listen_2_robot_pose()[1]
                # pose of objects in OptiTrack coordinate frame
                opti_T_obj_opti_pos = ros_listener.listen_2_object_pose(obj_name)[0]
                opti_T_obj_opti_ori = ros_listener.listen_2_object_pose(obj_name)[1]
                # pose of objects in robot coordinate frame
                rob_T_obj_opti_4_4 = compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, opti_T_obj_opti_pos, opti_T_obj_opti_ori)
                
                pw_T_obj_opti_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_obj_opti_4_4)
                pw_T_obj_opti_pos = [pw_T_obj_opti_4_4[0][3], pw_T_obj_opti_4_4[1][3], pw_T_obj_opti_4_4[2][3]]
                pw_T_obj_opti_ori = transformations.quaternion_from_matrix(pw_T_obj_opti_4_4)
            
            # PBPF pose information
            esti_obj_states_list = ros_listener.listen_2_estis_states()
            if len(esti_obj_states_list.objects) == 0:
                esti_obj_list_not_pub = 1
            else:
                if esti_obj_list_not_pub == 1 or esti_obj_list_not_pub == 2:
                    t_begin = time.time()
                esti_obj_list_not_pub = 0
                esti_object_pose = copy.deepcopy(esti_obj_states_list.objects[obj_index])
                esti_obj_name = esti_object_pose.name
                esti_obj_pos_x = esti_object_pose.pose.position.x
                esti_obj_pos_y = esti_object_pose.pose.position.y
                esti_obj_pos_z = esti_object_pose.pose.position.z
                
                esti_obj_ori_x = esti_object_pose.pose.orientation.x
                esti_obj_ori_y = esti_object_pose.pose.orientation.y
                esti_obj_ori_z = esti_object_pose.pose.orientation.z
                esti_obj_ori_w = esti_object_pose.pose.orientation.w
                
                if run_alg_flag == "PBPF":
                    pw_T_obj_PBPF_pos = [esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z]
                    pw_T_obj_PBPF_ori = [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]
                else:
                    pw_T_obj_CVPF_pos = [esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z]
                    pw_T_obj_CVPF_ori = [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]

            if esti_obj_list_not_pub == 0:
                if run_alg_flag == "PBPF":
                    
                    # error bt ground truth and observation
                    err_opti_obse_pos = compute_pos_err_bt_2_points(pw_T_obj_opti_pos, pw_T_obj_obse_pos)
                    # print("=======================================")
                    # print("obj_name:", object_name_list[obj_index]+use_gazebo)
                    # print("err_opti_obse_pos:", err_opti_obse_pos)
                    # print("pw_T_obj_opti_pos:", pw_T_obj_opti_pos)
                    # print("pw_T_obj_obse_pos:", pw_T_obj_obse_pos)
                    # print("=======================================")
                    err_opti_obse_ang = compute_ang_err_bt_2_points(pw_T_obj_opti_ori, pw_T_obj_obse_ori)
                    err_opti_obse_ang = angle_correction(err_opti_obse_ang)
                    # error bt ground truth and PBPF
                    err_opti_PBPF_pos = compute_pos_err_bt_2_points(pw_T_obj_opti_pos, pw_T_obj_PBPF_pos)
                    err_opti_PBPF_ang = compute_ang_err_bt_2_points(pw_T_obj_opti_ori, pw_T_obj_PBPF_ori)
                    err_opti_PBPF_ang = angle_correction(err_opti_PBPF_ang)
                    obj_scene = object_name_list[obj_index]+'_scene'+task_flag
                    t_before_record = time.time()
                    # if obse_is_fresh == True:
                    boss_obse_err_pos_df_list[obj_index].loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_pos, 'obse', obj_scene, particle_num, version, obj_name]
                    boss_obse_err_ang_df_list[obj_index].loc[flag_record_obse] = [flag_record_obse, t_before_record - t_begin, err_opti_obse_ang, 'obse', obj_scene, particle_num, version, obj_name]
                    flag_record_obse = flag_record_obse + 1
                    boss_PBPF_err_pos_df_list[obj_index].loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_PBPF_pos, 'PBPF-V', obj_scene, particle_num, version, obj_name]
                    boss_PBPF_err_ang_df_list[obj_index].loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_PBPF_ang, 'PBPF-V', obj_scene, particle_num, version, obj_name]
                    flag_record_PBPF = flag_record_PBPF + 1
                    
                else:
                    err_opti_CVPF_pos = compute_pos_err_bt_2_points(pw_T_obj_opti_pos, pw_T_obj_CVPF_pos)
                    err_opti_CVPF_ang = compute_ang_err_bt_2_points(pw_T_obj_opti_ori, pw_T_obj_CVPF_ori)
                    err_opti_CVPF_ang = angle_correction(err_opti_CVPF_ang)
                    obj_scene = object_name_list[obj_index]+'_scene'+task_flag
                    t_before_record = time.time()
                    # if obse_is_fresh == True:
                    boss_CVPF_err_pos_df_list[obj_index].loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_CVPF_pos, 'CVPF', obj_scene, particle_num, version, obj_name]
                    boss_CVPF_err_ang_df_list[obj_index].loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_CVPF_ang, 'CVPF', obj_scene, particle_num, version, obj_name]
                    flag_record_CVPF = flag_record_CVPF + 1
                
                
                
