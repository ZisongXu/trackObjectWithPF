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
import yaml

#Class of initialize the real world model
class Visualisation_World():
    def __init__(self, object_num=0, rob_num=1, particle_num=0):
        self.object_num = object_num
        self.rob_num = rob_num
        self.particle_num = particle_num
        self.p_visualisation = 0
        self.create_scene = Create_Scene(object_num, rob_num)
        self.ros_listener = Ros_Listener()
        self.listener = tf.TransformListener()
        self.visualisation_all = True
        with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
            self.parameter_info = yaml.safe_load(file)
        self.gazebo_flag = self.parameter_info['gazebo_flag']
        self.task_flag = self.parameter_info['task_flag']
        self.optitrack_flag = self.parameter_info['optitrack_flag']
        self.pw_T_rob_sim_pose_list = []
        self.pw_T_target_obj_obse_pose_lsit = []
        self.pw_T_target_obj_opti_pose_lsit = []
        self.pw_T_other_obj_opti_pose_list = []
        self.pw_T_objs_not_touching_targetObjs_list = []
        
        self.object_name_list = self.parameter_info['object_name_list']
        self.SIM_REAL_WORLD_FLAG = self.parameter_info['sim_real_world_flag']
        
        
        self.test = False 
        
    def initialize_visual_world_pybullet_env(self, task_flag):
        trans_ob = []
        rot_ob = []
        trans_gt = []
        rot_gt = []
        
        # (Basic Setting
        if self.visualisation_all == True:
            p_visualisation = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT, GUI_SERVER
        else:
            p_visualisation = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT, GUI_SERVER
        self.p_visualisation = p_visualisation
        p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_visualisation.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p_visualisation.setGravity(0, 0, -9.81)
        # p_visualisation.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0.1,0.1,0.1])      
        p_visualisation.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85, cameraTargetPosition=[0.3,0.1,0.1])    
        plane_id = p_visualisation.loadURDF("plane.urdf")
        if self.task_flag == "1":
            pw_T_pringles_pos = [0.6652218209791124, 0.058946644391304814, 0.8277292172960276]
            pw_T_pringles_ori = [ 0.67280124, -0.20574896, -0.20600051, 0.68012472] # x, y, z, w
            pringles_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/pringles.urdf"),
                                                            pw_T_pringles_pos,
                                                            pw_T_pringles_ori,
                                                            useFixedBase=1)
                                                            
        if self.task_flag == "5":
            pw_T_she_pos = [0.75889274, -0.24494845, 0.33818097+0.02]
            pw_T_she_ori = [0, 0, 0, 1]
            shelves_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/shelves.urdf"),
                                                  pw_T_she_pos,
                                                  pw_T_she_ori,
                                                  useFixedBase=1)
        
        # load robot in the pybullet world
        pw_T_rob_sim_pose_list = self.create_scene.initialize_robot()
        for rob_index in range(self.rob_num):
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
        self.pw_T_rob_sim_pose_list = pw_T_rob_sim_pose_list
        if self.SIM_REAL_WORLD_FLAG == True:
            table_pos_1 = [0.46, -0.01, 0.710]
            table_pos_1 = [0.46, -0.01, 0.700]
            table_ori_1 = p_visualisation.getQuaternionFromEuler([0,0,0])
            table_id_1 = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/table.urdf"), table_pos_1, table_ori_1, useFixedBase = 1)

            barry_pos_1 = [-1.074, 0.443, 0.895]
            barry_ori_1 = p_visualisation.getQuaternionFromEuler([0,math.pi/2,0])
            barry_id_1 = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_1, barry_ori_1, useFixedBase = 1)
            
            barry_pos_2 = [-1.074, -0.607, 0.895]
            barry_ori_2 = p_visualisation.getQuaternionFromEuler([0,math.pi/2,0])
            barry_id_2 = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_2, barry_ori_2, useFixedBase = 1)

            barry_pos_3 = [0.459, -0.972, 0.895]
            barry_ori_3 = p_visualisation.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
            barry_id_3 = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_3, barry_ori_3, useFixedBase = 1)

            # barry_pos_4 = [-0.549, 0.61, 0.895]
            # barry_ori_4 = p_visualisation.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
            # barry_id_4 = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_4, barry_ori_4, useFixedBase = 1)
            
            # barry_pos_5 = [0.499, 0.61, 0.895]
            # barry_ori_5 = p_visualisation.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
            # barry_id_5 = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_5, barry_ori_5, useFixedBase = 1)

            board_pos_1 = [0.274, 0.581, 0.87575]
            board_ori_1 = p_visualisation.getQuaternionFromEuler([math.pi/2,math.pi/2,0])
            board_id_1 = p_visualisation.loadURDF(os.path.expanduser("~/project/object/others/board.urdf"), board_pos_1, board_ori_1, useFixedBase = 1)
           

        # observation: target obejct pose list
        pw_T_target_obj_obse_pose_lsit, trans_ob_list, rot_ob_list = self.create_scene.initialize_object()
        self.pw_T_target_obj_obse_pose_lsit = pw_T_target_obj_obse_pose_lsit
        # print("I am here")
        
        # load other objects in the pybullet world
        if self.test == False:
            if self.optitrack_flag == True:
                print("Load Target Object from OptiTrackssssssssss")
                pw_T_target_obj_opti_pose_lsit, pw_T_other_obj_opti_pose_list, pw_T_objs_not_touching_targetObjs_list, trans_gt, rot_gt = self.create_scene.initialize_ground_truth_objects()
                
                self.pw_T_target_obj_opti_pose_lsit = pw_T_target_obj_opti_pose_lsit
                
                for obj_index in range(len(pw_T_other_obj_opti_pose_list)):
                    other_obj_name = pw_T_other_obj_opti_pose_list[obj_index].obj_name
                    other_obj_pos = pw_T_other_obj_opti_pose_list[obj_index].pos
                    other_obj_ori = pw_T_other_obj_opti_pose_list[obj_index].ori
                    use_gazebo = ""
                    if self.gazebo_flag == True:
                        use_gazebo = "gazebo_"
                    optitrack_base_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+use_gazebo+other_obj_name+"/base_of_cracker.urdf"),
                                                                other_obj_pos,
                                                                other_obj_ori)
                    pw_T_other_obj_opti_pose_list[obj_index].obj_id = optitrack_base_id

                self.pw_T_other_obj_opti_pose_list = pw_T_other_obj_opti_pose_list
                self.pw_T_objs_not_touching_targetObjs_list = pw_T_objs_not_touching_targetObjs_list

        # load objects in the pybullet world
        # for obj_index in range(self.object_num):
        #     obse_obj_name = pw_T_target_obj_obse_pose_lsit[obj_index].obj_name
        #     obse_obj_pos = pw_T_target_obj_obse_pose_lsit[obj_index].pos
        #     obse_obj_ori = pw_T_target_obj_obse_pose_lsit[obj_index].ori
        #     use_gazebo = ""
        #     if self.gazebo_flag == True:
        #         use_gazebo = "gazebo_"
        #     obse_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+use_gazebo+obse_obj_name+"/"+use_gazebo+obse_obj_name+"_obse_obj_with_visual_hor.urdf"),
        #                                               obse_obj_pos,
        #                                               obse_obj_ori)
        #     pw_T_target_obj_obse_pose_lsit[obj_index].obj_id = obse_object_id
        #     opti_obj_name = pw_T_target_obj_opti_pose_lsit[obj_index].obj_name
        #     opti_obj_pos = pw_T_target_obj_opti_pose_lsit[obj_index].pos
        #     opti_obj_ori = pw_T_target_obj_opti_pose_lsit[obj_index].ori
        #     use_gazebo = ""
        #     if self.gazebo_flag == True:
        #         use_gazebo = "gazebo_"
        #     opti_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+use_gazebo+opti_obj_name+"/"+use_gazebo+opti_obj_name+"_real_obj_with_visual_hor.urdf"),
        #                                               opti_obj_pos,
        #                                               opti_obj_ori)
        #     pw_T_target_obj_opti_pose_lsit[obj_index].obj_id = opti_object_id

        for i in range(240):
            p_visualisation.stepSimulation()
            
        return trans_ob_list, rot_ob_list, trans_gt, rot_gt
    
    def set_real_robot_JointPosition(self, pybullet_simulation_env, robot_id, position):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                pybullet_simulation_env.setJointMotorControl2(robot_id,
                                                              joint_index+2,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])
            else:
                pybullet_simulation_env.setJointMotorControl2(robot_id,
                                                              joint_index,
                                                              pybullet_simulation_env.POSITION_CONTROL,
                                                              targetPosition=position[joint_index])
        
    def display_object_in_visual_model(self, pybullet_simulation_env, object_info_list):
        obj_pos = object_info_list.pos
        obj_ori = object_info_list.ori
        object_id = object_info_list.obj_id
        pybullet_simulation_env.resetBasePositionAndOrientation(object_id,
                                                                obj_pos,
                                                                obj_ori)
            
    def init_display_particle(self, object_pose):
        obj_par_name = object_pose.name
        obj_pos_x = object_pose.pose.position.x
        obj_pos_y = object_pose.pose.position.y
        obj_pos_z = object_pose.pose.position.z
        obj_pos = [obj_pos_x, obj_pos_y, obj_pos_z]
        obj_ori_x = object_pose.pose.orientation.x
        obj_ori_y = object_pose.pose.orientation.y
        obj_ori_z = object_pose.pose.orientation.z
        obj_ori_w = object_pose.pose.orientation.w
        obj_ori = [obj_ori_x, obj_ori_y, obj_ori_z, obj_ori_w]
        use_gazebo = ""
        if self.gazebo_flag == True:
            use_gazebo = "gazebo_"
        visualize_particle_Id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+use_gazebo+obj_par_name+"/"+use_gazebo+obj_par_name+"_par_with_visual_PB_hor.urdf"),
                                                              obj_pos,
                                                              obj_ori)
        object_pose.id = visualize_particle_Id
            
    def display_particle_in_visual_model(self, object_pose):
        obj_par_id = object_pose.id
        obj_pos_x = object_pose.pose.position.x
        obj_pos_y = object_pose.pose.position.y
        obj_pos_z = object_pose.pose.position.z
        obj_pos = [obj_pos_x, obj_pos_y, obj_pos_z]
        obj_ori_x = object_pose.pose.orientation.x
        obj_ori_y = object_pose.pose.orientation.y
        obj_ori_z = object_pose.pose.orientation.z
        obj_ori_w = object_pose.pose.orientation.w
        obj_ori = [obj_ori_x, obj_ori_y, obj_ori_z, obj_ori_w]
        
        self.p_visualisation.resetBasePositionAndOrientation(obj_par_id,
                                                             obj_pos,
                                                             obj_ori)
    
    def init_display_particle_CV(self, particle_cloud_CV):
        for index, particle in enumerate(particle_cloud_CV):
            obj_id_list = []
            for obj_index in range(self.object_num):
                obj_par_name = particle[obj_index].par_name
                obj_par_pos = particle[obj_index].pos
                obj_par_ori = particle[obj_index].ori
                use_gazebo = ""
                if self.gazebo_flag == True:
                    use_gazebo = "gazebo_"
                visualize_particle_Id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+use_gazebo+obj_par_name+"/"+use_gazebo+obj_par_name+"_par_with_visual_CV_hor.urdf"),
                                                                      obj_par_pos,
                                                                      obj_par_ori)
                obj_id_list.append(visualize_particle_Id)
                particle[obj_index].visual_par_id = visualize_particle_Id
                
    def init_display_ground_truth_object(self, gt_object_pose):
        gt_obj_name = gt_object_pose.obj_name
        gt_obj_pos = copy.deepcopy(gt_object_pose.pos)
        gt_obj_ori = copy.deepcopy(gt_object_pose.ori)
        use_gazebo = ""
        if self.gazebo_flag == True:
            use_gazebo = "gazebo_"
        ground_truth_object_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+use_gazebo+gt_obj_name+"/"+use_gazebo+gt_obj_name+"_real_obj_with_visual_hor.urdf"),
                                                               gt_obj_pos,
                                                               gt_obj_ori)
        gt_object_pose.obj_id = ground_truth_object_id

    def init_display_observation_object(self, obse_object_pose):
        # print("i am here")
        obse_obj_name = obse_object_pose.obj_name
        obse_obj_pos = copy.deepcopy(obse_object_pose.pos)
        obse_obj_ori = copy.deepcopy(obse_object_pose.ori)
        use_gazebo = ""
        if self.gazebo_flag == True:
            use_gazebo = "gazebo_"
        observation_object_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+use_gazebo+obse_obj_name+"/"+use_gazebo+obse_obj_name+"_obse_obj_with_visual_hor.urdf"),
                                                            obse_obj_pos,
                                                            obse_obj_ori)
        obse_object_pose.obj_id = observation_object_id


    def init_display_estimated_object(self, esti_object_pose):
        esti_obj_name = esti_object_pose.name
        esti_obj_pos_x = esti_object_pose.pose.position.x
        esti_obj_pos_y = esti_object_pose.pose.position.y
        esti_obj_pos_z = esti_object_pose.pose.position.z
        esti_obj_pos = [esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z]
        esti_obj_ori_x = esti_object_pose.pose.orientation.x
        esti_obj_ori_y = esti_object_pose.pose.orientation.y
        esti_obj_ori_z = esti_object_pose.pose.orientation.z
        esti_obj_ori_w = esti_object_pose.pose.orientation.w
        esti_obj_ori = [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]
        use_gazebo = ""
        if self.gazebo_flag == True:
            use_gazebo = "gazebo_"
        estimated_object_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+use_gazebo+esti_obj_name+"/"+use_gazebo+esti_obj_name+"_est_obj_with_visual_PB_hor.urdf"),
                                                            esti_obj_pos,
                                                            esti_obj_ori)
        esti_object_pose.id = estimated_object_id

        
    def display_estimated_object_in_visual_model(self, esti_object_pose):
        esti_obj_id = esti_object_pose.id
        esti_obj_pos_x = esti_object_pose.pose.position.x
        esti_obj_pos_y = esti_object_pose.pose.position.y
        esti_obj_pos_z = esti_object_pose.pose.position.z
        esti_obj_pos = [esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z]
        esti_obj_ori_x = esti_object_pose.pose.orientation.x
        esti_obj_ori_y = esti_object_pose.pose.orientation.y
        esti_obj_ori_z = esti_object_pose.pose.orientation.z
        esti_obj_ori_w = esti_object_pose.pose.orientation.w
        esti_obj_ori = [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]
        
        self.p_visualisation.resetBasePositionAndOrientation(esti_obj_id,
                                                             esti_obj_pos,
                                                             esti_obj_ori)
        
    

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
        
# ctrl-c write down the error file
def signal_handler(sig, frame):
    sys.exit()

reset_flag = True

while reset_flag == True:
    reset_flag = False

    if __name__ == '__main__':
    #    par_obj_id = [[]*2 for _ in range(50)]
    #    print(par_obj_id)
    #    input("stop")
        rospy.init_node('visualization_world_GT') # ros node
        signal.signal(signal.SIGINT, signal_handler) # interrupt judgment
        time.sleep(0.5)
        with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
            parameter_info = yaml.safe_load(file)
        object_num = parameter_info['object_num']
        robot_num = 1
        
        particle_num = parameter_info['particle_num']
        optitrack_flag = parameter_info['optitrack_flag']
        
        init_gt_obj_flag = 0
        init_obse_flag = 0
        init_par_flag = 0
        init_esti_flag = 0
        display_par_flag = False
        display_esti_flag = False
        
        display_gt_flag = True
        if optitrack_flag == False:
            display_gt_flag = False
            
        display_obse_flag = True
        object_name_list = parameter_info['object_name_list']
        task_flag = parameter_info['task_flag'] # parameter_info['task_flag']
        dope_flag = parameter_info['dope_flag']

        OBJS_ARE_NOT_TOUCHING_TARGET_OBJS_NUM = parameter_info['objs_are_not_touching_target_objs_num']
        OBJS_TOUCHING_TARGET_OBJS_NUM = parameter_info['objs_touching_target_objs_num']

        visual_world = Visualisation_World(object_num, robot_num, particle_num)
        trans_ob_list, rot_ob_list, trans_gt, rot_gt = visual_world.initialize_visual_world_pybullet_env(task_flag)
        
        # print("I am here")
        # input("stop")
        listener_tf = visual_world.listener
        p_visual = visual_world.p_visualisation
        pw_T_rob_sim_pose_list_param = visual_world.pw_T_rob_sim_pose_list
        pw_T_target_obj_obse_pose_lsit_param = visual_world.pw_T_target_obj_obse_pose_lsit
        pw_T_target_obj_opti_pose_lsit_param = visual_world.pw_T_target_obj_opti_pose_lsit
        pw_T_other_obj_opti_pose_list_param = visual_world.pw_T_other_obj_opti_pose_list
        pw_T_objs_not_touching_targetObjs_list_param = visual_world.pw_T_objs_not_touching_targetObjs_list
        
        par_obj_id = [[]*object_num for _ in range(particle_num)]
        esti_obj_id = [0] * object_num
        # input("stop")
        while not rospy.is_shutdown():
            
            if reset_flag == False:
                continue_to_run = True
            elif reset_flag == True:
                break
            
            # synchronize robot arm changes
            joint_states = visual_world.ros_listener.current_joint_values
            for rob_index in range(robot_num):
                rob_id = pw_T_rob_sim_pose_list_param[rob_index].obj_id
                pw_T_rob_sim_4_4 = pw_T_rob_sim_pose_list_param[rob_index].trans_matrix
                visual_world.set_real_robot_JointPosition(p_visual, rob_id, joint_states)
            
            test = False 
            if test == False:
                if display_gt_flag == True:
                    # print("display_gt_flag")
                    for obj_index in range(object_num):
                        obj_name = object_name_list[obj_index]
                        # display ground truth (grtu)
                        if visual_world.gazebo_flag == True:
                            # print("Hello")
                            # model_pose, model_pose_added_noise = visual_world.ros_listener.listen_2_object_pose(object_name_list[obj_index])
                           
                            # gazebo_T_obj_pos = model_pose[0]
                            # gazebo_T_obj_ori = model_pose[1]
                            # gazebo_T_obj_pos_added_noise = model_pose_added_noise[0]
                            # gazebo_T_obj_ori_added_noise = model_pose_added_noise[1]
                            # gazebo_T_rob_pos = panda_pose[0]
                            # gazebo_T_rob_ori = panda_pose[1]
                            
                            # opti_T_rob_opti_pos = copy.deepcopy(gazebo_T_rob_pos)
                            # opti_T_rob_opti_ori = copy.deepcopy(gazebo_T_rob_ori)
                            # opti_T_obj_opti_pos = copy.deepcopy(gazebo_T_obj_pos)
                            # opti_T_obj_opti_ori = copy.deepcopy(gazebo_T_obj_ori)
                            # opti_T_obj_obse_pos = copy.deepcopy(gazebo_T_obj_pos_added_noise)
                            # opti_T_obj_obse_ori = copy.deepcopy(gazebo_T_obj_ori_added_noise)
                            gt_name = ""
                            if dope_flag == True:
                                gt_name = "_gt"
                            obse_is_fresh = True
                            try:
                                latest_obse_time = listener_tf.getLatestCommonTime('/panda_link0', '/'+object_name_list[obj_index]+gt_name)
                                if (rospy.get_time() - latest_obse_time.to_sec()) < 0.1:
                                    (trans_gt,rot_gt) = listener_tf.lookupTransform('/panda_link0', '/'+object_name_list[obj_index]+gt_name, rospy.Time(0))
                                    obse_is_fresh = True
                                    # print("obse is FRESH")
                                else:
                                    # obse has not been updating for a while
                                    obse_is_fresh = False
                                    # print("obse is NOT fresh")
                                # break
                            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                                # print("can not find tf")
                                continue
                            rob_T_obj_opti_pos = list(trans_gt)
                            rob_T_obj_opti_ori = list(rot_gt)
                            rob_T_obj_opti_3_3 = transformations.quaternion_matrix(rob_T_obj_opti_ori)
                            rob_T_obj_opti_4_4 = rotation_4_4_to_transformation_4_4(rob_T_obj_opti_3_3, rob_T_obj_opti_pos)
                            # robpw_T_robga_4_4 = [[1., 0., 0.,    0.],
                            #                         [0., 1., 0.,    0.],
                            #                         [0., 0., 1., -0.06],
                            #                         [0., 0., 0.,    1.]]
                            # robpw_T_robga_4_4 = np.array(robpw_T_robga_4_4)                
                            # rob_T_obj_opti_4_4 = np.dot(robpw_T_robga_4_4, rob_T_obj_opti_4_4)
                            pandalink0_T_obj_obse_4_4_test = visual_world.ros_listener.listen_2_test_matrix()
                            # test_rob_T_obj_obse_4_4 = np.dot(robpw_T_robga_4_4, test_rob_T_obj_obse_4_4)
                            # print(object_name_list[obj_index]+": matrix from /gazebo/model_states:")
                            # print(pandalink0_T_obj_obse_4_4_test)
                            # print(object_name_list[obj_index]+": matrix from tf:")
                            # print(rob_T_obj_opti_4_4)
                        else:
                            # if optitrack_flag == True:
                            opti_T_rob_opti_pos = visual_world.ros_listener.listen_2_robot_pose()[0]
                            opti_T_rob_opti_ori = visual_world.ros_listener.listen_2_robot_pose()[1]
                            opti_T_obj_opti_pos = visual_world.ros_listener.listen_2_object_pose(object_name_list[obj_index])[0]
                            opti_T_obj_opti_ori = visual_world.ros_listener.listen_2_object_pose(object_name_list[obj_index])[1]
                            # get ground truth data 
                            rob_T_obj_opti_4_4 = compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, opti_T_obj_opti_pos, opti_T_obj_opti_ori)
                            # else:
                            #     optitrack_flag = optitrack_flag
                                
                        # init gt object
                        # if optitrack_flag == True:
                        if init_gt_obj_flag == 0:
                            if obj_index == object_num - 1:
                                init_gt_obj_flag = 1
                            
                            visual_world.init_display_ground_truth_object(pw_T_target_obj_opti_pose_lsit_param[obj_index])
                            
                        pw_T_obj_opti_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_obj_opti_4_4)
                        pw_T_obj_opti_pos = [pw_T_obj_opti_4_4[0][3], pw_T_obj_opti_4_4[1][3], pw_T_obj_opti_4_4[2][3]]
                        pw_T_obj_opti_ori = transformations.quaternion_from_matrix(pw_T_obj_opti_4_4)
                        # print("pw_T_obj_opti_pos:")
                        # print(pw_T_obj_opti_pos)
                        # display gt object update pose
                        print("=======================================")
                        print("obj_name:", obj_name)
                        print("pw_T_obj_opti_pos:", pw_T_obj_opti_pos)
                        print("pw_T_obj_opti_ori:", pw_T_obj_opti_ori)
                        print("=======================================")
                        pw_T_target_obj_opti_pose_lsit_param[obj_index].pos = pw_T_obj_opti_pos
                        pw_T_target_obj_opti_pose_lsit_param[obj_index].ori = pw_T_obj_opti_ori
                        visual_world.display_object_in_visual_model(p_visual, pw_T_target_obj_opti_pose_lsit_param[obj_index])
                        
                        # get observation data
                        # if visual_world.gazebo_flag == True:
                        #    obse_is_fresh = True
                        #    rob_T_obj_obse_4_4 = compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, opti_T_obj_obse_pos, opti_T_obj_obse_ori)
                        # else:

            if display_obse_flag == True:
                # print("display_obse_flag")
                for obj_index in range(object_num):
                    if init_obse_flag == 0:
                        if obj_index == object_num - 1:
                            init_obse_flag = 1
                        visual_world.init_display_observation_object(pw_T_target_obj_obse_pose_lsit_param[obj_index])
                    
                    use_gazebo = ""
                    if visual_world.gazebo_flag == True:
                        use_gazebo = '_noise'
                        if dope_flag == True:
                            use_gazebo  = ""
                    obse_is_fresh = True
                    try:
                        # latest_obse_time = listener_tf.getLatestCommonTime('/panda_link0', '/'+object_name_list[obj_index]+use_gazebo)
                        # if (rospy.get_time() - latest_obse_time.to_sec()) < 0.1:
                        (trans_ob,rot_ob) = listener_tf.lookupTransform('/panda_link0', '/'+object_name_list[obj_index]+use_gazebo, rospy.Time(0))
                        # (trans_ob,rot_ob) = listener_tf.lookupTransform('/panda_link0', '/'+object_name_list[obj_index]+use_gazebo, rospy.Time(0))
                        # (trans_ob_cTo, rot_ob_cTo) = listener_tf.lookupTransform('/camera_gt', '/'+object_name_list[obj_index]+use_gazebo, rospy.Time(0))
                        # (trans_ob_pTc, rot_ob_pTc) = listener_tf.lookupTransform('/panda_link0', '/camera_gt', rospy.Time(0))
                        # (trans_ob_cTo, rot_ob_cTo) = listener_tf.lookupTransform('/camera_gt', '/'+object_name_list[obj_index]+use_gazebo, rospy.Time(0))
                        
                        trans_ob_list[obj_index] = trans_ob
                        rot_ob_list[obj_index] = rot_ob
                        
                        obse_is_fresh = True
                            # print("obse is FRESH")
                        # else:
                            # obse has not been updating for a while
                            # obse_is_fresh = False
                            # print("obse is NOT fresh")
                        # break
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        print("from pbpf")
                        print("In Visualisation_World_GT.py: can not find "+object_name_list[obj_index]+" tf (obse)")
                    rob_T_obj_obse_pos = list(trans_ob_list[obj_index])
                    rob_T_obj_obse_ori = list(rot_ob_list[obj_index])
                    rob_T_obj_obse_3_3 = transformations.quaternion_matrix(rob_T_obj_obse_ori)
                    rob_T_obj_obse_4_4 = rotation_4_4_to_transformation_4_4(rob_T_obj_obse_3_3,rob_T_obj_obse_pos)
                    # print("rob_T_obj_obse_pos")
                    # print(rob_T_obj_obse_pos)
                    pw_T_obj_obse = np.dot(pw_T_rob_sim_4_4, rob_T_obj_obse_4_4)
                    pw_T_obj_obse_pos = [pw_T_obj_obse[0][3],pw_T_obj_obse[1][3],pw_T_obj_obse[2][3]]
                    pw_T_obj_obse_ori = transformations.quaternion_from_matrix(pw_T_obj_obse)
                    
                    # cam_T_obj_obse_pos = list(trans_ob_cTo)
                    # cam_T_obj_obse_ori = list(rot_ob_cTo)
                    # cam_T_obj_obse_3_3 = transformations.quaternion_matrix(cam_T_obj_obse_ori)
                    # cam_T_obj_obse_4_4 = rotation_4_4_to_transformation_4_4(cam_T_obj_obse_3_3, cam_T_obj_obse_pos)
                    # pan_T_cam_obse_pos = list(trans_ob_pTc)
                    # pan_T_cam_obse_ori = list(rot_ob_pTc)
                    # pan_T_cam_obse_3_3 = transformations.quaternion_matrix(pan_T_cam_obse_ori)
                    # pan_T_cam_obse_4_4 = rotation_4_4_to_transformation_4_4(pan_T_cam_obse_3_3, pan_T_cam_obse_pos)
                    
                    # ga_T_cam_pos = [1.227862, 0.39, 0.225166]
                    # ga_T_cam_ori = [0.7197831034103083, 1.2023881355993746e-07, 0.6941990233679357, -1.246701067867603e-07] # x,y,z,w
                    # ga_T_cam_3_3 = transformations.quaternion_matrix(ga_T_cam_ori)
                    # ga_T_cam_4_4 = rotation_4_4_to_transformation_4_4(ga_T_cam_3_3, ga_T_cam_pos)

                    # pan_T_obj_obse = np.dot(pan_T_cam_obse_4_4, cam_T_obj_obse_4_4)
                    # pw_T_obj_obse = np.dot(pw_T_rob_sim_4_4, pan_T_obj_obse)
                    # pw_T_obj_obse_pos = [pw_T_obj_obse[0][3],pw_T_obj_obse[1][3],pw_T_obj_obse[2][3]]
                    # pw_T_obj_obse_ori = transformations.quaternion_from_matrix(pw_T_obj_obse)
                    
                    # print(pw_T_obj_obse_pos)
                    # print(pw_T_obj_obse_ori)
                    # update pose
                    # print("pw_T_obj_obse_pos")
                    # print(pw_T_obj_obse_pos)
                    pw_T_target_obj_obse_pose_lsit_param[obj_index].pos = pw_T_obj_obse_pos
                    pw_T_target_obj_obse_pose_lsit_param[obj_index].ori = pw_T_obj_obse_ori
                    visual_world.display_object_in_visual_model(p_visual, pw_T_target_obj_obse_pose_lsit_param[obj_index])
                    # print(pw_T_obj_obse_pos)
            
            # display other objects
            # update other objects touching
            for obj_index in range(OBJS_TOUCHING_TARGET_OBJS_NUM):
                opti_T_rob_opti_pos = visual_world.ros_listener.listen_2_robot_pose()[0]
                opti_T_rob_opti_ori = visual_world.ros_listener.listen_2_robot_pose()[1]
                base_of_cheezit_pos = visual_world.ros_listener.listen_2_object_pose("base")[0]
                base_of_cheezit_ori = visual_world.ros_listener.listen_2_object_pose("base")[1]                
                robot_T_base = compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, base_of_cheezit_pos, base_of_cheezit_ori)
                pw_T_base = np.dot(pw_T_rob_sim_4_4, robot_T_base)
                pw_T_base_pos = [pw_T_base[0][3], pw_T_base[1][3], pw_T_base[2][3]]
                pw_T_base_ori = transformations.quaternion_from_matrix(pw_T_base)
                # update pose
                pw_T_other_obj_opti_pose_list_param[obj_index].pos = pw_T_base_pos
                pw_T_other_obj_opti_pose_list_param[obj_index].ori = pw_T_base_ori
                visual_world.display_object_in_visual_model(p_visual, pw_T_other_obj_opti_pose_list_param[obj_index])

            # update other objects not touching
            for obj_index in range(OBJS_ARE_NOT_TOUCHING_TARGET_OBJS_NUM):
                opti_T_rob_opti_pos = visual_world.ros_listener.listen_2_robot_pose()[0]
                opti_T_rob_opti_ori = visual_world.ros_listener.listen_2_robot_pose()[1]
                base_of_cheezit_pos = visual_world.ros_listener.listen_2_object_pose("pringles")[0]
                base_of_cheezit_ori = visual_world.ros_listener.listen_2_object_pose("pringles")[1]
                robot_T_base = compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, base_of_cheezit_pos, base_of_cheezit_ori)
                pw_T_base = np.dot(pw_T_rob_sim_4_4, robot_T_base)
                pw_T_base_pos = [pw_T_base[0][3], pw_T_base[1][3], pw_T_base[2][3]]
                pw_T_base_ori = transformations.quaternion_from_matrix(pw_T_base)
                print("pringles:")
                print(pw_T_base_pos)
                print(pw_T_base_ori)
                print("===============================")
                # update pose
                pw_T_objs_not_touching_targetObjs_list_param[obj_index].pos = pw_T_base_pos
                pw_T_objs_not_touching_targetObjs_list_param[obj_index].ori = pw_T_base_ori
                visual_world.display_object_in_visual_model(p_visual, pw_T_objs_not_touching_targetObjs_list_param[obj_index])

            # display particles
    #        if display_par_flag == True:
    #            particles_states_list = visual_world.ros_listener.listen_2_pars_states()
    #            print("particles_states_list:", particles_states_list)
    #            for obj_index in range(object_num):
    #                if len(particles_states_list.particles) == 0:
    #                    par_list_not_pub = 0
    #                    print("Do not publish particle information to /par_list")
    #                else:
    #                    if init_par_flag == 0:
    #                        if obj_index == object_num - 1:
    #                            init_par_flag = 1
    #                        for par_index in range(particle_num):
    #                            visual_world.init_display_particle(particles_states_list.particles[par_index].objects[obj_index])
    #                            obj_visual_id = particles_states_list.particles[par_index].objects[obj_index].id
    #                            par_obj_id[par_index].append(obj_visual_id)
    #                    else:
    #                        for par_index in range(particle_num):
    #                            particles_states_list.particles[par_index].objects[obj_index].id = par_obj_id[par_index][obj_index]
    #                            visual_world.display_particle_in_visual_model(particles_states_list.particles[par_index].objects[obj_index])
            if display_par_flag == True:
                particles_states_list = visual_world.ros_listener.listen_2_pars_states()
                if len(particles_states_list.particles) == 0:
                    par_list_not_pub = 0
                else:
                    if init_par_flag == 0:
                        for obj_index in range(object_num):
                            if obj_index == object_num - 1:
                                init_par_flag = 1
                            for par_index in range(particle_num):
                                visual_world.init_display_particle(particles_states_list.particles[par_index].objects[obj_index])
                                obj_visual_id = particles_states_list.particles[par_index].objects[obj_index].id
                                par_obj_id[par_index].append(obj_visual_id)
                    else:
                        for obj_index in range(object_num):
                            for par_index in range(particle_num):
                                particles_states_list.particles[par_index].objects[obj_index].id = par_obj_id[par_index][obj_index]
                                visual_world.display_particle_in_visual_model(particles_states_list.particles[par_index].objects[obj_index])
                        
            # display estimates object
            if display_esti_flag == True:
                for obj_index in range(object_num):
                    esti_obj_states_list = visual_world.ros_listener.listen_2_estis_states()

                    if len(esti_obj_states_list.objects) == 0:
                        esti_obj_list_not_pub = 1
                    else:
                        if init_esti_flag == 0:
                            if obj_index == object_num - 1:
                                init_esti_flag = 1
                            visual_world.init_display_estimated_object(esti_obj_states_list.objects[obj_index])
                            esti_obj_id[obj_index] = esti_obj_states_list.objects[obj_index].id
                        else:
                            esti_obj_states_list.objects[obj_index].id = esti_obj_id[obj_index]
                            visual_world.display_estimated_object_in_visual_model(esti_obj_states_list.objects[obj_index])
                        
            p_visual.stepSimulation()
          

    #par_obj_id = [[]*object_num for _ in range(particle_num)]
    #    esti_obj_id = [0] * object_num
        
        
        
        
