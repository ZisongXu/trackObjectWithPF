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
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3
from PBPF.msg import object_pose, particle_pose, particle_list, estimated_obj_pose
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
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
#from sksurgerycore.algorithms.averagequaternions import average_quaternions
from quaternion_averaging import weightedAverageQuaternions
from Particle import Particle
from Object_Pose import Object_Pose
import yaml

#Class of initialize the simulation model
class InitialSimulationModel():
    def __init__(self, object_num, robot_num, particle_num, 
                 pw_T_rob_sim_pose_list_alg, 
                 pw_T_obj_obse_obj_list_alg,
                 pw_T_objs_touching_targetObjs_list,
                 update_style_flag, sim_time_step):
        self.object_num = object_num
        self.robot_num = robot_num
        self.particle_num = particle_num
        self.pw_T_rob_sim_pose_list_alg = pw_T_rob_sim_pose_list_alg
        self.pw_T_obj_obse_obj_list_alg = pw_T_obj_obse_obj_list_alg
        self.pw_T_objs_touching_targetObjs_list = pw_T_objs_touching_targetObjs_list
        self.update_style_flag = update_style_flag
        self.sim_time_step = sim_time_step
        
        self.particle_cloud = []
        self.esti_objs_cloud = []
        self.pybullet_particle_env_collection = []
        self.fake_robot_id_collection = []
        self.other_object_id_collection = []
        self.particle_no_visual_id_collection = []
        self.env_fix_obj_id_collection = []
        
        self.particle_cloud_CV = []
        self.pybullet_particle_env_collection_CV = []
        self.particle_no_visual_id_collection_CV = []
        
        self.boss_sigma_obs_pos_init = 0.08 # original value: 16cm/10CM 
        
        self.boss_sigma_obs_x = self.boss_sigma_obs_pos_init / math.sqrt(2)
        self.boss_sigma_obs_y = self.boss_sigma_obs_pos_init / math.sqrt(2)
        self.boss_sigma_obs_z = 0.02
        self.boss_sigma_obs_ang_init = 0.0216773873 * 10 # original value: 0.0216773873 * 20
        
        # mark
        # self.boss_sigma_obs_x = 0
        # self.boss_sigma_obs_y = 0
        # self.boss_sigma_obs_z = 0
        # self.boss_sigma_obs_ang_init = 0


        # self.boss_sigma_obs_ang_init = 0.0216773873 * 1
        
        with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
            self.parameter_info = yaml.safe_load(file)
        self.gazebo_flag = self.parameter_info['gazebo_flag']
        self.task_flag = self.parameter_info['task_flag']
        self.SIM_REAL_WORLD_FLAG = self.parameter_info['sim_real_world_flag']
        self.SHOW_RAY = self.parameter_info['show_ray'] 
        self.VK_RENDER_FLAG = self.parameter_info['vk_render_flag'] 

        self.OBJS_ARE_NOT_TOUCHING_TARGET_OBJS_NUM = self.parameter_info['objs_are_not_touching_target_objs_num']
        self.OBJS_TOUCHING_TARGET_OBJS_NUM = self.parameter_info['objs_touching_target_objs_num']

        
    def generate_random_pose(self, pw_T_obj_obse_pos, pw_T_obj_obse_ori):
        position = copy.deepcopy(pw_T_obj_obse_pos)
        quat = copy.deepcopy(pw_T_obj_obse_ori)#x,y,z,w
        quat_QuatStyle = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3])#w,x,y,z
        x = self.add_noise_to_init_par(position[0], self.boss_sigma_obs_x)
        y = self.add_noise_to_init_par(position[1], self.boss_sigma_obs_y)
        z = self.add_noise_to_init_par(position[2], self.boss_sigma_obs_z)
        random_dir = random.uniform(0, 2*math.pi)
        z_axis = random.uniform(-1,1)
        x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
        y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
        angle_noise = self.add_noise_to_init_par(0, self.boss_sigma_obs_ang_init)
        w_quat = math.cos(angle_noise/2.0)
        x_quat = math.sin(angle_noise/2.0) * x_axis
        y_quat = math.sin(angle_noise/2.0) * y_axis
        z_quat = math.sin(angle_noise/2.0) * z_axis
        ###nois_quat(w,x,y,z); new_quat(w,x,y,z)
        nois_quat = Quaternion(x=x_quat, y=y_quat, z=z_quat, w=w_quat)
        new_quat = nois_quat * quat_QuatStyle
        ###pb_quat(x,y,z,w)
        pb_quat = [new_quat[1], new_quat[2], new_quat[3], new_quat[0]]
        return [x, y, z], pb_quat
    
    
    def compute_estimate_pos_of_object(self, particle_cloud): # need to change
        for i in range(self.object_num):
            x_set = 0
            y_set = 0
            z_set = 0
            w_set = 0
            quaternions = []
            qws = []
            for index,particle in enumerate(particle_cloud):
                x_set = x_set + particle[i].pos[0] * particle[i].w
                y_set = y_set + particle[i].pos[1] * particle[i].w
                z_set = z_set + particle[i].pos[2] * particle[i].w
                q = self.quaternion_correction(particle[i].ori)
                qws.append(particle[i].w)
                quaternions.append([q[0], q[1], q[2], q[3]]) # x,y,z,w
                w_set = w_set + particle[i].w
            q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
            est_obj_pose = Object_Pose(particle[i].par_name, 0, [x_set/w_set, y_set/w_set, z_set/w_set], [q[0], q[1], q[2], q[3]], i)
            self.esti_objs_cloud.append(est_obj_pose)
        return self.esti_objs_cloud

    def initial_and_set_simulation_env(self):
        print_name_flag = 0
        PBPF_par_no_visual_id = [[]*self.object_num for _ in range(self.particle_num)]
        p_env_list = []
        shelves_id_list = []
        other_obj_id_list = []

        for par_index in range(self.particle_num):
            
            collision_detection_obj_id = []
            if self.SHOW_RAY == True:
                pybullet_simulation_env = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT,GUI_SERVER
            else:
                pybullet_simulation_env = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT,GUI_SERVER
            p_env_list.append(pybullet_simulation_env)
            self.pybullet_particle_env_collection.append(pybullet_simulation_env)
            if self.update_style_flag == "time":
                pybullet_simulation_env.setTimeStep(self.sim_time_step)
            pybullet_simulation_env.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85, cameraTargetPosition=[0.5, 0.3, 0.2])
            pybullet_simulation_env.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_simulation_env.setGravity(0, 0, -9.81)
            fake_plane_id = pybullet_simulation_env.loadURDF("plane.urdf")

            if self.task_flag == "1":
                pw_T_pringles_pos = [0.6652218209791124, 0.058946644391304814, 0.8277292172960276]
                pw_T_pringles_ori = [ 0.67280124, -0.20574896, -0.20600051, 0.68012472] # x, y, z, w
                pringles_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/pringles.urdf"),
                                                              pw_T_pringles_pos,
                                                              pw_T_pringles_ori,
                                                              useFixedBase=1)

            if self.task_flag == "5":
                pw_T_she_pos = [0.75889274, -0.24494845, 0.33818097+0.02]
                pw_T_she_ori = [0, 0, 0, 1]
                shelves_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/shelves.urdf"),
                                                              pw_T_she_pos,
                                                              pw_T_she_ori,
                                                              useFixedBase=1)
                collision_detection_obj_id.append(shelves_id)
                shelves_id_list.append(shelves_id)

            if self.task_flag == "4":
                for obj_index in range(self.OBJS_TOUCHING_TARGET_OBJS_NUM):
                    other_obj_name = self.pw_T_objs_touching_targetObjs_list[obj_index].obj_name
                    other_obj_pos = self.pw_T_objs_touching_targetObjs_list[obj_index].pos
                    other_obj_ori = self.pw_T_objs_touching_targetObjs_list[obj_index].ori
                    sim_base_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/base_of_cracker/base_of_cracker.urdf"),
                                                                other_obj_pos,
                                                                other_obj_ori,
                                                                useFixedBase=1)
                    collision_detection_obj_id.append(sim_base_id)   
                    other_obj_id_list.append(sim_base_id)

            # mark
            for rob_index in range(self.robot_num):
                real_robot_start_pos = self.pw_T_rob_sim_pose_list_alg[rob_index].pos
                real_robot_start_ori = self.pw_T_rob_sim_pose_list_alg[rob_index].ori
                joint_of_robot = self.pw_T_rob_sim_pose_list_alg[rob_index].joints
                fake_robot_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                                                 real_robot_start_pos,
                                                                 real_robot_start_ori,
                                                                 useFixedBase=1)
                self.set_sim_robot_JointPosition(pybullet_simulation_env, fake_robot_id, joint_of_robot)
                self.fake_robot_id_collection.append(fake_robot_id)
            # need to change
            collision_detection_obj_id.append(fake_robot_id)
            
            if self.SIM_REAL_WORLD_FLAG == True:
                table_pos_1 = [0.46, -0.01, 0.710]
                table_ori_1 = pybullet_simulation_env.getQuaternionFromEuler([0,0,0])
                table_id_1 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/table.urdf"), table_pos_1, table_ori_1)

                barry_pos_1 = [-0.694, 0.443, 0.895]
                barry_ori_1 = pybullet_simulation_env.getQuaternionFromEuler([0,math.pi/2,0])
                barry_id_1 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_1, barry_ori_1, useFixedBase = 1)
                
                barry_pos_2 = [-0.694, -0.607, 0.895]
                barry_ori_2 = pybullet_simulation_env.getQuaternionFromEuler([0,math.pi/2,0])
                barry_id_2 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_2, barry_ori_2, useFixedBase = 1)

                barry_pos_3 = [0.459, -0.972, 0.895]
                barry_ori_3 = pybullet_simulation_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
                barry_id_3 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_3, barry_ori_3, useFixedBase = 1)

                # barry_pos_4 = [-0.549, 0.61, 0.895]
                # barry_ori_4 = pybullet_simulation_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
                # barry_id_4 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_4, barry_ori_4, useFixedBase = 1)
                
                # barry_pos_5 = [0.499, 0.61, 0.895]
                # barry_ori_5 = pybullet_simulation_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
                # barry_id_5 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_5, barry_ori_5, useFixedBase = 1)

                board_pos_1 = [0.274, 0.581, 0.87575]
                board_ori_1 = pybullet_simulation_env.getQuaternionFromEuler([math.pi/2,math.pi/2,0])
                board_id_1 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/board.urdf"), board_pos_1, board_ori_1, useFixedBase = 1)

                self.env_fix_obj_id_collection.append(board_id_1)
                collision_detection_obj_id.append(board_id_1)

            object_list = []
            # mark
            bias_obse_x = 0
            bias_obse_y = 0
            bias_obse_z = 0
            for obj_index in range(self.object_num):
                obj_obse_pos = self.pw_T_obj_obse_obj_list_alg[obj_index].pos
                obj_obse_pos = [obj_obse_pos[0]+bias_obse_x, obj_obse_pos[1]+bias_obse_y, obj_obse_pos[2]+bias_obse_z]
                obj_obse_ori = self.pw_T_obj_obse_obj_list_alg[obj_index].ori
                obj_obse_name = self.pw_T_obj_obse_obj_list_alg[obj_index].obj_name

                if print_name_flag == obj_index:
                    print_name_flag = print_name_flag + 1
                    print("Generate particles for the target object:")
                    print(obj_obse_name)
                particle_pos, particle_ori = self.generate_random_pose(obj_obse_pos, obj_obse_ori)
                gazebo_contain = ""
                if self.gazebo_flag == True:
                    gazebo_contain = "gazebo_"
                particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/"+gazebo_contain+obj_obse_name+"/"+gazebo_contain+obj_obse_name+"_par_no_visual_hor.urdf"),
                                                                         particle_pos,
                                                                         particle_ori)
                collision_detection_obj_id.append(particle_no_visual_id)
                conter = 0
                while True:
                    flag = 0
                    conter = conter + 1
                    length_collision_detection_obj_id = len(collision_detection_obj_id)
                    for check_num in range(length_collision_detection_obj_id-1):
                        pybullet_simulation_env.stepSimulation()
                        contacts = pybullet_simulation_env.getContactPoints(bodyA=collision_detection_obj_id[check_num], bodyB=collision_detection_obj_id[-1])
                        for contact in contacts:
                            contactNormalOnBtoA = contact[7]
                            contact_dis = contact[8]
                            # print(contact_dis)
                            # print(contactNormalOnBtoA)
                            # input("stop")
                            if contact_dis < -0.001:
                                par_x_ = particle_pos[0] + contactNormalOnBtoA[0]*contact_dis/2
                                par_y_ = particle_pos[1] + contactNormalOnBtoA[1]*contact_dis/2
                                par_z_ = particle_pos[2] + contactNormalOnBtoA[2]*contact_dis/2
                                particle_pos = [par_x_, par_y_, par_z_]
                                if conter > 20:
                                    print("init more than 20 times")
                                    conter = 0
                                    particle_pos, particle_ori = self.generate_random_pose(obj_obse_pos, obj_obse_ori)
                                pybullet_simulation_env.resetBasePositionAndOrientation(particle_no_visual_id, particle_pos, particle_ori)
                                flag = 1
                                break
                        if flag == 1:
                            break
                    if flag == 0:
                        break



                objPose = Particle(obj_obse_name, 0, particle_no_visual_id, particle_pos, particle_ori, 1/self.particle_num, par_index, 0, 0)
                object_list.append(objPose)
                PBPF_par_no_visual_id[par_index].append(particle_no_visual_id)

            self.particle_cloud.append(object_list)
            self.particle_no_visual_id_collection = copy.deepcopy(PBPF_par_no_visual_id)
        
        if len(shelves_id_list) != 0:
            self.other_object_id_collection.append(shelves_id_list)
        if len(other_obj_id_list) != 0:
            self.other_object_id_collection.append(other_obj_id_list)        
        print(self.other_object_id_collection)


        esti_objs_cloud_temp_parameter = self.compute_estimate_pos_of_object(self.particle_cloud)
        return esti_objs_cloud_temp_parameter, self.particle_cloud, p_env_list
        
    def initial_and_set_simulation_env_CV(self):
        CVPF_par_no_visual_id = [[]*self.object_num for _ in range(self.particle_num)]
        p_env_list = []
        shelves_id_list = []
        other_obj_id_list = []

        for par_index in range(self.particle_num):
            collision_detection_obj_id = []
            pybullet_simulation_env = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT,GUI_SERVER
            p_env_list.append(pybullet_simulation_env)
            self.pybullet_particle_env_collection_CV.append(pybullet_simulation_env)
            pybullet_simulation_env.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85, cameraTargetPosition=[0.5, 0.3, 0.2])
            pybullet_simulation_env.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_simulation_env.setGravity(0, 0, -9.81)
            fake_plane_id = pybullet_simulation_env.loadURDF("plane.urdf")
            
            if self.task_flag == "1":
                pw_T_pringles_pos = [0.6646962577067851, 0.059950902446081644, 0.8269063881730724]
                pw_T_pringles_ori = [ 0.69423769, -0.11003228, -0.11355051,  0.70216323] # x, y, z, w
                pringles_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/pringles.urdf"),
                                                              pw_T_pringles_pos,
                                                              pw_T_pringles_ori,
                                                              useFixedBase=1)

            if self.task_flag == "5":
                pw_T_she_pos = [0.75889274, -0.24494845, 0.33818097+0.02]
                pw_T_she_ori = [0, 0, 0, 1]
                shelves_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/shelves.urdf"),
                                                              pw_T_she_pos,
                                                              pw_T_she_ori,
                                                              useFixedBase=1)
                collision_detection_obj_id.append(shelves_id)
                shelves_id_list.append(shelves_id)
            if self.task_flag == "4":
                for obj_index in range(self.OBJS_TOUCHING_TARGET_OBJS_NUM):
                    other_obj_name = self.pw_T_objs_touching_targetObjs_list[obj_index].obj_name
                    other_obj_pos = self.pw_T_objs_touching_targetObjs_list[obj_index].pos
                    other_obj_ori = self.pw_T_objs_touching_targetObjs_list[obj_index].ori
                    sim_base_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/"+other_obj_name+"/base_of_cracker.urdf"),
                                                                other_obj_pos,
                                                                other_obj_ori,
                                                                useFixedBase=1)
                    collision_detection_obj_id.append(sim_base_id)                
                    other_obj_id_list.append(sim_base_id)
                
            for rob_index in range(self.robot_num):
                real_robot_start_pos = self.pw_T_rob_sim_pose_list_alg[rob_index].pos
                real_robot_start_ori = self.pw_T_rob_sim_pose_list_alg[rob_index].ori
                joint_of_robot = self.pw_T_rob_sim_pose_list_alg[rob_index].joints
                fake_robot_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                                                 real_robot_start_pos,
                                                                 real_robot_start_ori,
                                                                 useFixedBase=1)
                self.set_sim_robot_JointPosition(pybullet_simulation_env, fake_robot_id, joint_of_robot)
                self.fake_robot_id_collection.append(fake_robot_id)

            collision_detection_obj_id.append(fake_robot_id)

            if self.SIM_REAL_WORLD_FLAG == True:
                table_pos_1 = [0.46, -0.01, 0.710]
                table_ori_1 = pybullet_simulation_env.getQuaternionFromEuler([0,0,0])
                table_id_1 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/table.urdf"), table_pos_1, table_ori_1)

                barry_pos_1 = [-0.694, 0.443, 0.895]
                barry_ori_1 = pybullet_simulation_env.getQuaternionFromEuler([0,math.pi/2,0])
                barry_id_1 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_1, barry_ori_1, useFixedBase = 1)
                
                barry_pos_2 = [-0.694, -0.607, 0.895]
                barry_ori_2 = pybullet_simulation_env.getQuaternionFromEuler([0,math.pi/2,0])
                barry_id_2 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_2, barry_ori_2, useFixedBase = 1)

                barry_pos_3 = [0.459, -0.972, 0.895]
                barry_ori_3 = pybullet_simulation_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
                barry_id_3 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_3, barry_ori_3, useFixedBase = 1)

                # barry_pos_4 = [-0.549, 0.61, 0.895]
                # barry_ori_4 = pybullet_simulation_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
                # barry_id_4 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_4, barry_ori_4, useFixedBase = 1)
                
                # barry_pos_5 = [0.499, 0.61, 0.895]
                # barry_ori_5 = pybullet_simulation_env.getQuaternionFromEuler([0,math.pi/2,math.pi/2])
                # barry_id_5 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/barrier.urdf"), barry_pos_5, barry_ori_5, useFixedBase = 1)

                board_pos_1 = [0.274, 0.581, 0.87575]
                board_ori_1 = pybullet_simulation_env.getQuaternionFromEuler([math.pi/2,math.pi/2,0])
                board_id_1 = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/others/board.urdf"), board_pos_1, board_ori_1, useFixedBase = 1)

                self.env_fix_obj_id_collection.append(board_id_1)
                collision_detection_obj_id.append(board_id_1)

            object_list = []

            for obj_index in range(self.object_num):
                obj_obse_pos = self.pw_T_obj_obse_obj_list_alg[obj_index].pos
                obj_obse_ori = self.pw_T_obj_obse_obj_list_alg[obj_index].ori
                obj_obse_name = self.pw_T_obj_obse_obj_list_alg[obj_index].obj_name
                particle_pos, particle_ori = self.generate_random_pose(obj_obse_pos, obj_obse_ori)
                gazebo_contain = ""
                if self.gazebo_flag == True:
                    gazebo_contain = "gazebo_"
                particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/"+gazebo_contain+obj_obse_name+"/"+gazebo_contain+obj_obse_name+"_par_no_visual_hor.urdf"),
                                                                         particle_pos,
                                                                         particle_ori)
                collision_detection_obj_id.append(particle_no_visual_id)
                conter = 0
                while True:
                    flag = 0
                    conter = conter + 1
                    length_collision_detection_obj_id = len(collision_detection_obj_id)
                    for check_num in range(length_collision_detection_obj_id-1):
                        pybullet_simulation_env.stepSimulation()
                        contacts = pybullet_simulation_env.getContactPoints(bodyA=collision_detection_obj_id[check_num], bodyB=collision_detection_obj_id[-1])
                        for contact in contacts:
                            contactNormalOnBtoA = contact[7]
                            contact_dis = contact[8]
                            if contact_dis < -0.001:
                                par_x_ = particle_pos[0] + contactNormalOnBtoA[0]*contact_dis/2
                                par_y_ = particle_pos[1] + contactNormalOnBtoA[1]*contact_dis/2
                                par_z_ = particle_pos[2] + contactNormalOnBtoA[2]*contact_dis/2
                                particle_pos = [par_x_, par_y_, par_z_]
                                if conter > 20:
                                    print("init more than 20 times")
                                    conter = 0
                                    particle_pos, particle_ori = self.generate_random_pose(obj_obse_pos, obj_obse_ori)
                                # particle_pos, particle_ori = self.generate_random_pose(obj_obse_pos, obj_obse_ori)
                                pybullet_simulation_env.resetBasePositionAndOrientation(particle_no_visual_id, particle_pos, particle_ori)
                                flag = 1
                                break
                        if flag == 1:
                            break
                    if flag == 0:
                        break

                objPose = Particle(obj_obse_name, 0, particle_no_visual_id, particle_pos, particle_ori, 1/self.particle_num, par_index, 0, 0)
                object_list.append(objPose)
                CVPF_par_no_visual_id[par_index].append(particle_no_visual_id)
                
            self.particle_cloud_CV.append(object_list)
            self.particle_no_visual_id_collection_CV = copy.deepcopy(CVPF_par_no_visual_id)

        if len(shelves_id_list) != 0:
            self.other_object_id_collection.append(shelves_id_list)
        if len(other_obj_id_list) != 0:
            self.other_object_id_collection.append(other_obj_id_list) 
        

        esti_objs_cloud_temp_parameter = self.compute_estimate_pos_of_object(self.particle_cloud_CV)
        return esti_objs_cloud_temp_parameter, self.particle_cloud_CV, p_env_list

    def set_sim_robot_JointPosition(self, pybullet_simulation_env, robot, position):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                pybullet_simulation_env.resetJointState(robot,
                                                joint_index+2,
                                                targetValue=position[joint_index])
            else:
                pybullet_simulation_env.resetJointState(robot,
                                                joint_index,
                                                targetValue=position[joint_index])
                
    def add_noise_to_init_par(self, current_pos, sigma_init):
        mean = current_pos
        sigma = sigma_init
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def take_easy_gaussian_value(self, mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal
    
    # make sure all quaternions all between -pi and +pi
    def quaternion_correction(self, quaternion): # x,y,z,w
        new_quat = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]) # w,x,y,z
        cos_theta_over_2 = new_quat.w
        sin_theta_over_2 = math.sqrt(new_quat.x ** 2 + new_quat.y ** 2 + new_quat.z ** 2)
        theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
        theta = theta_over_2 * 2
        if theta >= math.pi or theta <= -math.pi:
            new_quaternion = [-quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
            return new_quaternion
        return quaternion
