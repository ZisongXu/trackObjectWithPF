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
from Object_Pose import Object_Pose
#Class of initialize the simulation model
class InitialSimulationModel():
    def __init__(self, object_num, robot_num, other_obj_num, particle_num, 
                 pw_T_rob_sim_pose_list,                            
                 pw_T_obj_obse_objects_list,
                 pw_T_other_obj_opti_pose_list,
                 p_visualisation,
                 update_style_flag, change_sim_time, task_flag, object_flag):
        self.object_num = object_num
        self.robot_num = robot_num
        self.other_obj_num = other_obj_num
        self.particle_num = particle_num
        self.pw_T_rob_sim_pose_list = pw_T_rob_sim_pose_list
        self.pw_T_obj_obse_objects_list = pw_T_obj_obse_objects_list
        self.pw_T_other_obj_opti_pose_list = pw_T_other_obj_opti_pose_list
        self.p_visualisation = p_visualisation
        self.update_style_flag = update_style_flag
        self.change_sim_time = change_sim_time
        self.task_flag = task_flag
        self.object_flag = object_flag
        
        self.particle_cloud = []
        self.esti_objs_cloud = []
        self.pybullet_particle_env_collection = []
        self.fake_robot_id_collection = []
        self.particle_no_visual_id_collection = []
        
        self.particle_cloud_CV = []
        self.pybullet_particle_env_collection_CV = []
        self.particle_no_visual_id_collection_CV = []
        
        self.boss_sigma_obs_x = 0.032860982 * 2.0
        self.boss_sigma_obs_y = 0.012899399 * 1.5
        self.boss_sigma_obs_z = 0.01
        self.boss_sigma_obs_ang_init = 0.0216773873 * 2.0
        
        
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


    def display_particle(self):
        for index, particle in enumerate(self.particle_cloud):
            for obj_index in range(self.object_num):
                obj_par_name = particle[obj_index].par_name
                obj_par_pos = particle[obj_index].pos
                obj_par_ori = particle[obj_index].ori
                visualize_particle_Id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+obj_par_name+"/"+obj_par_name+"_par_with_visual_PB_hor.urdf"),
                                                                      obj_par_pos,
                                                                      obj_par_ori)
                particle[obj_index].visual_par_id = visualize_particle_Id
            
            
    def display_particle_CV(self):
        for index, particle in enumerate(self.particle_cloud_CV):
            for obj_index in range(self.object_num):
                obj_par_name = particle[obj_index].par_name
                obj_par_pos = particle[obj_index].pos
                obj_par_ori = particle[obj_index].ori
                visualize_particle_Id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+obj_par_name+"/"+obj_par_name+"_par_with_visual_CV_hor.urdf"),
                                                                      obj_par_pos,
                                                                      obj_par_ori)
                particle[obj_index].visual_par_id = visualize_particle_Id


    def initial_and_set_simulation_env(self):
        PBPF_par_no_visual_id = [[]*self.object_num for _ in range(self.particle_num)]
        for par_index in range(self.particle_num):
            collision_detection_obj_id = []
            pybullet_simulation_env = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT,GUI_SERVER
            self.pybullet_particle_env_collection.append(pybullet_simulation_env)
            if self.update_style_flag == "time":
                pybullet_simulation_env.setTimeStep(self.change_sim_time)
            pybullet_simulation_env.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85, cameraTargetPosition=[0.5, 0.3, 0.2])
            pybullet_simulation_env.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_simulation_env.setGravity(0,0,-9.81)
            fake_plane_id = pybullet_simulation_env.loadURDF("plane.urdf")
            
            for obj_index in range(self.other_obj_num):
                other_obj_name = self.pw_T_other_obj_opti_pose_list[obj_index].obj_name
                other_obj_pos = self.pw_T_other_obj_opti_pose_list[obj_index].pos
                other_obj_ori = self.pw_T_other_obj_opti_pose_list[obj_index].ori
                sim_base_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/"+other_obj_name+"/base_of_cracker.urdf"),
                                                               other_obj_pos,
                                                               other_obj_ori,
                                                               useFixedBase=1)
                
            for rob_index in range(self.robot_num):
                real_robot_start_pos = self.pw_T_rob_sim_pose_list[rob_index].pos
                real_robot_start_ori = self.pw_T_rob_sim_pose_list[rob_index].ori
                joint_of_robot = self.pw_T_rob_sim_pose_list[rob_index].joints
                fake_robot_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                                                 real_robot_start_pos,
                                                                 real_robot_start_ori,
                                                                 useFixedBase=1)
                self.set_sim_robot_JointPosition(pybullet_simulation_env, fake_robot_id, joint_of_robot)
            self.fake_robot_id_collection.append(fake_robot_id)
            collision_detection_obj_id.append(fake_robot_id)
            particle_list = []
            for obj_index in range(self.object_num):
                obj_obse_pos = self.pw_T_obj_obse_objects_list[obj_index].pos
                obj_obse_ori = self.pw_T_obj_obse_objects_list[obj_index].ori
                obj_obse_name = self.pw_T_obj_obse_objects_list[obj_index].obj_name
                particle_pos, particle_ori = self.generate_random_pose(obj_obse_pos, obj_obse_ori)
                particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/"+obj_obse_name+"/"+obj_obse_name+"_par_no_visual_hor.urdf"),
                                                                         particle_pos,
                                                                         particle_ori)
                collision_detection_obj_id.append(particle_no_visual_id)
                while True:
                    flag = 0
                    for check_num in range(obj_index+1):
                        pybullet_simulation_env.stepSimulation()
                        contacts = pybullet_simulation_env.getContactPoints(bodyA=collision_detection_obj_id[check_num], bodyB=collision_detection_obj_id[-1])
                        for contact in contacts:
                            contact_dis = contact[8]
                            if contact_dis < -0.001:
                                particle_pos, particle_ori = self.generate_random_pose(obj_obse_pos, obj_obse_ori)
                                pybullet_simulation_env.resetBasePositionAndOrientation(particle_no_visual_id, particle_pos, particle_ori)
                                flag = 1
                                break
                        if flag == 1:
                            break
                    if flag == 0:
                        break
    
                objPose = Particle(obj_obse_name, 0, particle_no_visual_id, particle_pos, particle_ori, 1/self.particle_num, par_index, 0, 0)
                particle_list.append(objPose)
                PBPF_par_no_visual_id[par_index].append(particle_no_visual_id)
            self.particle_cloud.append(particle_list)
            self.particle_no_visual_id_collection = copy.deepcopy(PBPF_par_no_visual_id)

        esti_objs_cloud_temp_parameter = self.compute_estimate_pos_of_object(self.particle_cloud)
        return esti_objs_cloud_temp_parameter
        
    
    def initial_and_set_simulation_env_CV(self):
        self.particle_cloud_CV = copy.deepcopy(self.particle_cloud)
        for index, particle in enumerate(self.particle_cloud_CV):
            pybullet_simulation_env = bc.BulletClient(connection_mode=p.DIRECT) # GUI_SERVER, DIRECT
            self.pybullet_particle_env_collection_CV.append(pybullet_simulation_env)
            pybullet_simulation_env.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_simulation_env.setGravity(0, 0, -9.81)
            fake_plane_id = pybullet_simulation_env.loadURDF("plane.urdf")
            for obj_index in range(self.object_num):
                obj_par_pos = particle[obj_index].pos
                obj_par_ori = particle[obj_index].ori
                obj_par_name = particle[obj_index].par_name
                particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/"+obj_par_name+"/"+obj_par_name+"_par_no_visual_hor.urdf"),
                                                                         obj_par_pos,
                                                                         obj_par_ori)
                particle[obj_index].no_visual_par_id = particle_no_visual_id
#            self.particle_no_visual_id_collection_CV.append(particle_no_visual_id)
        esti_objs_cloud_temp_parameter = self.compute_estimate_pos_of_object(self.particle_cloud_CV)
        return esti_objs_cloud_temp_parameter


    def set_sim_robot_JointPosition(self,pybullet_simulation_env,robot, position):
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
