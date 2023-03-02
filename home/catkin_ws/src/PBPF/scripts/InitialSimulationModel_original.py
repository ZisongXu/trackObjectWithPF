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
#Class of initialize the simulation model
class InitialSimulationModel():
    def __init__(self,particle_num,real_robot_start_pos,real_robot_start_ori,noise_obj_pos,noise_obj_ang,pw_T_object_ori_dope):
        self.particle_num = particle_num
        self.noise_obj_pos = noise_obj_pos
        self.noise_obj_ang = noise_obj_ang
        self.real_robot_start_pos = real_robot_start_pos
        self.real_robot_start_ori = real_robot_start_ori
        self.pw_T_object_ori_dope = pw_T_object_ori_dope
        self.particle_cloud = []
        self.pybullet_particle_env_collection = []
        self.fake_robot_id_collection = []
        self.particle_no_visual_id_collection = []
        self.particle_with_visual_id_collection =[]
        self.noise_object_pose = []

        self.particle_cloud_PM = []
        self.pybullet_particle_env_collection_PM = []
        self.particle_no_visual_id_collection_PM = []
        self.particle_with_visual_id_collection_PM =[]

    def initial_particle(self):
        noise_obj_x = copy.deepcopy(self.noise_obj_pos[0])
        noise_obj_y = copy.deepcopy(self.noise_obj_pos[1])
        noise_obj_z = copy.deepcopy(self.noise_obj_pos[2])
        noise_obj_pos = [noise_obj_x,noise_obj_y,noise_obj_z]
        noise_obj_x_ang = copy.deepcopy(self.noise_obj_ang[0])
        noise_obj_y_ang = copy.deepcopy(self.noise_obj_ang[1])
        noise_obj_z_ang = copy.deepcopy(self.noise_obj_ang[2])
        noise_obj_ang = [noise_obj_x_ang,noise_obj_y_ang,noise_obj_z_ang]

        self.noise_object_pose = [noise_obj_x,noise_obj_y,noise_obj_z,noise_obj_x_ang,noise_obj_y_ang,noise_obj_z_ang]

        for i in range(self.particle_num):
            x,y,z,x_angle,y_angle,z_angle,new_quat = self.generate_random_pose(self.noise_object_pose,self.pw_T_object_ori_dope)
            w = 1/self.particle_num
            particle = Particle(x,y,z,x_angle,y_angle,z_angle,w,index=i)
            self.particle_cloud.append(particle)

    def generate_random_pose(self,noise_object_pose, pw_T_object_ori_dope):
        angle = copy.deepcopy([noise_object_pose[3],noise_object_pose[4],noise_object_pose[5]])
        quat = copy.deepcopy(pw_T_object_ori_dope)#x,y,z,w
        quat_QuatStyle = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3])#w,x,y,z
        x = self.add_noise_to_init_par(noise_object_pose[0],boss_sigma_obs_x)
        y = self.add_noise_to_init_par(noise_object_pose[1],boss_sigma_obs_y)
        z = self.add_noise_to_init_par(noise_object_pose[2],boss_sigma_obs_z)
        random_dir = random.uniform(0, 2*math.pi)
        z_axis = random.uniform(-1,1)
        x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
        y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
        angle_noise = self.add_noise_to_init_par(0,boss_sigma_obs_ang_init)
        w_quat = math.cos(angle_noise/2.0)
        x_quat = math.sin(angle_noise/2.0) * x_axis
        y_quat = math.sin(angle_noise/2.0) * y_axis
        z_quat = math.sin(angle_noise/2.0) * z_axis
        ###nois_quat(w,x,y,z); new_quat(w,x,y,z)
        nois_quat = Quaternion(x=x_quat,y=y_quat,z=z_quat,w=w_quat)
        new_quat = nois_quat * quat_QuatStyle
        ###pb_quat(x,y,z,w)
        pb_quat = [new_quat[1],new_quat[2],new_quat[3],new_quat[0]]
        new_angle = p_visualisation.getEulerFromQuaternion(pb_quat)
        x_angle = new_angle[0]
        y_angle = new_angle[1]
        z_angle = new_angle[2]
        return x,y,z,x_angle,y_angle,z_angle,pb_quat
    def compute_estimate_pos_of_object(self, particle_cloud):
        x_set = 0
        y_set = 0
        z_set = 0
        w_set = 0

        quaternions = []
        qws = []
        for index,particle in enumerate(particle_cloud):
            x_set = x_set + particle.x * particle.w
            y_set = y_set + particle.y * particle.w
            z_set = z_set + particle.z * particle.w
            q = p_visualisation.getQuaternionFromEuler([particle.x_angle, particle.y_angle, particle.z_angle])
            qws.append(particle.w)
            quaternions.append([q[0], q[1], q[2], q[3]])
            w_set = w_set + particle.w

        # q = average_quaternions(np.array(quaternions))
        q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
        x_angle, y_angle, z_angle = p_visualisation.getEulerFromQuaternion([q[0], q[1], q[2], q[3]])

        return x_set/w_set,y_set/w_set,z_set/w_set,x_angle,y_angle,z_angle


    def display_particle(self):
        for index, particle in enumerate(self.particle_cloud):
            visualize_particle_pos = [particle.x, particle.y, particle.z]
            visualize_particle_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            visualize_particle_orientation = p_visualisation.getQuaternionFromEuler(visualize_particle_angle)
            if object_cracker_flag == True:
                visualize_particle_Id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cube/cheezit_par_with_visual_small_PE_hor.urdf"),
                                                                visualize_particle_pos,
                                                                visualize_particle_orientation)
            if object_soup_flag == True:
                visualize_particle_Id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/camsoup_par_with_visual_small_PE_hor.urdf"),
                                                                visualize_particle_pos,
                                                                visualize_particle_orientation)
            self.particle_with_visual_id_collection.append(visualize_particle_Id)
    def display_particle_PM(self):
        for index, particle in enumerate(self.particle_cloud_PM):
            visualize_particle_pos = [particle.x, particle.y, particle.z]
            visualize_particle_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            visualize_particle_orientation = p_visualisation.getQuaternionFromEuler(visualize_particle_angle)
            if object_cracker_flag == True:
                visualize_particle_Id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cube/cheezit_par_with_visual_small_PM_hor.urdf"),
                                                                visualize_particle_pos,
                                                                visualize_particle_orientation)
            if object_soup_flag == True:
                visualize_particle_Id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/camsoup_par_with_visual_small_PM_hor.urdf"),
                                                                visualize_particle_pos,
                                                                visualize_particle_orientation)
            self.particle_with_visual_id_collection_PM.append(visualize_particle_Id)

    def initial_and_set_simulation_env(self,joint_of_robot):
        for index, particle in enumerate(self.particle_cloud):
            pybullet_simulation_env = bc.BulletClient(connection_mode=p.DIRECT)#DIRECT,GUI_SERVER
            self.pybullet_particle_env_collection.append(pybullet_simulation_env)
            if update_style_flag == "time":
                pybullet_simulation_env.setTimeStep(change_sim_time)
            pybullet_simulation_env.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85,
                                                       cameraTargetPosition=[0.5, 0.3, 0.2])
            pybullet_simulation_env.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_simulation_env.setGravity(0,0,-9.81)
            fake_plane_id = pybullet_simulation_env.loadURDF("plane.urdf")
            if task_flag == "2":
                if object_cracker_flag == True:
                    sim_base_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/cube/base_of_cheezit.urdf"),
                                                                pw_T_base_pos,
                                                                pw_T_base_ori,
                                                                useFixedBase=1)
                if object_soup_flag == True:
                    sim_base_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/cube/base_of_cheezit.urdf"),
                                                                pw_T_base_pos,
                                                                pw_T_base_ori,
                                                                useFixedBase=1)
            fake_robot_start_pos = self.real_robot_start_pos
            fake_robot_start_orientation = self.real_robot_start_ori
            fake_robot_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                                             fake_robot_start_pos,
                                                             fake_robot_start_orientation,
                                                             useFixedBase=1)
            self.fake_robot_id_collection.append(fake_robot_id)

            #set joint of fake robot
            self.set_sim_robot_JointPosition(pybullet_simulation_env,fake_robot_id,joint_of_robot)

            particle_no_visual_start_pos = [particle.x, particle.y, particle.z]
            particle_no_visual_start_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            particle_no_visual_start_orientation = pybullet_simulation_env.getQuaternionFromEuler(particle_no_visual_start_angle)
            if object_cracker_flag == True:
                particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/cube/cheezit_par_no_visual_small_hor.urdf"),
                                                                        particle_no_visual_start_pos,
                                                                        particle_no_visual_start_orientation)
            if object_soup_flag == True:
                particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/soup/camsoup_par_no_visual_small_hor.urdf"),
                                                                        particle_no_visual_start_pos,
                                                                        particle_no_visual_start_orientation)
            while True:
                pybullet_simulation_env.stepSimulation()
                flag = 0
                contacts = pybullet_simulation_env.getContactPoints(bodyA=fake_robot_id, bodyB=particle_no_visual_id)
                # pmin,pmax = pybullet_simulation_env.getAABB(particle_no_visual_id)
                # collide_ids = pybullet_simulation_env.getOverlappingObjects(pmin,pmax)
                # length = len(collide_ids)
                for contact in contacts:
                    contact_dis = contact[8]
                    if contact_dis < -0.001:
                        #print("detected contact during initialization. BodyA: %d, BodyB: %d, LinkOfA: %d, LinkOfB: %d", contact[1], contact[2], contact[3], contact[4])
                        Px,Py,Pz,Px_angle,Py_angle,Pz_angle,P_quat = self.generate_random_pose(self.noise_object_pose,self.pw_T_object_ori_dope)
                        pybullet_simulation_env.resetBasePositionAndOrientation(particle_no_visual_id,
                                                                                [Px,Py,Pz],
                                                                                P_quat)
                        flag = 1
                        particle.x = Px
                        particle.y = Py
                        particle.z = Pz
                        particle.x_angle = Px_angle
                        particle.y_angle = Py_angle
                        particle.z_angle = Pz_angle
                        break
                if flag == 0:
                    break
            #pybullet_simulation_env.changeDynamics(particle_no_visual_id,-1,mass=3,lateralFriction = 0.75)
            self.particle_no_visual_id_collection.append(particle_no_visual_id)
        obj_est_set = self.compute_estimate_pos_of_object(self.particle_cloud)
        return obj_est_set[0],obj_est_set[1],obj_est_set[2],obj_est_set[3],obj_est_set[4],obj_est_set[5]
    def initial_and_set_simulation_env_PM(self,joint_of_robot):
        self.particle_cloud_PM = copy.deepcopy(self.particle_cloud)
        for index, particle in enumerate(self.particle_cloud_PM):
            pybullet_simulation_env = bc.BulletClient(connection_mode=p.DIRECT) # GUI_SERVER, DIRECT
            self.pybullet_particle_env_collection_PM.append(pybullet_simulation_env)
            pybullet_simulation_env.setAdditionalSearchPath(pybullet_data.getDataPath())
            pybullet_simulation_env.setGravity(0,0,-9.81)
            fake_plane_id = pybullet_simulation_env.loadURDF("plane.urdf")

            particle_no_visual_start_pos = [particle.x, particle.y, particle.z]
            particle_no_visual_start_angle = [particle.x_angle, particle.y_angle, particle.z_angle]
            particle_no_visual_start_orientation = pybullet_simulation_env.getQuaternionFromEuler(particle_no_visual_start_angle)
            if object_cracker_flag == True:
                particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/cube/cheezit_par_no_visual_small_hor.urdf"),
                                                                        particle_no_visual_start_pos,
                                                                        particle_no_visual_start_orientation)
            if object_soup_flag == True:
                particle_no_visual_id = pybullet_simulation_env.loadURDF(os.path.expanduser("~/project/object/soup/camsoup_par_no_visual_small_hor.urdf"),
                                                                        particle_no_visual_start_pos,
                                                                        particle_no_visual_start_orientation)
            #pybullet_simulation_env.changeDynamics(particle_no_visual_id,-1,mass=3,lateralFriction = 0.7)
            self.particle_no_visual_id_collection_PM.append(particle_no_visual_id)
        obj_est_set_PM = self.compute_estimate_pos_of_object(self.particle_cloud_PM)
        return obj_est_set_PM[0],obj_est_set_PM[1],obj_est_set_PM[2],obj_est_set_PM[3],obj_est_set_PM[4],obj_est_set_PM[5]


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
                
    def add_noise_to_init_par(self,current_pos,sigma_init):
        mean = current_pos
        sigma = sigma_init
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    def take_easy_gaussian_value(self,mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal
