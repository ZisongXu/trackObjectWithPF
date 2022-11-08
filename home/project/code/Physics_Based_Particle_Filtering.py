# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:57:49 2021

@author: 12106
"""
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
#class in other files
from Franka_robot import Franka_robot
from Ros_listener import Ros_listener
from Particle import Particle
from InitialRealworldModel import InitialRealworldModel
from InitialSimulationModel import InitialSimulationModel
from EstimatedObjectPose import EstimatedObjectPose
from ObservationPose import ObservationPose
from OptitrackPose import OptitrackPose

# CVPF Pose list (motion model)
boss_obs_pose_CVPF = []
boss_est_pose_CVPF = []


#Class of Physics-based Particle Filtering
class PFMove():
    def __init__(self, obj_num=0):
        # initialize internal parameters
        self.obj_num = obj_num
        self.particle_cloud = copy.deepcopy(initial_parameter.particle_cloud)
        self.particle_no_visual_id_collection = copy.deepcopy(initial_parameter.particle_no_visual_id_collection)
        self.pybullet_env_id_collection = copy.deepcopy(initial_parameter.pybullet_particle_env_collection)
        self.pybullet_sim_fake_robot_id_collection = copy.deepcopy(initial_parameter.fake_robot_id_collection)
        self.particle_with_visual_id_collection = copy.deepcopy(initial_parameter.particle_with_visual_id_collection)
        self.joint_num = 7
        self.object_estimate_pose_x = []
        self.object_estimate_pose_y = []
        self.object_real_____pose_x = []
        self.object_real_____pose_y = []

    
    # buffer function
    def real_robot_control_PB(self, real_robot_joint_pos, pw_T_obj_obse_objects_pose_list, do_obs_update):
        # begin to run the PBPF algorithm
        self.update_particle_filter_PB(self.pybullet_env_id_collection, # simulation environment per particle
                                       self.pybullet_sim_fake_robot_id_collection, # fake robot id per sim_env
                                       real_robot_joint_pos, # execution actions of the fake robot
                                       pw_T_obj_obse_objects_pose_list, # [object_pose1, object_pose2]
                                       do_obs_update) # flag for judging Obse work
    
    def get_real_robot_joint(self, pybullet_env_id, real_robot_id):
        real_robot_joint_list = []
        for index in range(self.joint_num):
            real_robot_info = pybullet_env_id.getJointState(real_robot_id,index)[0]
            real_robot_joint_list.append(real_robot_info)
        return real_robot_joint_list
        
    def set_real_robot_JointPosition(self,pybullet_env,robot, position):
        num_joints = 9
        for joint_index in range(num_joints):
            if joint_index == 7 or joint_index == 8:
                pybullet_env.setJointMotorControl2(robot,
                                                   joint_index+2,
                                                   pybullet_env.POSITION_CONTROL,
                                                   targetPosition=position[joint_index])
            else:
                pybullet_env.setJointMotorControl2(robot,
                                                   joint_index,
                                                   pybullet_env.POSITION_CONTROL,
                                                   targetPosition=position[joint_index])

    def compute_pos_err_bt_2_points(self,pos1,pos2):
        x_d = pos1[0]-pos2[0]
        y_d = pos1[1]-pos2[1]
        z_d = pos1[2]-pos2[2]
        distance = math.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)
        return distance
    
    # executed_control
    def update_particle_filter_PB(self, pybullet_sim_env, fake_robot_id, real_robot_joint_pos,
                                  pw_T_obj_obse_objects_pose_list, do_obs_update):
        global flag_record_obse
        global flag_record_PBPF
        global flag_record
        self.times = []
        t1 = time.time()
        # motion model
        self.motion_update_PB_parallelised(pybullet_sim_env, fake_robot_id, real_robot_joint_pos)
        t2 = time.time()
        self.times.append(t2-t1)
        # observation model
        if do_obs_update:
            self.observation_update_PB(pw_T_obj_obse_objects_pose_list)
        # Compute mean of particles
        object_estimate_pose = self.compute_estimate_pos_of_object(self.particle_cloud)
        # display estimated object
        if visualisation_flag == True and visualisation_mean == True:
            self.display_estimated_object_in_visual_model(object_estimate_pose)
        # display particles
        if visualisation_particle_flag == True:
            self.display_particle_in_visual_model_PB(self.particle_cloud)
        # self.draw_contrast_figure(estimated_object_pos,observation)
        return
    
    # judge if any particles are contact
    def isAnyParticleInContact(self):
        for index, particle in enumerate(self.particle_cloud):
            for obj_index in range(object_num):
                # get pose from particle
                pw_T_par_sim_pw_env = self.pybullet_env_id_collection[index]
#                pw_T_par_sim_id = self.particle_no_visual_id_collection[index][obj_index]
                pw_T_par_sim_id = particle[obj_index].no_visual_par_id
#                sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(self.pybullet_env_id_collection[index], pw_T_par_sim_id)
                sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(pw_T_par_sim_pw_env, pw_T_par_sim_id)
                # reset pose of object in pybullet vis to the pose
                p_visualisation.resetBasePositionAndOrientation(contact_obj_id_list[obj_index],
                                                                sim_par_cur_pos,
                                                                sim_par_cur_ori)
                # check contact 
                pmin, pmax = p_visualisation.getAABB(contact_obj_id_list[obj_index])
                collide_ids = p_visualisation.getOverlappingObjects(pmin, pmax)
                length = len(collide_ids)
                for t_i in range(length):
                    # print("body id: ",collide_ids[t_i][1])
                    if collide_ids[t_i][1] == 8 or collide_ids[t_i][1] == 9 or collide_ids[t_i][1] == 10 or collide_ids[t_i][1] == 11:
                        return True
                # print("check collision")
                # p_visualisation.stepSimulation()
                # contacts = p_visualisation.getContactPoints(bodyA=real_robot_id, bodyB=contact_obj_id_list[obj_index])
                # for contact in contacts:
                #     contact_dis = contact[8]
                #     if contact_dis < 0.001:
                #         return True
        return False
    
    # update particle cloud particle angle
    def update_partcile_cloud_pose_PB(self, index, obj_index, x, y, z, ori, linearVelocity, angularVelocity):
        self.particle_cloud[index][obj_index].pos = [x, y, z]
        self.particle_cloud[index][obj_index].ori = copy.deepcopy(ori)
        self.particle_cloud[index][obj_index].linearVelocity = linearVelocity
        self.particle_cloud[index][obj_index].angularVelocity = angularVelocity
        
    # motion model
    def motion_update_PB_parallelised(self, pybullet_sim_env, fake_robot_id, real_robot_joint_pos):
        global simRobot_touch_par_flag
        threads = []
        for index, pybullet_env in enumerate(pybullet_sim_env):
            # execute code in parallel
            if simRobot_touch_par_flag == 1:
                thread = threading.Thread(target=self.function_to_parallelise, args=(index, pybullet_env, fake_robot_id, real_robot_joint_pos))
            else:
                thread = threading.Thread(target=self.sim_robot_move_direct, args=(index, pybullet_env, fake_robot_id, real_robot_joint_pos))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        if simRobot_touch_par_flag == 0:
            return
    
    # synchronizing the motion of the robot in the simulation
    def sim_robot_move_direct(self, index, pybullet_env, robot_id, position):
        if len(position) == 3:
            a = 1
        else:
            num_joints = 9
            for joint_index in range(num_joints):
                if joint_index == 7 or joint_index == 8:
                    pybullet_env.resetJointState(robot_id[index],
                                                 joint_index+2,
                                                 targetValue=position[joint_index])
                else:
                    pybullet_env.resetJointState(robot_id[index],
                                                 joint_index,
                                                 targetValue=position[joint_index])
    
    def pose_sim_robot_move(self, index, pybullet_env, fake_robot_id, real_robot_joint_pos):
        if len(real_robot_joint_pos) == 3:
            print(len(real_robot_joint_pos))
            print(real_robot_joint_pos)
            input("stop")
        else:
            flag_set_sim = 1
            # ensure the robot arm in the simulation moves to the final state on each update
            while True:
                if flag_set_sim == 0:
                    break
                self.set_real_robot_JointPosition(pybullet_env, fake_robot_id[index], real_robot_joint_pos)
                pybullet_env.stepSimulation()
                real_rob_joint_list_cur = self.get_real_robot_joint(pybullet_env, fake_robot_id[index])
                flag_set_sim = self.compare_rob_joint(real_rob_joint_list_cur, real_robot_joint_pos)
            
    def function_to_parallelise(self, index, pybullet_env, fake_robot_id, real_robot_joint_pos):
        collision_detection_obj_id = []
        # ensure that each update of particles in the simulation inherits the velocity of the previous update 
        for obj_index in range(self.obj_num):
            pw_T_par_sim_id = self.particle_cloud[index][obj_index].no_visual_par_id
            pybullet_env.resetBaseVelocity(pw_T_par_sim_id,
                                           self.particle_cloud[index][obj_index].linearVelocity,
                                           self.particle_cloud[index][obj_index].angularVelocity)
            # change particle parameters
            self.change_obj_parameters(pybullet_env, pw_T_par_sim_id)
        # execute the control
        if update_style_flag == "pose":
            self.pose_sim_robot_move(index, pybullet_env, fake_robot_id, real_robot_joint_pos)
        elif update_style_flag == "time":
            # change simulation time
            pf_update_interval_in_sim = boss_pf_update_interval_in_real / change_sim_time
            # make sure all particles are updated
            for time_index in range(int(pf_update_interval_in_sim)):
                self.set_real_robot_JointPosition(pybullet_env, fake_robot_id[index], real_robot_joint_pos)
                pybullet_env.stepSimulation()
        ### ori: x,y,z,w
        # get velocity of each particle
        collision_detection_obj_id.append(fake_robot_id[index])
        for obj_index in range(self.obj_num):
            pw_T_par_sim_id = self.particle_cloud[index][obj_index].no_visual_par_id
            linearVelocity, angularVelocity = pybullet_env.getBaseVelocity(pw_T_par_sim_id)
            sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(pybullet_env, pw_T_par_sim_id)
            # add noise on pose of each particle
            normal_x, normal_y, normal_z, P_quat = self.add_noise_pose(sim_par_cur_pos, sim_par_cur_ori)
            pybullet_env.resetBasePositionAndOrientation(pw_T_par_sim_id,
                                                         [normal_x, normal_y, normal_z],
                                                         P_quat)
            collision_detection_obj_id.append(pw_T_par_sim_id)
            # check collision
            while True:
                flag = 0
                for check_num in range(obj_index+1):
                    pybullet_env.stepSimulation()
                    contacts = pybullet_env.getContactPoints(bodyA=collision_detection_obj_id[check_num], 
                                                             bodyB=collision_detection_obj_id[-1])
                    # pmin,pmax = pybullet_simulation_env.getAABB(particle_no_visual_id)
                    # collide_ids = pybullet_simulation_env.getOverlappingObjects(pmin,pmax)
                    # length = len(collide_ids)
                    for contact in contacts:
                        contact_dis = contact[8]
                        if contact_dis < -0.001:
                            #print("detected contact during initialization. BodyA: %d, BodyB: %d, LinkOfA: %d, LinkOfB: %d", contact[1], contact[2], contact[3], contact[4])
                            normal_x, normal_y, normal_z, P_quat = self.add_noise_pose(sim_par_cur_pos, sim_par_cur_ori)
                            pybullet_env.resetBasePositionAndOrientation(pw_T_par_sim_id,
                                                                         [normal_x, normal_y, normal_z],
                                                                         P_quat)
                            flag = 1
                            break
                    if flag == 1:
                        break
                if flag == 0:
                    break
            self.update_partcile_cloud_pose_PB(index, obj_index, normal_x, normal_y, normal_z, P_quat, linearVelocity, angularVelocity)
        # pipe.send()
    
    # add noise
    def add_noise_pose(self, sim_par_cur_pos, sim_par_cur_ori):
        normal_x = self.add_noise_2_par(sim_par_cur_pos[0])
        normal_y = self.add_noise_2_par(sim_par_cur_pos[1])
        normal_z = self.add_noise_2_par(sim_par_cur_pos[2])
        #add noise on ang of each particle
        quat = copy.deepcopy(sim_par_cur_ori)#x,y,z,w
        quat_QuatStyle = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])# w,x,y,z
        random_dir = random.uniform(0, 2*math.pi)
        z_axis = random.uniform(-1,1)
        x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
        y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
        angle_noise = self.add_noise_2_ang(0)
        w_quat = math.cos(angle_noise/2.0)
        x_quat = math.sin(angle_noise/2.0) * x_axis
        y_quat = math.sin(angle_noise/2.0) * y_axis
        z_quat = math.sin(angle_noise/2.0) * z_axis
        ###nois_quat(w,x,y,z); new_quat(w,x,y,z)
        nois_quat = Quaternion(x=x_quat, y=y_quat, z=z_quat, w=w_quat)
        new_quat = nois_quat * quat_QuatStyle
        ###pb_quat(x,y,z,w)
        pb_quat = [new_quat[1],new_quat[2],new_quat[3],new_quat[0]]
        new_angle = p_visualisation.getEulerFromQuaternion(pb_quat)
        P_quat = p_visualisation.getQuaternionFromEuler(new_angle)
        # pipe.send()
        return normal_x, normal_y, normal_z, P_quat
    
    # observation model
    def observation_update_PB(self, pw_T_obj_obse_objects_pose_list):
        for obj_index in range(self.obj_num):
            obse_obj_pos = pw_T_obj_obse_objects_pose_list[obj_index].pos
            obse_obj_ori = pw_T_obj_obse_objects_pose_list[obj_index].ori # pybullet x,y,z,w
            
            # make sure theta between -pi and pi
            obse_obj_ori_corr = quaternion_correction(obse_obj_ori)
#            nois_obj_quat = Quaternion(x=nois_obj_ori[0],y=nois_obj_ori[1],z=nois_obj_ori[2],w=nois_obj_ori[3]) # w,x,y,z
#            cos_theta_over_2 = nois_obj_quat.w
#            sin_theta_over_2 = math.sqrt(nois_obj_quat.x ** 2 + nois_obj_quat.y ** 2 + nois_obj_quat.z ** 2)
#            theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
#            theta = theta_over_2 * 2
#            if theta >= math.pi or theta <= -math.pi:
#                nois_obj_ori = [-nois_obj_x_ori, -nois_obj_y_ori, -nois_obj_z_ori, -nois_obj_w_ori]
                
            for index, particle in enumerate(self.particle_cloud): # particle angle
                particle_x = particle[obj_index].pos[0]
                particle_y = particle[obj_index].pos[1]
                particle_z = particle[obj_index].pos[2]
                mean = 0
                # position weight
                dis_x = abs(particle_x - obse_obj_pos[0])
                dis_y = abs(particle_y - obse_obj_pos[1])
                dis_z = abs(particle_z - obse_obj_pos[2])
                dis_xyz = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
                weight_xyz = self.normal_distribution(dis_xyz, mean, boss_sigma_obs_pos)
                # rotation weight
                par_ori = quaternion_correction(particle[obj_index].ori)
                nois_obj_quat = Quaternion(x=obse_obj_ori_corr[0], 
                                           y=obse_obj_ori_corr[1], 
                                           z=obse_obj_ori_corr[2], 
                                           w=obse_obj_ori_corr[3]) # Quaternion(): w,x,y,z
                par_quat = Quaternion(x=par_ori[0], y=par_ori[1], z=par_ori[2], w=par_ori[3])
                err_bt_par_obse = par_quat * nois_obj_quat.inverse
                cos_theta_over_2 = err_bt_par_obse.w
                sin_theta_over_2 = math.sqrt(err_bt_par_obse.x ** 2 + err_bt_par_obse.y ** 2 + err_bt_par_obse.z ** 2)
                theta_over_2 = math.atan2(sin_theta_over_2, cos_theta_over_2)
                theta = theta_over_2 * 2
                weight_ang = self.normal_distribution(theta, mean, boss_sigma_obs_ang)
                weight = weight_xyz * weight_ang
                particle[obj_index].w = weight
            # old resample function
            # Flag = self.normalize_particles()
            # self.resample_particles()
            # new resample function
        self.resample_particles_update()
        self.set_paticle_in_each_sim_env()
        return

    def compare_rob_joint(self,real_rob_joint_list_cur,real_robot_joint_pos):
        for i in range(self.joint_num):
            diff = 10
            diff = abs(real_rob_joint_list_cur[i] - real_robot_joint_pos[i])
            if diff > 0.005:
                return 1
        return 0
    
    # change particle parameters
    def change_obj_parameters(self, pybullet_env, par_id):
        mass_a = self.take_easy_gaussian_value(mass_mean, mass_sigma)
        if mass_a < 0.001:
            mass_a = 0.05
        lateralFriction = self.take_easy_gaussian_value(friction_mean, friction_sigma)
        spinningFriction = self.take_easy_gaussian_value(friction_mean, friction_sigma)
        rollingFriction = self.take_easy_gaussian_value(friction_mean, friction_sigma)
        if lateralFriction < 0.001:
            lateralFriction = 0.001
        if spinningFriction < 0.001:
            spinningFriction = 0.001
        if rollingFriction < 0.001:
            rollingFriction = 0.001
        restitution = self.take_easy_gaussian_value(restitution_mean, restitution_sigma)
        # if restitution > 1:
        # mass_a = 0.351
        # fricton_b = 0.30
        # mean_damping = 0.4
        # mean_stiffness = 0.9
        # contactStiffness = self.take_easy_gaussian_value(mean_stiffness, 0.3)
        # contactDamping = self.take_easy_gaussian_value(mean_damping, 0.1)
        pybullet_env.changeDynamics(par_id, -1, mass = mass_a, 
                                    lateralFriction = lateralFriction, 
                                    spinningFriction = spinningFriction, 
                                    rollingFriction = rollingFriction, 
                                    restitution = restitution)
                                    #contactStiffness=contactStiffness,
                                    #contactDamping=contactDamping)

    def get_item_pos(self,pybullet_env,item_id):
        item_info = pybullet_env.getBasePositionAndOrientation(item_id)
        return item_info[0],item_info[1]

    def add_noise_2_par(self,current_pos):
        mean = current_pos
        sigma = pos_noise
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def add_noise_2_ang(self,cur_angle):
        mean = cur_angle
        sigma = boss_sigma_obs_ang
        sigma = ang_noise
        new_angle_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_angle_is_added_noise

    def take_easy_gaussian_value(self,mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal

    def normal_distribution(self, x, mean, sigma):
        return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)

    def normalize_particles(self):
        flag_1 = 0
        tot_weight = sum([particle.w for particle in self.particle_cloud])
        if tot_weight == 0:
            # print("Error!,PBPF particles total weight is 0")
            tot_weight = 1
            flag_1 = 1
        for particle in self.particle_cloud:
            if flag_1 == 0:
                particle_w = particle.w/tot_weight
                particle.w = particle_w
            else:
                particle.w = 1/particle_num
                
    # old particle angle
    def resample_particles(self):
        particles_w = []
        newParticles = []
        n_particle = len(self.particle_cloud)
        for particle in self.particle_cloud:
            particles_w.append(particle.w)
        particle_array= np.random.choice(a = n_particle, size = n_particle, replace=True, p= particles_w)
        particle_array_list = list(particle_array)
        for index,i in enumerate(particle_array_list):
            particle = Particle(self.particle_cloud[i].pos,
                                self.particle_cloud[i].ori,
                                1.0/particle_num, index)
            newParticles.append(particle)
        self.particle_cloud = copy.deepcopy(newParticles)
    
    # new
    def resample_particles_update(self):
        n_particle = len(self.particle_cloud)
        newParticles_list = [[]*self.obj_num for _ in range(n_particle)]
        for obj_index in range(self.obj_num):
            particles_w = []
            # newParticles = []
            base_w = 0
            base_w_list = []
            base_w_list.append(base_w)
            particle_array_list = []
            for particle in self.particle_cloud:
                particles_w.append(particle[obj_index].w)
                base_w = base_w + particle[obj_index].w
                base_w_list.append(base_w)
            w_sum = sum(particles_w)
            r = random.uniform(0, w_sum)
            for index in range(n_particle):
                if w_sum > 0.00000001:
                    position = (r + index * w_sum / particle_num) % w_sum
                    position_index = self.compute_position(position, base_w_list)
                    particle_array_list.append(position_index)
                else:
                    particle_array_list.append(index)
            for index,i in enumerate(particle_array_list): # particle angle
                particle = Particle(self.particle_cloud[i][obj_index].par_name,
                                    self.particle_cloud[index][obj_index].visual_par_id,
                                    self.particle_cloud[index][obj_index].no_visual_par_id,
                                    self.particle_cloud[i][obj_index].pos,
                                    self.particle_cloud[i][obj_index].ori,
                                    1.0/particle_num, 
                                    index,
                                    self.particle_cloud[i][obj_index].linearVelocity,
                                    self.particle_cloud[i][obj_index].angularVelocity)
                newParticles_list[index].append(particle)
#                newParticles.append(particle)
        self.particle_cloud = copy.deepcopy(newParticles_list)

    def compute_position(self, position, base_w_list):
        for index in range(1, len(base_w_list)):
            if position <= base_w_list[index] and position > base_w_list[index - 1]:
                return index - 1
            else:
                continue
               
    def set_paticle_in_each_sim_env(self): # particle angle
        for obj_index in range(self.obj_num):
            for index, pybullet_env in enumerate(self.pybullet_env_id_collection):
#                pw_T_par_sim_id = self.particle_no_visual_id_collection[index][obj_index]
                pw_T_par_sim_id = self.particle_cloud[index][obj_index].no_visual_par_id
                par_obj_pos = self.particle_cloud[index][obj_index].pos
                par_obj_ori = self.particle_cloud[index][obj_index].ori
                pybullet_env.resetBasePositionAndOrientation(pw_T_par_sim_id,
                                                             par_obj_pos,
                                                             par_obj_ori)
        return

    def display_particle_in_visual_model_PB(self, particle_cloud): # particle angle
        for obj_index in range(self.obj_num):
            for index, particle in enumerate(particle_cloud):
                w_T_par_sim_id = particle[obj_index].visual_par_id
                par_obj_pos = particle[obj_index].pos
                par_obj_ori = particle[obj_index].ori
                p_visualisation.resetBasePositionAndOrientation(w_T_par_sim_id,
                                                                par_obj_pos,
                                                                par_obj_ori)

    def display_estimated_object_in_visual_model(self, estimated_object_set):
        for obj_index in range(self.obj_num):
            esti_obj_id = estimated_object_set[obj_index].esti_obj_id
            esti_obj_pos = estimated_object_set[obj_index].pos
            esti_obj_ori = estimated_object_set[obj_index].ori
            p_visualisation.resetBasePositionAndOrientation(esti_obj_id,
                                                            esti_obj_pos,
                                                            esti_obj_ori)

    def draw_contrast_figure(self, estimated_object_pos, observation):
        self.object_estimate_pose_x.append(estimated_object_pos[0])
        self.object_estimate_pose_y.append(estimated_object_pos[1])
        self.object_real_____pose_x.append(observation[0])
        self.object_real_____pose_y.append(observation[1])
        plt.plot(self.object_estimate_pose_x,self.object_estimate_pose_y,"x-",label="Estimated Object Pose")
        plt.plot(self.object_real_____pose_x,self.object_real_____pose_y,"*-",label="Real Object Pose")
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.legend()
        plt.savefig('chart.png')
        plt.close()
        return

    def compute_estimate_pos_of_object(self, particle_cloud): # need to change
        esti_objs_cloud = []
        for obj_index in range(self.obj_num):
            x_set = 0
            y_set = 0
            z_set = 0
            w_set = 0
            quaternions = []
            qws = []
            for index, particle in enumerate(particle_cloud):
                x_set = x_set + particle[obj_index].pos[0] * particle[obj_index].w
                y_set = y_set + particle[obj_index].pos[1] * particle[obj_index].w
                z_set = z_set + particle[obj_index].pos[2] * particle[obj_index].w
                q = quaternion_correction(particle[obj_index].ori)
                qws.append(particle[obj_index].w)
                quaternions.append([q[0], q[1], q[2], q[3]])
                w_set = w_set + particle[obj_index].w
            q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
            est_obj_pose = EstimatedObjectPose(particle[i].par_name, 
                                               estimated_object_set[obj_index].esti_obj_id, 
                                               [x_set/w_set, y_set/w_set, z_set/w_set], 
                                               [q[0], q[1], q[2], q[3]], obj_index)
            esti_objs_cloud.append(est_obj_pose)
        return esti_objs_cloud

    def rotation_4_4_to_transformation_4_4(self, rotation_4_4,pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4


#Class of Constant-velocity Particle Filtering
class PFMoveCV():
    def __init__(self, obj_num=0):
        # init internals   
        self.obj_num = obj_num
        self.particle_cloud_CV = copy.deepcopy(initial_parameter.particle_cloud_CV)
        self.particle_no_visual_id_collection_CV = copy.deepcopy(initial_parameter.particle_no_visual_id_collection_CV)
        self.pybullet_env_id_collection_CV = copy.deepcopy(initial_parameter.pybullet_particle_env_collection_CV)
        self.particle_with_visual_id_collection_CV = copy.deepcopy(initial_parameter.particle_with_visual_id_collection_CV)
        self.object_estimate_pose_x = []
        self.object_estimate_pose_y = []
        self.object_real_____pose_x = []
        self.object_real_____pose_y = []

    def compute_pos_err_bt_2_points(self,pos1,pos2):
        x_d = pos1[0]-pos2[0]
        y_d = pos1[1]-pos2[1]
        z_d = pos1[2]-pos2[2]
        distance = math.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)
        return distance

    # executed_control
    def update_particle_filter_CV(self, pw_T_obj_obse_objects_pose_list, do_obs_update):
        global flag_record_CVPF
        global flag_record
        # motion model
        self.motion_update_CV(pw_T_obj_obse_objects_pose_list)
        # observation model
        if do_obs_update:
            self.observation_update_CV(pw_T_obj_obse_objects_pose_list)
        # Compute mean of particles
        object_estimate_pose_CV = self.compute_estimate_pos_of_object(self.particle_cloud_CV)
        
        boss_est_pose_CVPF.append(object_estimate_pose_CV)
        # display estimated object
        if visualisation_flag == True and visualisation_mean == True:
            self.display_estimated_object_in_visual_model(object_estimate_pose_CV)
        # display particles
        if visualisation_particle_flag == True:
            self.display_particle_in_visual_model_CV(self.particle_cloud_CV)
        return

    def isAnyParticleInContact(self):
        for index, particle in enumerate(self.particle_cloud_CV):
            for obj_index in range(object_num):
                
                # get pose from particle
                pw_T_par_sim_pw_env = self.pybullet_env_id_collection_CV[index]
#                pw_T_par_sim_id = self.particle_no_visual_id_collection[index][obj_index]
                pw_T_par_sim_id = particle[obj_index].no_visual_par_id
#                sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(self.pybullet_env_id_collection[index], pw_T_par_sim_id)
                sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(pw_T_par_sim_pw_env, pw_T_par_sim_id)
                # reset pose of object in pybullet vis to the pose
                p_visualisation.resetBasePositionAndOrientation(contact_obj_id_list[obj_index],
                                                                sim_par_cur_pos,
                                                                sim_par_cur_ori)
                # check contact 
                pmin, pmax = p_visualisation.getAABB(contact_obj_id_list[obj_index])
                collide_ids = p_visualisation.getOverlappingObjects(pmin, pmax)
                length = len(collide_ids)
                for t_i in range(length):
                    # print("body id: ",collide_ids[t_i][1])
                    if collide_ids[t_i][1] == 8 or collide_ids[t_i][1] == 9 or collide_ids[t_i][1] == 10 or collide_ids[t_i][1] == 11:
                        return True
                # p_visualisation.stepSimulation()
                # contacts = p_visualisation.getContactPoints(bodyA=real_robot_id, bodyB=contact_obj_id_list[obj_index])
                # for contact in contacts:
                #     contact_dis = contact[8]
                #     if contact_dis > 0.001:
                #         return True
                
        return False

    def motion_update_CV(self, pw_T_obj_obse_objects_pose_list):
        # t0, t1: use observation data (obs0, obs1) to update motion
        if flag_update_num_CV < 2:
            length = len(boss_obs_pose_CVPF)
            obs_curr_pose_list = copy.deepcopy(boss_obs_pose_CVPF[length-1]) # [obse_obj1_n,   obse_obj2_n]
            obs_last_pose_list = copy.deepcopy(boss_obs_pose_CVPF[length-2]) # [obse_obj1_n-1, obse_obj2_n-1]
            for obj_index in range (self.obj_num):
                obs_curr_pose = obs_curr_pose_list[obj_index] # class objext
                obs_last_pose = obs_last_pose_list[obj_index] # class objext
                obs_last_pos = obs_last_pose.pos
                obs_last_ori = obs_last_pose.ori
                obs_curr_pos = obs_curr_pose.pos
                obs_curr_ori = obs_curr_pose.ori
                obsO_T_obsN = self.compute_transformation_matrix(obs_last_pos, obs_last_ori, obs_curr_pos, obs_curr_ori)
                parO_T_parN = copy.deepcopy(obsO_T_obsN)
                self.update_particle_in_motion_model_CV(obj_index, parO_T_parN, pw_T_obj_obse_objects_pose_list)
        # after t1: use (est0, est1) to update motion
        else:
            length = len(boss_est_pose_CVPF)
            est_curr_pose_list = copy.deepcopy(boss_est_pose_CVPF[length-1]) # [esti_obj1_n,   esti_obj2_n]
            est_last_pose_list = copy.deepcopy(boss_est_pose_CVPF[length-2]) # [esti_obj1_n,   esti_obj2_n]
            for obj_index in range (self.obj_num):
                est_curr_pose = est_curr_pose_list[obj_index]
                est_last_pose = est_last_pose_list[obj_index]
                est_curr_pos = est_curr_pose.pos
                est_curr_ori = est_curr_pose.ori
                est_last_pos = est_last_pose.pos
                est_last_ori = est_last_pose.ori
                estO_T_estN = self.compute_transformation_matrix(est_last_pos, est_last_ori, est_curr_pos, est_curr_ori)
                parO_T_parN = copy.deepcopy(estO_T_estN)
                self.update_particle_in_motion_model_CV(obj_index, parO_T_parN, pw_T_obj_obse_objects_pose_list)
        return

    def observation_update_CV(self, pw_T_obj_obse_objects_pose_list):
        for obj_index in range(self.obj_num):
            obse_obj_pos = pw_T_obj_obse_objects_pose_list[obj_index].pos
            obse_obj_ori = pw_T_obj_obse_objects_pose_list[obj_index].ori # pybullet x,y,z,w
                
            # make sure theta between -pi and pi
            obse_obj_ori_corr = quaternion_correction(obse_obj_ori)
#            nois_obj_quat = Quaternion(x=nois_obj_ori[0],y=nois_obj_ori[1],z=nois_obj_ori[2],w=nois_obj_ori[3]) # w,x,y,z
#            cos_theta_over_2 = nois_obj_quat.w
#            sin_theta_over_2 = math.sqrt(nois_obj_quat.x ** 2 + nois_obj_quat.y ** 2 + nois_obj_quat.z ** 2)
#            theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
#            theta = theta_over_2 * 2
#            if theta >= math.pi or theta <= -math.pi:
#                nois_obj_ori = [-nois_obj_x_ori, -nois_obj_y_ori, -nois_obj_z_ori, -nois_obj_w_ori]
           
            for index,particle in enumerate(self.particle_cloud_CV):
                particle_x = particle[obj_index].pos[0]
                particle_y = particle[obj_index].pos[1]
                particle_z = particle[obj_index].pos[2]
                mean = 0
                # position weight
                dis_x = abs(particle_x - obse_obj_pos[0])
                dis_y = abs(particle_y - obse_obj_pos[1])
                dis_z = abs(particle_z - obse_obj_pos[2])
                dis_xyz = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
                weight_xyz = self.normal_distribution(dis_xyz, mean, boss_sigma_obs_pos)
                # rotation weight    
                par_ori = quaternion_correction(particle[obj_index].ori)
                nois_obj_quat = Quaternion(x=obse_obj_ori_corr[0], 
                                           y=obse_obj_ori_corr[1], 
                                           z=obse_obj_ori_corr[2], 
                                           w=obse_obj_ori_corr[3]) # Quaternion(): w,x,y,z
                par_quat = Quaternion(x=par_ori[0], y=par_ori[1], z=par_ori[2], w=par_ori[3])
                err_bt_par_obse = par_quat * nois_obj_quat.inverse
                cos_theta_over_2 = err_bt_par_obse.w
                sin_theta_over_2 = math.sqrt(err_bt_par_obse.x ** 2 + err_bt_par_obse.y ** 2 + err_bt_par_obse.z ** 2)
                theta_over_2 = math.atan2(sin_theta_over_2, cos_theta_over_2)
                theta = theta_over_2 * 2
                weight_ang = self.normal_distribution(theta, mean, boss_sigma_obs_ang)
                weight = weight_xyz * weight_ang
                particle[obj_index].w = weight
        # old resample function
        # Flag = self.normalize_particles_CV()
        # self.resample_particles_CV()
        # new resample function
        self.resample_particles_CV_update()
        self.set_paticle_in_each_sim_env_CV()
        return

    def update_particle_in_motion_model_CV(self, obj_index, parO_T_parN, pw_T_obj_obse_objects_list):
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection_CV):
            pw_T_parO_pos = copy.deepcopy(self.particle_cloud_CV[index][obj_index].pos)
            pw_T_parO_ori = copy.deepcopy(self.particle_cloud_CV[index][obj_index].ori)
            pw_T_parO_3_3 = transformations.quaternion_matrix(pw_T_parO_ori)
            pw_T_parO_4_4 = self.rotation_4_4_to_transformation_4_4(pw_T_parO_3_3,pw_T_parO_pos)
            pw_T_parN = np.dot(pw_T_parO_4_4, parO_T_parN)
            pw_T_parN_pos = [pw_T_parN[0][3], pw_T_parN[1][3], pw_T_parN[2][3]]
            # pw_T_parN_ori = transformations.quaternion_from_matrix(pw_T_parN)
            # pw_T_parN_ang = pybullet_env.getEulerFromQuaternion(pw_T_parN_ori)
            
            # add noise on particle filter
            normal_x = self.add_noise_2_par(pw_T_parN_pos[0])
            normal_y = self.add_noise_2_par(pw_T_parN_pos[1])
            normal_z = self.add_noise_2_par(pw_T_parN_pos[2])
            
            quat = copy.deepcopy(pw_T_obj_obse_objects_list[obj_index].ori)
#            quat = copy.deepcopy(nois_obj_ori_cur) # x,y,z,w
            quat_QuatStyle = Quaternion(x=quat[0],y=quat[1],z=quat[2],w=quat[3]) # w,x,y,z
            random_dir = random.uniform(0, 2*math.pi)
            z_axis = random.uniform(-1,1)
            x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
            y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
            angle_noise = self.add_noise_2_ang(0)
            w_quat = math.cos(angle_noise/2.0)
            x_quat = math.sin(angle_noise/2.0) * x_axis
            y_quat = math.sin(angle_noise/2.0) * y_axis
            z_quat = math.sin(angle_noise/2.0) * z_axis
            ### nois_quat(w,x,y,z); new_quat(w,x,y,z)
            nois_quat = Quaternion(x=x_quat,y=y_quat,z=z_quat,w=w_quat)
            new_quat = nois_quat * quat_QuatStyle
            ### pb_quat(x,y,z,w)
            pb_quat = [new_quat[1], new_quat[2], new_quat[3], new_quat[0]]

            self.particle_cloud_CV[index][obj_index].pos = [normal_x, normal_y, normal_z]
            self.particle_cloud_CV[index][obj_index].ori = copy.deepcopy(pb_quat)


    def get_item_pos(self,pybullet_env, item_id):
        item_info = pybullet_env.getBasePositionAndOrientation(item_id)
        return item_info[0], item_info[1]

    def add_noise_2_par(self,current_pos):
        mean = current_pos
        sigma = pos_noise
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def add_noise_2_ang(self,cur_angle):
        mean = cur_angle
        sigma = boss_sigma_obs_ang
        sigma = ang_noise
        new_angle_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_angle_is_added_noise

    def take_easy_gaussian_value(self,mean,sigma):
        normal = random.normalvariate(mean, sigma)
        return normal

    def normal_distribution(self, x, mean, sigma):
        return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)* sigma)

    def normalize_particles_CV(self):
        flag_1 = 0
        tot_weight = sum([particle.w for particle in self.particle_cloud_CV])
        if tot_weight == 0:
            # print("Error!,CVPF particles total weight is 0")
            tot_weight = 1
            flag_1 = 1
        for particle in self.particle_cloud_CV:
            if flag_1 == 0:
                particle_w = particle.w/tot_weight
                particle.w = particle_w
            else:
                particle.w = 1.0/particle_num
    
    # old particle angle
    def resample_particles_CV(self):
        particles_w = []
        newParticles = []
        n_particle = len(self.particle_cloud_CV)
        for particle in self.particle_cloud_CV:
            particles_w.append(particle.w)
        particle_array= np.random.choice(a = n_particle, size = n_particle, replace=True, p= particles_w)
        particle_array_list = list(particle_array)
        for index,i in enumerate(particle_array_list):
            particle = Particle(self.particle_cloud_CV[i].pos,
                                self.particle_cloud_CV[i].ori,
                                1.0/particle_num,index)
            newParticles.append(particle)
        self.particle_cloud_CV = copy.deepcopy(newParticles)
        
    def resample_particles_CV_update(self):
        n_particle = len(self.particle_cloud_CV)
        newParticles_list = [[]*self.obj_num for _ in range(n_particle)]
        for obj_index in range(self.obj_num):
            particles_w = []
#            newParticles = []
            base_w = 0
            base_w_list = []
            base_w_list.append(base_w)
            particle_array_list = []
            for particle in self.particle_cloud_CV:
                particles_w.append(particle[obj_index].w)
                base_w = base_w + particle[obj_index].w
                base_w_list.append(base_w)
            w_sum = sum(particles_w)
            r = random.uniform(0, w_sum)
            for index in range(n_particle):
                if w_sum != 0:
                    position = (r + index * w_sum / particle_num) % w_sum
                    position_index = self.compute_position_CV(position, base_w_list)
                    particle_array_list.append(position_index)
                elif w_sum == 0:
                    particle_array_list.append(index)
            for index,i in enumerate(particle_array_list):
                particle = Particle(self.particle_cloud_CV[i][obj_index].par_name,
                                    self.particle_cloud_CV[index][obj_index].visual_par_id,
                                    self.particle_cloud_CV[index][obj_index].no_visual_par_id,
                                    self.particle_cloud_CV[i][obj_index].pos,
                                    self.particle_cloud_CV[i][obj_index].ori,
                                    1.0/particle_num, 
                                    index,
                                    self.particle_cloud_CV[i][obj_index].linearVelocity,
                                    self.particle_cloud_CV[i][obj_index].angularVelocity)
                newParticles_list[index].append(particle)
        self.particle_cloud_CV = copy.deepcopy(newParticles_list)
        
    def compute_position_CV(self, position, base_w_list):
        for index in range(1, len(base_w_list)):
            if position <= base_w_list[index] and position > base_w_list[index - 1]:
                return index - 1
            else:
                continue
            
    def set_paticle_in_each_sim_env_CV(self):
        for obj_index in range(self.obj_num):
            for index, pybullet_env in enumerate(self.pybullet_env_id_collection_CV):
                pw_T_par_sim_id = self.particle_cloud_CV[index][obj_index].no_visual_par_id
                par_obj_pos = self.particle_cloud_CV[index][obj_index].pos
                par_obj_ori = self.particle_cloud_CV[index][obj_index].ori
                pybullet_env.resetBasePositionAndOrientation(pw_T_par_sim_id,
                                                             par_obj_pos,
                                                             par_obj_ori)
        return

    def display_particle_in_visual_model_CV(self, particle_cloud): # particle angle
        for obj_index in range(self.obj_num):
            for index, particle in enumerate(particle_cloud):
                w_T_par_sim_id = particle[obj_index].visual_par_id
                par_obj_pos = particle[obj_index].pos
                par_obj_ori = particle[obj_index].ori
                p_visualisation.resetBasePositionAndOrientation(w_T_par_sim_id,
                                                                par_obj_pos,
                                                                par_obj_ori)

    def display_estimated_object_in_visual_model(self, estimated_object_set):
        for obj_index in range(self.obj_num):
            esti_obj_id = estimated_object_set[obj_index].esti_obj_id
            esti_obj_pos = estimated_object_set[obj_index].pos
            esti_obj_ori = estimated_object_set[obj_index].ori
            p_visualisation.resetBasePositionAndOrientation(esti_obj_id,
                                                            esti_obj_pos,
                                                            esti_obj_ori)

    def draw_contrast_figure(self,estimated_object_pos,observation):
        self.object_estimate_pose_x.append(estimated_object_pos[0])
        self.object_estimate_pose_y.append(estimated_object_pos[1])
        self.object_real_____pose_x.append(observation[0])
        self.object_real_____pose_y.append(observation[1])
        plt.plot(self.object_estimate_pose_x,self.object_estimate_pose_y,"x-",label="Estimated Object Pose")
        plt.plot(self.object_real_____pose_x,self.object_real_____pose_y,"*-",label="Real Object Pose")
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.legend()
        plt.savefig('chart.png')
        plt.close()
        return

    def compute_estimate_pos_of_object(self, particle_cloud):
        esti_objs_cloud = []
        for obj_index in range(self.obj_num):
            x_set = 0
            y_set = 0
            z_set = 0
            w_set = 0
            quaternions = []
            qws = []
            for index, particle in enumerate(particle_cloud):
                x_set = x_set + particle[obj_index].pos[0] * particle[obj_index].w
                y_set = y_set + particle[obj_index].pos[1] * particle[obj_index].w
                z_set = z_set + particle[obj_index].pos[2] * particle[obj_index].w
                q = quaternion_correction(particle[obj_index].ori)
                qws.append(particle[obj_index].w)
                quaternions.append([q[0], q[1], q[2], q[3]])
                w_set = w_set + particle[obj_index].w
            q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
            est_obj_pose = EstimatedObjectPose(particle[i].par_name, 
                                               estimated_object_set[obj_index].esti_obj_id, 
                                               [x_set/w_set, y_set/w_set, z_set/w_set], 
                                               [q[0], q[1], q[2], q[3]], obj_index)
            esti_objs_cloud.append(est_obj_pose)
        return esti_objs_cloud

    def compute_transformation_matrix(self, a_pos, a_ori, b_pos, b_ori):
        ow_T_a_3_3 = transformations.quaternion_matrix(a_ori)
        ow_T_a_4_4 = self.rotation_4_4_to_transformation_4_4(ow_T_a_3_3,a_pos)
        ow_T_b_3_3 = transformations.quaternion_matrix(b_ori)
        ow_T_b_4_4 = self.rotation_4_4_to_transformation_4_4(ow_T_b_3_3,b_pos)
        a_T_ow_4_4 = np.linalg.inv(ow_T_a_4_4)
        a_T_b_4_4 = np.dot(a_T_ow_4_4,ow_T_b_4_4)
        return a_T_b_4_4

    def rotation_4_4_to_transformation_4_4(self, rotation_4_4,pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4

# function independent of Class
# add position into transformation matrix
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
# compute the transformation matrix represent that the pose of object in the robot world
def compute_transformation_matrix(a_pos, a_ori, b_pos, b_ori):
    ow_T_a_3_3 = transformations.quaternion_matrix(a_ori)
    ow_T_a_4_4 = rotation_4_4_to_transformation_4_4(ow_T_a_3_3,a_pos)
    ow_T_b_3_3 = transformations.quaternion_matrix(b_ori)
    ow_T_b_4_4 = rotation_4_4_to_transformation_4_4(ow_T_b_3_3,b_pos)
    a_T_ow_4_4 = np.linalg.inv(ow_T_a_4_4)
    a_T_b_4_4 = np.dot(a_T_ow_4_4,ow_T_b_4_4)
    return a_T_b_4_4

# get pose of item
def get_item_pos(pybullet_env, item_id):
    item_info = pybullet_env.getBasePositionAndOrientation(item_id)
    return item_info[0],item_info[1]
# add noise
def add_noise_pose(sim_par_cur_pos, sim_par_cur_ori):
    normal_x = add_noise_2_par(sim_par_cur_pos[0])
    normal_y = add_noise_2_par(sim_par_cur_pos[1])
    normal_z = add_noise_2_par(sim_par_cur_pos[2])
    pos_added_noise = [normal_x, normal_y, normal_z]
    # add noise on ang of each particle
    quat = copy.deepcopy(sim_par_cur_ori)# x,y,z,w
    quat_QuatStyle = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])# w,x,y,z
    random_dir = random.uniform(0, 2*math.pi)
    z_axis = random.uniform(-1,1)
    x_axis = math.cos(random_dir) * math.sqrt(1 - z_axis ** 2)
    y_axis = math.sin(random_dir) * math.sqrt(1 - z_axis ** 2)
    angle_noise = add_noise_2_ang(0)
    w_quat = math.cos(angle_noise/2.0)
    x_quat = math.sin(angle_noise/2.0) * x_axis
    y_quat = math.sin(angle_noise/2.0) * y_axis
    z_quat = math.sin(angle_noise/2.0) * z_axis
    ###nois_quat(w,x,y,z); new_quat(w,x,y,z)
    nois_quat = Quaternion(x=x_quat, y=y_quat, z=z_quat, w=w_quat)
    new_quat = nois_quat * quat_QuatStyle
    ###pb_quat(x,y,z,w)
    ori_added_noise = [new_quat[1],new_quat[2],new_quat[3],new_quat[0]]
    # ori_added_noise = quat
    # new_angle = p_visualisation.getEulerFromQuaternion(pb_quat)
    # x_angle = new_angle[0]
    # y_angle = new_angle[1]
    # z_angle = new_angle[2]
    # x_angle = sim_par_cur_ang[0]
    # y_angle = sim_par_cur_ang[1]
    # z_angle = sim_par_cur_ang[2]
    # P_quat = p_visualisation.getQuaternionFromEuler([x_angle, y_angle, z_angle])
    # pipe.send()
    return pos_added_noise, ori_added_noise
def add_noise_2_par(current_pos):
    mean = current_pos
    pos_noise_sigma = 0.01
    sigma = pos_noise_sigma
    new_pos_is_added_noise = take_easy_gaussian_value(mean, sigma)
    return new_pos_is_added_noise
def add_noise_2_ang(cur_angle):
    mean = cur_angle
    sigma = boss_sigma_obs_ang
    ang_noise_sigma = 0.1
    sigma = ang_noise_sigma
    new_angle_is_added_noise = take_easy_gaussian_value(mean, sigma)
    return new_angle_is_added_noise
# random values generated from a Gaussian distribution
def take_easy_gaussian_value(mean, sigma):
    normal = random.normalvariate(mean, sigma)
    return normal
# display objects or particles in the visualization window
def display_real_object_in_visual_model(ID, opti_obj_pos, opti_obj_ori):
    p_visualisation.resetBasePositionAndOrientation(ID,
                                                    opti_obj_pos,
                                                    opti_obj_ori)
# make sure all angles all between -pi and +pi
def angle_correction(angle):
    if math.pi <= angle <= (3.0 * math.pi):
        angle = angle - 2 * math.pi
    elif -(3.0 * math.pi) <= angle <= -math.pi:
        angle = angle + 2 * math.pi
    angle = abs(angle)
    return angle
# make sure all quaternions all between -pi and +pi
def quaternion_correction(quaternion): # x,y,z,w
    new_quat = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]) # w,x,y,z
    cos_theta_over_2 = new_quat.w
    sin_theta_over_2 = math.sqrt(new_quat.x ** 2 + new_quat.y ** 2 + new_quat.z ** 2)
    theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
    theta = theta_over_2 * 2
    if theta >= math.pi or theta <= -math.pi:
        new_quaternion = [-quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
        return new_quaternion
    return quaternion
# ctrl-c write down the error file
def signal_handler(sig, frame):
    sys.exit()

def cheat_robot_move(real_robot_id, x=0.0, y=0.0, z=0.0):
    p_visualisation.resetBaseVelocity(real_robot_id, [x, y, z])
    p_visualisation.stepSimulation()

if __name__ == '__main__':
    rospy.init_node('PF_for_obse') # ros node
    signal.signal(signal.SIGINT, signal_handler) # interrupt judgment
    # the flag of visualisation
    visualisation_all = True
    visualisation_flag = True # obse and OptiTrack
    visualisation_mean = False
    visualisation_particle_flag = True
    # the flag of object judgment
    observation_cheating_flag = False
    object_flag = "cracker" # cracker/soup
    # OptiTrack works fine flag
    optitrack_working_flag = True
    # number of times to run the algorithm
    file_time = 1
    # which algorithm to run
    run_PBPF_flag = True
    run_CVPF_flag = False
    # scene
    task_flag = "1"
    # update mode (pose/time)
    update_style_flag = "time"
    # the flag is used to determine whether the robot touches the particle in the simulation
    simRobot_touch_par_flag = 0
    object_num = 1
    if update_style_flag == "pose":
        particle_num = 50
    elif update_style_flag == "time":
        if run_PBPF_flag == True:
            particle_num = 50
        elif run_CVPF_flag == True:
            particle_num = 50
    print("This is "+update_style_flag+" update in scene"+task_flag)    
    # some parameters
    d_thresh = 0.005
    a_thresh = 0.01
    d_thresh_CV = 0.0002
    a_thresh_CV = 0.0010
    flag_record = 0
    flag_record_obse = 0
    flag_record_PBPF = 0
    flag_record_CVPF = 0
    flag_update_num_CV = 0
    flag_update_num_PB = 0
    change_sim_time = 1.0/90
    if run_PBPF_flag == True:
        boss_pf_update_interval_in_real = 0.16
    elif run_CVPF_flag == True:
        boss_pf_update_interval_in_real = 0.02
    pf_update_rate = rospy.Rate(1.0/boss_pf_update_interval_in_real)
    # error in xyz axis obse before recalibrating
    boss_sigma_obs_x = 0.03973017808163751 / 2.0
    boss_sigma_obs_y = 0.01167211468503462 / 2.0
    boss_sigma_obs_z = 0.02820930183351492 / 2.0
    # new obse error
    boss_sigma_obs_x = 0.032860982 * 2.0
    boss_sigma_obs_y = 0.012899399 * 1.5
    boss_sigma_obs_z = 0.01
    boss_sigma_obs_ang_init = 0.0216773873 * 2.0
    # Motion model Noise
    pos_noise = 0.001 * 5.0
    ang_noise = 0.05 * 1.0
    # standard deviation of computing the weight
    boss_sigma_obs_ang = 0.216773873
    boss_sigma_obs_ang = 0.0216773873
    boss_sigma_obs_ang = 0.0216773873 * 4
    boss_sigma_obs_pos = 0.038226405
    boss_sigma_obs_pos = 0.004
    boss_sigma_obs_pos = 0.005 * 4
    mass_mean = 0.380
    mass_sigma = 0.5
    friction_mean = 0.1
    friction_sigma = 0.3
    restitution_mean = 0.9
    restitution_sigma = 0.2
    
    PBPF_time_cosuming_list = []
    
    # multi-objects/robot list
    pw_T_obj_obse_objects_list = []
    pw_T_obj_opti_objects_list = []
    # visualisation_model
    if visualisation_all == True:
        p_visualisation = bc.BulletClient(connection_mode=p.GUI_SERVER)#DIRECT,GUI_SERVER
    elif visualisation_all == False:
        p_visualisation = bc.BulletClient(connection_mode=p.DIRECT)#DIRECT,GUI_SERVER
    p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
    p_visualisation.setGravity(0, 0, -9.81)
    if task_flag == "4":
        p_visualisation.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=90, cameraPitch=-10, cameraTargetPosition=[0.5,0.1,0.2])
    else:
        p_visualisation.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85, cameraTargetPosition=[0.3,0.1,0.2])
    
    plane_id = p_visualisation.loadURDF("plane.urdf")   
    # build an object of class "Ros_listener"
    ros_listener = Ros_listener(optitrack_working_flag, object_flag)
    # get object pose in robot world info from obse
    listener = tf.TransformListener()
    # robot pose in sim world (pybullet)
    pw_T_rob_sim_pos = [0.0, 0.0, 0.026]
    pw_T_rob_sim_ori = [0, 0, 0, 1]
    pw_T_rob_sim_3_3 = transformations.quaternion_matrix(pw_T_rob_sim_ori)
    pw_T_rob_sim_4_4 = rotation_4_4_to_transformation_4_4(pw_T_rob_sim_3_3, pw_T_rob_sim_pos)

    
    if observation_cheating_flag == False:
        print("before while loop")
        # observation multi-objects
        objects_name_list = ["cracker", "soup"]
        for i in range(object_num):
            while True:
                try:
                    if object_flag == "cracker":
                        (trans,rot) = listener.lookupTransform('/panda_link0', '/cracker', rospy.Time(0))
                    if object_flag == "soup":
                        (trans,rot) = listener.lookupTransform('/panda_link0', '/soup', rospy.Time(0))
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            # trans_list.append(trans)
            # rot_list.append(rot)
            rob_T_obj_obse_pos = list(trans)
            rob_T_obj_obse_ori = list(rot)
            rob_T_obj_obse_3_3 = transformations.quaternion_matrix(rob_T_obj_obse_ori)
            rob_T_obj_obse_4_4 = rotation_4_4_to_transformation_4_4(rob_T_obj_obse_3_3, rob_T_obj_obse_pos)
            # compute pose of obse object in sim world (pybullet)
            pw_T_obj_obse = np.dot(pw_T_rob_sim_4_4, rob_T_obj_obse_4_4)
            pw_T_obj_obse_pos = [pw_T_obj_obse[0][3], pw_T_obj_obse[1][3], pw_T_obj_obse[2][3]]
            pw_T_obj_obse_ori = transformations.quaternion_from_matrix(pw_T_obj_obse)
            
            # load the obse object
            if visualisation_flag == True and object_flag == "cracker":
                obse_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_obse_obj_with_visual_hor.urdf"),
                                                          pw_T_obj_obse_pos,
                                                          pw_T_obj_obse_ori)
            if visualisation_flag == True and object_flag == "soup":
                obse_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/soup_obse_obj_with_visual_hor.urdf"),
                                                          pw_T_obj_obse_pos,
                                                          pw_T_obj_obse_ori)
            # represent observation multi-objects as a class
            obse_object = ObservationPose(objects_name_list[i], obse_object_id, pw_T_obj_obse_pos, pw_T_obj_obse_ori, index=i)
            pw_T_obj_obse_objects_list.append(obse_object)
        print("after while loop")
        # give some time to update the data
        time.sleep(0.5)
        if optitrack_working_flag == True:
            opti_T_rob_opti_pos = ros_listener.robot_pos
            opti_T_rob_opti_ori = ros_listener.robot_ori
            # compute and load the pose of optitrack_base (only task 4)
            pw_T_base_pos = [0,0,0]
            pw_T_base_ori = [0,0,0,1]
            if task_flag == "4":
                base_of_cheezit_pos = ros_listener.base_pos
                base_of_cheezit_ori = ros_listener.base_ori
                robot_T_base = compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, base_of_cheezit_pos, base_of_cheezit_ori)
                # pw_T_rob_sim_3_3 = transformations.quaternion_matrix(pw_T_rob_sim_ori)
                # pw_T_rob_sim_4_4 = rotation_4_4_to_transformation_4_4(pw_T_rob_sim_3_3, pw_T_rob_sim_pos)
                pw_T_base = np.dot(pw_T_rob_sim_4_4, robot_T_base)
                pw_T_base_pos = [pw_T_base[0][3], pw_T_base[1][3], pw_T_base[2][3]]
                pw_T_base_ori = transformations.quaternion_from_matrix(pw_T_base)
                pw_T_base_ang = p_visualisation.getEulerFromQuaternion(pw_T_base_ori)
                if visualisation_flag == True and object_flag == "cracker":
                    optitrack_base_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/base_of_cracker.urdf"),
                                                                pw_T_base_pos,
                                                                pw_T_base_ori)
                if visualisation_flag == True and object_flag == "soup":
                    optitrack_base_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/base_of_cracker.urdf"),
                                                                pw_T_base_pos,
                                                                pw_T_base_ori)
                
            # optitrack multi-objects
            for i in range(object_num):
                opti_T_obj_opti_pos = ros_listener.object_pos
                opti_T_obj_opti_ori = ros_listener.object_ori
                # compute transformation matrix (OptiTrack)
                rob_T_obj_opti_4_4 = compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, opti_T_obj_opti_pos, opti_T_obj_opti_ori)
                pw_T_obj_opti_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_obj_opti_4_4)
                pw_T_obj_opti_pos = [pw_T_obj_opti_4_4[0][3], pw_T_obj_opti_4_4[1][3], pw_T_obj_opti_4_4[2][3]]
                pw_T_obj_opti_ori = transformations.quaternion_from_matrix(pw_T_obj_opti_4_4)
                # load the groud truth object
                if visualisation_flag == True and object_flag == "cracker":
                    optitrack_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_real_obj_with_visual_hor.urdf"),
                                                                   pw_T_obj_opti_pos,
                                                                   pw_T_obj_opti_ori)
                if visualisation_flag == True and object_flag == "soup":
                    optitrack_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/soup_real_obj_with_visual_hor.urdf"),
                                                                   pw_T_obj_opti_pos,
                                                                   pw_T_obj_opti_ori)
                # represent optitrack multi-objects as a class
                opti_object = OptitrackPose(objects_name_list[i], optitrack_object_id, pw_T_obj_opti_pos, pw_T_obj_opti_ori, index=i)
                pw_T_obj_opti_objects_list.append(opti_object)
            
    
        
    elif observation_cheating_flag == True:
        # load the groud truth object
        pw_T_objs_opti_pose_list = []
        pw_T_obj0_opti_pose = []
        pw_T_obj1_opti_pose = []
        pw_T_objs_obse_pose_list = []
        # simulate getting names of objects
        objects_name_list = ["cracker", "soup"]
        pw_T_obj0_opti_pos = [0.4472889147344443, 0.08677179678403951, 0.0821006075425945]
        pw_T_obj0_opti_ori = [0.52338279, 0.47884367, 0.52129429, -0.47437481]
        pw_T_obj0_opti_pose.append(pw_T_obj0_opti_pos)
        pw_T_obj0_opti_pose.append(pw_T_obj0_opti_ori)
        pw_T_objs_opti_pose_list.append(pw_T_obj0_opti_pose)
        pw_T_obj1_opti_pos = [0.4472889147344443, 0.15677179678403951, 0.05]
        pw_T_obj1_opti_ori = [1.0, 0.0, 0.0, 1.0]
        pw_T_obj1_opti_pose.append(pw_T_obj1_opti_pos)
        pw_T_obj1_opti_pose.append(pw_T_obj1_opti_ori)
        pw_T_objs_opti_pose_list.append(pw_T_obj1_opti_pose)
        for i in range(object_num):
            # simulate getting ground truth of objects from OptiTrack
            opti_T_obj_opti_pos = copy.deepcopy(pw_T_objs_opti_pose_list[i][0])
            opti_T_obj_opti_ori = copy.deepcopy(pw_T_objs_opti_pose_list[i][1])
            optitrack_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+objects_name_list[i]+"/"+objects_name_list[i]+"_real_obj_hor.urdf"),
                                                           opti_T_obj_opti_pos,
                                                           opti_T_obj_opti_ori)
            opti_object = OptitrackPose(objects_name_list[i], optitrack_object_id, opti_T_obj_opti_pos, opti_T_obj_opti_ori, index=i)
            pw_T_obj_opti_objects_list.append(opti_object)
            # simulate getting observation of objects from observation data
            pw_T_obj_obse_pos, pw_T_obj_obse_ori = add_noise_pose(opti_T_obj_opti_pos, opti_T_obj_opti_ori)
            obse_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+objects_name_list[i]+"/"+objects_name_list[i]+"_obse_obj_with_visual_hor.urdf"),
                                                      pw_T_obj_obse_pos,
                                                      pw_T_obj_obse_ori)
            obse_object = ObservationPose(objects_name_list[i], obse_object_id, pw_T_obj_obse_pos, pw_T_obj_obse_ori, index=i)
            pw_T_obj_obse_objects_list.append(obse_object)

        pw_T_base_pos = [0,0,0]
        pw_T_base_ori = [0,0,0,1]
        # compute and load the pose of optitrack_base (only task 4)
        if task_flag == "4":
            qwertyuiop = 1
            
    
    # initialization pose of obse

    boss_obs_pose_CVPF.append(pw_T_obj_obse_objects_list)
    # initial visualisation world model
    # build an object of class "InitialRealworldModel"
    if observation_cheating_flag == False:
        real_world_object = InitialRealworldModel(object_num, ros_listener.current_joint_values, object_flag, p_visualisation)
    elif observation_cheating_flag == True:
        real_world_object = InitialRealworldModel(object_num, 0, object_flag, p_visualisation)
    #initialize the real robot in the pybullet
    real_robot_id = real_world_object.initial_robot(robot_pos=pw_T_rob_sim_pos, robot_orientation=pw_T_rob_sim_ori)
    # initialize the real object in the pybullet
    # real_object_id = real_world_object.initial_target_object(object_pos = pw_T_obj_opti_pos, object_orientation = pw_T_obj_opti_ori)
    if optitrack_working_flag == True:
        contact_obj_id_list = real_world_object.initial_contact_object(pw_T_obj_opti_objects_list)
        # contact_particle_id = real_world_object.initial_contact_object(pw_T_obj_opti_objects_list, pw_T_obj_opti_pos, pw_T_obj_opti_ori)
    elif optitrack_working_flag == False:
        contact_obj_id_list = real_world_object.initial_contact_object(pw_T_obj_obse_objects_list, pw_T_obj_obse_pos, pw_T_obj_obse_ori)
    # build an object of class "Franka_robot"
    franka_robot = Franka_robot(real_robot_id, p_visualisation)
    # initialize sim world (particles)
    # initial_parameter = InitialSimulationModel(particle_num, pw_T_rob_sim_pos, pw_T_rob_sim_ori, obse_obj_pos_init, obse_obj_ori_init)
    initial_parameter = InitialSimulationModel(object_num, particle_num, pw_T_rob_sim_pos, pw_T_rob_sim_ori, 
                                               pw_T_obj_obse_objects_list,
                                               pw_T_base_pos, pw_T_base_ori,
                                               p_visualisation,
                                               update_style_flag, change_sim_time, task_flag, object_flag)
    # get estimated object
    if observation_cheating_flag == False:
        estimated_object_set = initial_parameter.initial_and_set_simulation_env(ros_listener.current_joint_values)
    elif observation_cheating_flag == True:
        estimated_object_set = initial_parameter.initial_and_set_simulation_env(0)

    
    boss_est_pose_CVPF.append(estimated_object_set) # [esti_obj1, esti_obj2]
    initial_parameter.initial_and_set_simulation_env_CV(ros_listener.current_joint_values)

    # display particles
    if visualisation_particle_flag == True:
        if run_PBPF_flag == True:
            initial_parameter.display_particle()
        if run_CVPF_flag == True:
            initial_parameter.display_particle_CV()
    
    # load object in the sim world    
    if visualisation_flag == True and visualisation_mean == True:
        for obj_index in range(object_num):
            esti_obj_name = estimated_object_set[obj_index].est_obj_name
            esti_obj_pos = estimated_object_set[obj_index].pos
            esti_obj_ori = estimated_object_set[obj_index].ori
            if run_PBPF_flag == True:
                estimated_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+esti_obj_name+"/"+esti_obj_name+"_est_obj_with_visual_PB_hor.urdf"),
                                                               esti_obj_pos,
                                                               esti_obj_ori)
                estimated_object_set[obj_index].esti_obj_id = estimated_object_id
            if run_CVPF_flag == True:
                estimated_object_id_CV = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+esti_obj_name+"/"+esti_obj_name+"_est_obj_with_visual_CV_hor.urdf"),
                                                                  esti_obj_pos,
                                                                  esti_obj_ori)
                estimated_object_set[obj_index].esti_obj_id = estimated_object_id_CV
            
    
    # initial_parameter.particle_cloud #parameter of particle
    # initial_parameter.pybullet_particle_env_collection #env of simulation
    # initial_parameter.fake_robot_id_collection #id of robot in simulation
    # initial_parameter.particle_no_visual_id_collection #id of particle in simulation
    
    # run the simulation
    Flag = True
    # compute obse object old pose
    # obse_obj_pos_old = copy.deepcopy(pw_T_obj_obse_pos)
    # obse_obj_ori_old = copy.deepcopy(pw_T_obj_obse_ori)
    # obse_obj_pos_old_CV = copy.deepcopy(pw_T_obj_obse_pos)
    # obse_obj_ori_old_CV = copy.deepcopy(pw_T_obj_obse_ori)
    # compute pose of robot arm
    if observation_cheating_flag == False:
        rob_link_9_pose_old_PB = p_visualisation.getLinkState(real_robot_id, 9)
        rob_link_9_pose_old_CV = p_visualisation.getLinkState(real_robot_id, 9)
    elif observation_cheating_flag == True:
        rob_link_9_pose_old_PB = p_visualisation.getBasePositionAndOrientation(real_robot_id)
        rob_link_9_pose_old_CV = p_visualisation.getBasePositionAndOrientation(real_robot_id)
    rob_link_9_ang_old_PB = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_old_PB[1])
    rob_link_9_ang_old_CV = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_old_CV[1])
    print("Welcome to Our Approach !")
    robot1 = PFMove(object_num)
    robot2 = PFMoveCV(object_num)
    while not rospy.is_shutdown():
        
        # shi fou chong zhi lie biao 
        #panda robot moves in the visualization window
        temp_pw_T_obj_obse_objs_list = []
        temp_pw_T_obj_opti_objs_list = []
        if observation_cheating_flag == False:
            for i in range(object_num):
                # need to change
                franka_robot.fanka_robot_move(ros_listener.current_joint_values)
                # get obse data
                obse_is_fresh = True
                try:
                    if object_flag == "cracker":
                        latest_obse_time = listener.getLatestCommonTime('/panda_link0', '/cracker')
                    if object_flag == "soup":
                        latest_obse_time = listener.getLatestCommonTime('/panda_link0', '/soup')
                    #print("latest_obse_time: ",latest_obse_time.to_sec())
                    #print("rospy.get_time: ",rospy.get_time())
                    if (rospy.get_time() - latest_obse_time.to_sec()) < 0.1:
                        if object_flag == "cracker":
                            (trans,rot) = listener.lookupTransform('/panda_link0', '/cracker', rospy.Time(0))
                        if object_flag == "soup":
                            (trans,rot) = listener.lookupTransform('/panda_link0', '/soup', rospy.Time(0))
                        obse_is_fresh = True
                        # print("obse is FRESH")
                    else:
                        # obse has not been updating for a while
                        obse_is_fresh = False
                        print("obse is NOT fresh")
                    # break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print("can not find tf")
                rob_T_obj_obse_pos = list(trans)
                rob_T_obj_obse_ori = list(rot)
                rob_T_obj_obse_3_3 = transformations.quaternion_matrix(rob_T_obj_obse_ori)
                rob_T_obj_obse_4_4 = rotation_4_4_to_transformation_4_4(rob_T_obj_obse_3_3,rob_T_obj_obse_pos)
                pw_T_obj_obse = np.dot(pw_T_rob_sim_4_4, rob_T_obj_obse_4_4)
                pw_T_obj_obse_pos = [pw_T_obj_obse[0][3],pw_T_obj_obse[1][3],pw_T_obj_obse[2][3]]
                pw_T_obj_obse_ori = transformations.quaternion_from_matrix(pw_T_obj_obse)
                
                pw_T_obj_obse_name = pw_T_obj_obse_objects_list[i].obse_obj_name
                pw_T_obj_obse_id = pw_T_obj_obse_objects_list[i].obse_obj_id
                obse_object = ObservationPose(pw_T_obj_obse_name, pw_T_obj_obse_id, pw_T_obj_obse_pos, pw_T_obj_obse_ori, index=i)
                temp_pw_T_obj_obse_objs_list.append(obse_object)
                if visualisation_flag == True:
                    display_real_object_in_visual_model(obse_object_id, pw_T_obj_obse_pos, pw_T_obj_obse_ori)
                
                # get ground truth data 
                if optitrack_working_flag == True:
                    rob_T_obj_opti_4_4 = compute_transformation_matrix(ros_listener.robot_pos,
                                                                       ros_listener.robot_ori,
                                                                       ros_listener.object_pos,
                                                                       ros_listener.object_ori)
                    pw_T_rob_sim_3_3 = transformations.quaternion_matrix(pw_T_rob_sim_ori)
                    pw_T_rob_sim_4_4 = rotation_4_4_to_transformation_4_4(pw_T_rob_sim_3_3, pw_T_rob_sim_pos)
                    pw_T_obj_opti_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_obj_opti_4_4)
                    pw_T_obj_opti_pos = [pw_T_obj_opti_4_4[0][3], pw_T_obj_opti_4_4[1][3], pw_T_obj_opti_4_4[2][3]]
                    pw_T_obj_opti_ori = transformations.quaternion_from_matrix(pw_T_obj_opti_4_4)
                    
                    pw_T_obj_opti_name = pw_T_obj_opti_objects_list[i].opti_obj_name
                    pw_T_obj_opti_id = pw_T_obj_opti_objects_list[i].opti_obj_id
                    opti_object = OptitrackPose(pw_T_obj_opti_name, pw_T_obj_opti_id, pw_T_obj_opti_pos, pw_T_obj_opti_ori, index=i)
                    temp_pw_T_obj_opti_objs_list.append(opti_object)

            pw_T_obj_obse_objects_list = copy.deepcopy(temp_pw_T_obj_obse_objs_list)
            pw_T_obj_opti_objects_list = copy.deepcopy(temp_pw_T_obj_opti_objs_list)
            
            # compute distance between old robot and cur robot (position and angle)
            rob_link_9_pose_cur_PB = p_visualisation.getLinkState(real_robot_id, 9)
            rob_link_9_pose_cur_CV = p_visualisation.getLinkState(real_robot_id, 9)
            rob_link_9_ang_cur_PB = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_cur_PB[1])
            rob_link_9_ang_cur_CV = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_cur_CV[1])
            dis_robcur_robold_PB = compute_pos_err_bt_2_points(rob_link_9_pose_cur_PB[0], rob_link_9_pose_old_PB[0])
            dis_robcur_robold_CV = compute_pos_err_bt_2_points(rob_link_9_pose_cur_CV[0], rob_link_9_pose_old_CV[0])
            
        elif observation_cheating_flag == True:
            # need to add robot move
            cheat_robot_move(real_robot_id, x=0.0, y=0.028, z=0.0)
            for i in range(object_num):
                # get ground truth data
                pw_T_obj_opti_name = pw_T_obj_opti_objects_list[i].opti_obj_name
                pw_T_obj_opti_id = pw_T_obj_opti_objects_list[i].opti_obj_id
                pw_T_obj_opti_pos, pw_T_obj_opti_ori = get_item_pos(p_visualisation, pw_T_obj_opti_id)
                opti_object = OptitrackPose(pw_T_obj_opti_name, pw_T_obj_opti_id, pw_T_obj_opti_pos, pw_T_obj_opti_ori, index=i)
                temp_pw_T_obj_opti_objs_list.append(opti_object)
                # cheat get obse data
                pw_T_obj_obse_name = pw_T_obj_obse_objects_list[i].obse_obj_name
                pw_T_obj_obse_id = pw_T_obj_obse_objects_list[i].obse_obj_id
                pw_T_obj_obse_pos, pw_T_obj_obse_ori = add_noise_pose(pw_T_obj_opti_pos, pw_T_obj_opti_ori)
                obse_object = ObservationPose(pw_T_obj_obse_name, pw_T_obj_obse_id, pw_T_obj_obse_pos, pw_T_obj_obse_ori, index=i)
                temp_pw_T_obj_obse_objs_list.append(obse_object)
                if visualisation_flag == True:
                    display_real_object_in_visual_model(pw_T_obj_obse_id, pw_T_obj_obse_pos, pw_T_obj_obse_ori)

            pw_T_obj_obse_objects_list = copy.deepcopy(temp_pw_T_obj_obse_objs_list)
            pw_T_obj_opti_objects_list = copy.deepcopy(temp_pw_T_obj_opti_objs_list)
            
            rob_link_9_pose_cur_PB = p_visualisation.getBasePositionAndOrientation(real_robot_id)
            rob_link_9_pose_cur_CV = p_visualisation.getBasePositionAndOrientation(real_robot_id)
            dis_robcur_robold_PB = compute_pos_err_bt_2_points(rob_link_9_pose_cur_PB[0], rob_link_9_pose_old_PB[0])
            dis_robcur_robold_CV = compute_pos_err_bt_2_points(rob_link_9_pose_cur_CV[0], rob_link_9_pose_old_CV[0])
        # compute distance between old obse obj and cur obse obj (position and angle)
        # dis_betw_cur_and_old = compute_pos_err_bt_2_points(obse_obj_pos_cur, obse_obj_pos_old)
        # ang_betw_cur_and_old = compute_ang_err_bt_2_points(obse_obj_ori_cur, obse_obj_ori_old)
        # dis_betw_cur_and_old_CV = compute_pos_err_bt_2_points(obse_obj_pos_cur, obse_obj_pos_old_CV)
        # ang_betw_cur_and_old_CV = compute_ang_err_bt_2_points(obse_obj_ori_cur, obse_obj_ori_old_CV)
        # compute distance between old robot arm and cur robot arm (position and angle)
        
        # update according to the pose
        if update_style_flag == "pose":
            # PBPF algorithm
            if run_PBPF_flag == True:
                if (dis_robcur_robold_PB > d_thresh):
                    # judgement for any particles contact
                    if robot1.isAnyParticleInContact():
                        simRobot_touch_par_flag = 1
                        t_begin_PBPF = time.time()
                        flag_update_num_PB = flag_update_num_PB + 1
                        pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                        # execute PBPF algorithm movement
                        if observation_cheating_flag == False:
                            robot1.real_robot_control_PB(ros_listener.current_joint_values, # joints of robot arm
                                                         pw_T_obj_obse_objects_pose_list,
                                                         do_obs_update=obse_is_fresh) # flag for judging obse work
                        elif observation_cheating_flag == True:
                            robot1.real_robot_control_PB(ros_listener.current_joint_values, # joints of robot arm
                                                         pw_T_obj_obse_objects_pose_list,
                                                         do_obs_update=obse_is_fresh) # flag for judging obse work
                        rob_link_9_pose_old_PB = copy.deepcopy(rob_link_9_pose_cur_PB)
                        if visualisation_flag == True and optitrack_working_flag == True:
                            for obj_index in range(object_num):
                                optitrack_object_id = pw_T_obj_opti_objects_list[obj_index].opti_obj_id
                                pw_T_obj_opti_pos = pw_T_obj_opti_objects_list[obj_index].pos
                                pw_T_obj_opti_ori = pw_T_obj_opti_objects_list[obj_index].ori
                                display_real_object_in_visual_model(optitrack_object_id, pw_T_obj_opti_pos, pw_T_obj_opti_ori)
                        elif visualisation_flag == True and optitrack_working_flag == False:
                            for obj_index in range(object_num):
                                optitrack_object_id = pw_T_obj_opti_objects_list[obj_index].opti_obj_id
                                display_real_object_in_visual_model(optitrack_object_id, ros_listener.fake_opti_pos, ros_listener.fake_opti_ori)
                        # print("Average time of updating: ",np.mean(robot1.times))
                        t_finish_PBPF = time.time()
                        PBPF_time_cosuming_list.append(t_finish_PBPF - t_begin_PBPF)
                        # print("Time consuming:", t_finish_PBPF - t_begin_PBPF)
                        simRobot_touch_par_flag = 0
                    else:
                        # also update the pose of the robot arm in the simulation when particles are not touched
                        robot1.motion_update_PB_parallelised(initial_parameter.pybullet_particle_env_collection,
                                                             initial_parameter.fake_robot_id_collection,
                                                             ros_listener.current_joint_values)
            # CVPF algorithm
            if run_CVPF_flag == True:
                # if (dis_betw_cur_and_old_CV > d_thresh_CV) or (ang_betw_cur_and_old_CV > a_thresh_CV) or (dis_robcur_robold_CV > d_thresh_CV):
                if (dis_robcur_robold_CV > d_thresh_CV) and robot2.isAnyParticleInContact():
                    flag_update_num_CV = flag_update_num_CV + 1
                    boss_obs_pose_CVPF.append(pw_T_obj_obse_objects_list)
                    # execute CVPF algorithm movement
                    pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                    robot2.update_particle_filter_CV(pw_T_obj_obse_objects_pose_list, # [obse_obj1_pose, obse_obj2_pose]
                                                     do_obs_update=obse_is_fresh) # flag for judging obse work
                    rob_link_9_pose_old_CV = copy.deepcopy(rob_link_9_pose_cur_CV)
                    if visualisation_flag == True:
                        for obj_index in range(object_num):
                            optitrack_object_id = pw_T_obj_opti_objects_list[obj_index].opti_obj_id
                            pw_T_obj_opti_pos = pw_T_obj_opti_objects_list[obj_index].pos
                            pw_T_obj_opti_ori = pw_T_obj_opti_objects_list[obj_index].ori
                            display_real_object_in_visual_model(optitrack_object_id, pw_T_obj_opti_pos, pw_T_obj_opti_ori)
        # update according to the time
        elif update_style_flag == "time":
            while True:
                # PBPF algorithm
                if run_PBPF_flag == True:
                    if robot1.isAnyParticleInContact():
                        simRobot_touch_par_flag = 1
                        t_begin_PBPF = time.time()
                        flag_update_num_PB = flag_update_num_PB + 1
                        pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                        # execute PBPF algorithm movement
                        if observation_cheating_flag == False:
                            robot1.real_robot_control_PB(ros_listener.current_joint_values, # joints of robot arm
                                                         pw_T_obj_obse_objects_pose_list, # [obse_obj1_pose, obse_obj2_pose]
                                                         do_obs_update=obse_is_fresh) # flag for judging obse work
                        elif observation_cheating_flag == True:
                            real_rob_pos, real_rob_ori = get_item_pos(p_visualisation, real_robot_id)
                            print(real_rob_pos)
                            print(len(real_rob_pos))
                            input("stop")
                            robot1.real_robot_control_PB(real_rob_pos, # joints of robot arm
                                                         pw_T_obj_obse_objects_pose_list, # [obse_obj1_pose, obse_obj2_pose]
                                                         do_obs_update=obse_is_fresh) # flag for judging obse work
                        if visualisation_flag == True and optitrack_working_flag == True:
                            for obj_index in range(object_num):
                                optitrack_object_id = pw_T_obj_opti_objects_list[obj_index].opti_obj_id
                                pw_T_obj_opti_pos = pw_T_obj_opti_objects_list[obj_index].pos
                                pw_T_obj_opti_ori = pw_T_obj_opti_objects_list[obj_index].ori
                                display_real_object_in_visual_model(optitrack_object_id, pw_T_obj_opti_pos, pw_T_obj_opti_ori)
                        elif visualisation_flag == True and optitrack_working_flag == False:
                            for obj_index in range(object_num):
                                optitrack_object_id = pw_T_obj_opti_objects_list[obj_index].opti_obj_id
                                display_real_object_in_visual_model(optitrack_object_id, ros_listener.fake_opti_pos, ros_listener.fake_opti_ori)
                        t_finish_PBPF = time.time()
                        PBPF_time_cosuming_list.append(t_finish_PBPF - t_begin_PBPF)
                        simRobot_touch_par_flag = 0
                    else:
                        if observation_cheating_flag == False:
                            robot1.motion_update_PB_parallelised(initial_parameter.pybullet_particle_env_collection,
                                                                 initial_parameter.fake_robot_id_collection,
                                                                 ros_listener.current_joint_values)
                        elif observation_cheating_flag == True:
                            real_rob_pos, real_rob_ori = get_item_pos(p_visualisation, real_robot_id)
                            robot1.motion_update_PB_parallelised(initial_parameter.pybullet_particle_env_collection,
                                                                 initial_parameter.fake_robot_id_collection,
                                                                 real_rob_pos)
                # CVPF algorithm
                if run_CVPF_flag == True:
                    if robot2.isAnyParticleInContact():
                        flag_update_num_CV = flag_update_num_CV + 1
                        boss_obs_pose_CVPF.append(pw_T_obj_obse_objects_list)
                        # execute CVPF algorithm movement
                        pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                        robot2.update_particle_filter_CV(pw_T_obj_obse_objects_pose_list,
                                                         do_obs_update=obse_is_fresh) # flag for judging obse work
                        if visualisation_flag == True and optitrack_working_flag == True:
                            for obj_index in range(object_num):
                                optitrack_object_id = pw_T_obj_opti_objects_list[obj_index].opti_obj_id
                                pw_T_obj_opti_pos = pw_T_obj_opti_objects_list[obj_index].pos
                                pw_T_obj_opti_ori = pw_T_obj_opti_objects_list[obj_index].ori
                                display_real_object_in_visual_model(optitrack_object_id, pw_T_obj_opti_pos, pw_T_obj_opti_ori)
                pf_update_rate.sleep()
                break    
        t_end_while = time.time()
        if Flag is False:
            break
        
    p_visualisation.disconnect()
    


