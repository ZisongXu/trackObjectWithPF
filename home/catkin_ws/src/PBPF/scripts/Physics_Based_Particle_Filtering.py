#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:57:49 2021

@author: 12106
"""
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
from Center_T_Point_for_Ray import Center_T_Point_for_Ray
# CVPF Pose list (motion model)
boss_obs_pose_CVPF = []
boss_est_pose_CVPF = []


#Class of Physics-based Particle Filtering
class PBPFMove():
    def __init__(self, obj_num=0):
        # initialize internal parameters
        self.obj_num = obj_num
        self.particle_cloud = copy.deepcopy(initial_parameter.particle_cloud)
        self.particle_no_visual_id_collection = copy.deepcopy(initial_parameter.particle_no_visual_id_collection)
        self.pybullet_env_id_collection = copy.deepcopy(initial_parameter.pybullet_particle_env_collection)
        self.pybullet_sim_fake_robot_id_collection = copy.deepcopy(initial_parameter.fake_robot_id_collection)
        self.pybullet_sim_other_object_id_collection = copy.deepcopy(initial_parameter.other_object_id_collection)
        self.other_obj_num = other_obj_num
        self.joint_num = 7
        self.object_estimate_pose_x = []
        self.object_estimate_pose_y = []
        self.object_real_____pose_x = []
        self.object_real_____pose_y = []
        self.do_obs_update = True
        self.rays_id_list = []
        self.camera_parPoint_list = []
        self.ray_list_empty = True
        
    def get_real_robot_joint(self, pybullet_env_id, real_robot_id):
        real_robot_joint_list = []
        for index in range(self.joint_num):
            real_robot_info = pybullet_env_id.getJointState(real_robot_id,index)[0]
            real_robot_joint_list.append(real_robot_info)
        return real_robot_joint_list
        
    def set_real_robot_JointPosition(self,pybullet_env, robot, position):
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
    def update_particle_filter_PB(self, real_robot_joint_pos, pw_T_obj_obse_objects_pose_list, do_obs_update):
        global flag_record_obse
        global flag_record_PBPF
        global flag_record
        
        self.camera_parPoint_list = []
        self.rays_id_list = []
        
        self.do_obs_update = do_obs_update
        pybullet_sim_env = self.pybullet_env_id_collection
        fake_robot_id = self.pybullet_sim_fake_robot_id_collection
        self.times = []
        t1 = time.time()
        # motion model
        self.motion_update_PB_parallelised(pybullet_sim_env, fake_robot_id, real_robot_joint_pos)
        t2 = time.time()
        self.times.append(t2-t1)
        # observation model
        if do_obs_update:
            self.observation_update_PB(pw_T_obj_obse_objects_pose_list)
        if (version == "ray" or version == "multiray") and do_obs_update == False:
            self.resample_particles_update(pw_T_obj_obse_objects_pose_list)
            self.set_paticle_in_each_sim_env()
        # Compute mean of particles
        object_estimate_pose, dis_std_list, ang_std_list = self.compute_estimate_pos_of_object(self.particle_cloud)
        
        # publish pose of particles
        publish_par_pose_info(self.particle_cloud)
        publish_esti_pose_info(object_estimate_pose)
        
        if show_ray == True:
            p_sim.removeAllUserDebugItems()
        
        return object_estimate_pose, dis_std_list, ang_std_list, self.particle_cloud
    
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

                # check contact 
                pmin, pmax = pw_T_par_sim_pw_env.getAABB(pw_T_par_sim_id)
                collide_ids = pw_T_par_sim_pw_env.getOverlappingObjects(pmin, pmax)
                length = len(collide_ids)
                for t_i in range(length):
                    # print("body id: ",collide_ids[t_i][1])
                    if collide_ids[t_i][1] == 8 or collide_ids[t_i][1] == 9 or collide_ids[t_i][1] == 10 or collide_ids[t_i][1] == 11:
                        return True
        return False
    
    # update particle cloud particle angle
    def update_partcile_cloud_pose_PB(self, index, obj_index, x, y, z, ori, linearVelocity, angularVelocity):
        self.particle_cloud[index][obj_index].pos = [x, y, z]
        self.particle_cloud[index][obj_index].ori = copy.deepcopy(ori)
        self.particle_cloud[index][obj_index].linearVelocity = linearVelocity
        self.particle_cloud[index][obj_index].angularVelocity = angularVelocity
#        self.particle_cloud[index][obj_index].rayTraceList = [1,2,3]
        if version == "ray" and self.do_obs_update == False:
            camera_parPoint = []
            pw_T_par_sim_pos = copy.deepcopy([x, y, z])
            # pybullet_sim_env[index]
            rayTest_info = p_sim.rayTest(pw_T_cam_tf_pos, pw_T_par_sim_pos)
            camera_parPoint.append(pw_T_cam_tf_pos)
            camera_parPoint.append(pw_T_par_sim_pos)
            self.camera_parPoint_list.append(camera_parPoint)
            hit_obj_id = rayTest_info[0][0]
            print(hit_obj_id)
            if hit_obj_id == -1:
                weight = 0.1
#                weight = 0.9
            else:
                weight = 0.9
#                weight = 0.1
            self.particle_cloud[index][obj_index].w = weight
        
        elif version == "multiray" and self.do_obs_update == False:
            # need to change
            camera_parPoint = []
            pw_T_parC_pos = copy.deepcopy([x, y, z])
            pw_T_parC_ori = copy.deepcopy(ori) # x, y, z, w
            pw_T_parC_ori = quaternion_correction(pw_T_parC_ori)
            pw_T_parC_3_3 = transformations.quaternion_matrix(pw_T_parC_ori)
            pw_T_parC_4_4 = rotation_4_4_to_transformation_4_4(pw_T_parC_3_3, pw_T_parC_pos)
            
            point_list, point_pos_list = generate_point_for_ray(pw_T_parC_pos, pw_T_parC_4_4, obj_index)
            point_pos_list.append(pw_T_parC_pos)
            camera_pos_list = []
            list_length = len(point_pos_list)
            for point_index in range(list_length):
                camera_pos_list.append(pw_T_cam_tf_pos)
            # pybullet_sim_env[index]
            rayTestBatch_info = p_sim.rayTestBatch(camera_pos_list, point_pos_list)
            camera_parPoint.append(camera_pos_list)
            camera_parPoint.append(point_pos_list)
            self.camera_parPoint_list.append(camera_parPoint)
            
            if show_ray == True:
                ray_id_list = []
                for list_index in range(list_length):
                    ray_id = p_sim.addUserDebugLine(camera_pos_list[list_index], point_pos_list[list_index], [0,1,0], 2)
                ray_id = p_sim.addUserDebugLine(point_pos_list[0], point_pos_list[1], [0,1,0], 5)
                ray_id_list.append(ray_id)
                ray_id = p_sim.addUserDebugLine(point_pos_list[0], point_pos_list[2], [0,1,0], 5)
                ray_id_list.append(ray_id)
                ray_id = p_sim.addUserDebugLine(point_pos_list[0], point_pos_list[4], [0,1,0], 5)
                ray_id_list.append(ray_id)
                ray_id = p_sim.addUserDebugLine(point_pos_list[1], point_pos_list[3], [0,1,0], 5)
                ray_id_list.append(ray_id)
                ray_id = p_sim.addUserDebugLine(point_pos_list[1], point_pos_list[5], [0,1,0], 5)
                ray_id_list.append(ray_id)
                ray_id = p_sim.addUserDebugLine(point_pos_list[2], point_pos_list[3], [0,1,0], 5)
                ray_id_list.append(ray_id)
                ray_id = p_sim.addUserDebugLine(point_pos_list[2], point_pos_list[6], [0,1,0], 5)
                ray_id_list.append(ray_id)
                ray_id = p_sim.addUserDebugLine(point_pos_list[3], point_pos_list[7], [0,1,0], 5)
                ray_id_list.append(ray_id)
                ray_id = p_sim.addUserDebugLine(point_pos_list[4], point_pos_list[5], [0,1,0], 5)
                ray_id_list.append(ray_id)
                ray_id = p_sim.addUserDebugLine(point_pos_list[4], point_pos_list[6], [0,1,0], 5)
                ray_id_list.append(ray_id)
                ray_id = p_sim.addUserDebugLine(point_pos_list[5], point_pos_list[7], [0,1,0], 5)
                ray_id_list.append(ray_id)
                ray_id = p_sim.addUserDebugLine(point_pos_list[6], point_pos_list[7], [0,1,0], 5)
                ray_id_list.append(ray_id)
                self.rays_id_list.append(ray_id_list)
#            publish_ray_trace_info(camera_pos_list, point_pos_list)
            
            point_hit_num = 0
            for point_index in range(list_length):
                hit_obj_id = rayTestBatch_info[point_index][0]
                if hit_obj_id == -1:
                    point_hit_num = point_hit_num - 1
                else:
                    point_hit_num = point_hit_num + 1
            if point_hit_num > 0:
                weight = 0.9
            else:
                weight = 0.1
            self.particle_cloud[index][obj_index].w = weight
            
            
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
        for oto_index in range(self.other_obj_num):
            collision_detection_obj_id.append(self.pybullet_sim_other_object_id_collection[oto_index])

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
            if motion_noise == True:
                nTries = 0
                while nTries < 20:
                    nTries=nTries+1
                    # print("checking")
                    flag = 0
                    length_collision_detection_obj_id = len(collision_detection_obj_id)
                    for check_num in range(length_collision_detection_obj_id-1):
                        pybullet_env.stepSimulation()
                        contacts = pybullet_env.getContactPoints(bodyA=collision_detection_obj_id[check_num], # robot, other object...
                                                                 bodyB=collision_detection_obj_id[-1]) # main(target) object
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
                if nTries >= 10: # This means we could not find a non-colliding particle position.
                    print("WARNING: Could not find a non-colliding particle position after motion noise. Moving particle object to noise-less pose. Particle index, object index ", index, obj_index)
                    pybullet_env.resetBasePositionAndOrientation(pw_T_par_sim_id, sim_par_cur_pos, sim_par_cur_ori)

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
        new_angle = p_sim.getEulerFromQuaternion(pb_quat)
        P_quat = p_sim.getQuaternionFromEuler(new_angle)
        # pipe.send()
        return normal_x, normal_y, normal_z, P_quat
    
    # observation model
    def observation_update_PB(self, pw_T_obj_obse_objects_pose_list):
        for obj_index in range(self.obj_num):
            obse_obj_pos = pw_T_obj_obse_objects_pose_list[obj_index].pos
            obse_obj_ori = pw_T_obj_obse_objects_pose_list[obj_index].ori # pybullet x,y,z,w
            
            # make sure theta between -pi and pi
            obse_obj_ori_corr = quaternion_correction(obse_obj_ori)
#            obse_obj_quat = Quaternion(x=nois_obj_ori[0],y=nois_obj_ori[1],z=nois_obj_ori[2],w=nois_obj_ori[3]) # w,x,y,z
#            cos_theta_over_2 = obse_obj_quat.w
#            sin_theta_over_2 = math.sqrt(obse_obj_quat.x ** 2 + obse_obj_quat.y ** 2 + obse_obj_quat.z ** 2)
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
                obse_obj_quat = Quaternion(x=obse_obj_ori_corr[0], 
                                           y=obse_obj_ori_corr[1], 
                                           z=obse_obj_ori_corr[2], 
                                           w=obse_obj_ori_corr[3]) # Quaternion(): w,x,y,z
                par_quat = Quaternion(x=par_ori[0], y=par_ori[1], z=par_ori[2], w=par_ori[3])
                err_bt_par_obse = par_quat * obse_obj_quat.inverse
                err_bt_par_obse_corr = quaternion_correction([err_bt_par_obse.x,err_bt_par_obse.y,err_bt_par_obse.z,err_bt_par_obse.w])
                err_bt_par_obse_corr_quat = Quaternion(x=err_bt_par_obse_corr[0], 
                                           y=err_bt_par_obse_corr[1], 
                                           z=err_bt_par_obse_corr[2], 
                                           w=err_bt_par_obse_corr[3])
                cos_theta_over_2 = err_bt_par_obse_corr_quat.w
                sin_theta_over_2 = math.sqrt(err_bt_par_obse_corr_quat.x ** 2 + err_bt_par_obse_corr_quat.y ** 2 + err_bt_par_obse_corr_quat.z ** 2)
                theta_over_2 = math.atan2(sin_theta_over_2, cos_theta_over_2)
                theta = theta_over_2 * 2.0
                weight_ang = self.normal_distribution(theta, mean, boss_sigma_obs_ang)
                weight = weight_xyz * weight_ang
                
                if version == "ray":
                    camera_parPoint = []
                    pw_T_par_sim_pos = [particle_x, particle_y, particle_z]
                    # pybullet_sim_env[index]
                    rayTest_info = p_sim.rayTest(pw_T_cam_tf_pos, pw_T_par_sim_pos)
                    camera_parPoint.append(pw_T_cam_tf_pos)
                    camera_parPoint.append(pw_T_par_sim_pos)
                    self.camera_parPoint_list.append(camera_parPoint)
                    # ray_id = p_sim.addUserDebugLine(pw_T_cam_tf_pos, pw_T_par_sim_pos)
                    # self.rays_id_list[index].append(ray_id)
                            
                    hit_obj_id = rayTest_info[0][0]
                    if hit_obj_id == -1:
                        weight = weight
                    else:
                        weight = weight / 2.0
                elif version == "multiray":
                    # need to change
                    camera_parPoint = []
                    pw_T_parC_pos = copy.deepcopy([particle_x, particle_y, particle_z])
                    pw_T_parC_ori = copy.deepcopy(par_ori) # x, y, z, w
                    pw_T_parC_3_3 = transformations.quaternion_matrix(pw_T_parC_ori)
                    pw_T_parC_4_4 = rotation_4_4_to_transformation_4_4(pw_T_parC_3_3, pw_T_parC_pos)
                    
                    point_list, point_pos_list = generate_point_for_ray(pw_T_parC_pos, pw_T_parC_4_4, obj_index)
                    point_pos_list.append(pw_T_parC_pos)
                    camera_pos_list = []
                    list_length = len(point_pos_list)
                    for point_index in range(list_length):
                        camera_pos_list.append(pw_T_cam_tf_pos)
                    # pybullet_sim_env[index]
                    rayTestBatch_info = p_sim.rayTestBatch(camera_pos_list, point_pos_list)
                    camera_parPoint.append(camera_pos_list)
                    camera_parPoint.append(point_pos_list)
                    self.camera_parPoint_list.append(camera_parPoint)
                    
                    if show_ray == True:
                        ray_id_list = []
                        for list_index in range(list_length):
                            ray_id = p_sim.addUserDebugLine(camera_pos_list[list_index], point_pos_list[list_index], [0,1,0], 2)
                            ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[0], point_pos_list[1], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[0], point_pos_list[2], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[0], point_pos_list[4], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[1], point_pos_list[3], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[1], point_pos_list[5], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[2], point_pos_list[3], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[2], point_pos_list[6], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[3], point_pos_list[7], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[4], point_pos_list[5], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[4], point_pos_list[6], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[5], point_pos_list[7], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        ray_id = p_sim.addUserDebugLine(point_pos_list[6], point_pos_list[7], [0,1,0], 5)
                        ray_id_list.append(ray_id)
                        self.rays_id_list.append(ray_id_list)
                    
                    point_hit_num = 0
                    for point_index in range(list_length):
                        hit_obj_id = rayTestBatch_info[point_index][0]
                        if hit_obj_id == -1:
                            point_hit_num = point_hit_num - 1
                        else:
                            point_hit_num = point_hit_num + 1
                    if point_hit_num > 0:
                        weight = weight / 2.0
                    else:
                        weight = weight

                particle[obj_index].w = weight
            # old resample function
            # Flag = self.normalize_particles()
            # self.resample_particles()
            # new resample function
        self.resample_particles_update(pw_T_obj_obse_objects_pose_list)
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
    def resample_particles_update(self, pw_T_obj_obse_objects_pose_list):
        local_pick_particle_rate = pick_particle_rate
        if self.do_obs_update == False:
            local_pick_particle_rate = 0.0
        pw_T_obj_obse_objs_pose_list = copy.deepcopy(pw_T_obj_obse_objects_pose_list)
        n_particle = len(self.particle_cloud)
        par_num_on_obse = int(math.ceil(n_particle * local_pick_particle_rate))
        par_num_for_resample = int(n_particle) - int(par_num_on_obse)

        newParticles_list = [[]*self.obj_num for _ in range(n_particle)]
        for obj_index in range(self.obj_num):
            obse_obj_pos = pw_T_obj_obse_objs_pose_list[obj_index].pos
            obse_obj_ori = pw_T_obj_obse_objs_pose_list[obj_index].ori # pybullet x,y,z,w

            particles_w = []
            # newParticles = []
            base_w = 0
            base_w_list = []
            base_w_list.append(base_w)
            particle_array_list = []
            # compute sum of weight
            for particle in self.particle_cloud:
                particles_w.append(particle[obj_index].w)
                base_w = base_w + particle[obj_index].w
                base_w_list.append(base_w)
            w_sum = sum(particles_w)
            r = random.uniform(0, w_sum)

            for index in range(par_num_for_resample):
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
            for index_leftover in range(par_num_on_obse):
                index = index + 1
                particle = Particle(self.particle_cloud[index_leftover][obj_index].par_name,
                                    self.particle_cloud[index][obj_index].visual_par_id,
                                    self.particle_cloud[index][obj_index].no_visual_par_id,
                                    obse_obj_pos,
                                    obse_obj_ori,
                                    1.0/particle_num, 
                                    index,
                                    self.particle_cloud[index_leftover][obj_index].linearVelocity,
                                    self.particle_cloud[index_leftover][obj_index].angularVelocity)
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
        dis_std_list = []
        ang_std_list = []
        # remenber after resampling weight of each particle is the same
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
            ###################################
            esti_obj_pos_x = x_set/w_set
            esti_obj_pos_y = y_set/w_set
            esti_obj_pos_z = z_set/w_set
            esti_obj_pos = [esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z]
            esti_obj_ori_x = q[0]
            esti_obj_ori_y = q[1]
            esti_obj_ori_z = q[2]
            esti_obj_ori_w = q[3]
            esti_obj_ori = [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]
            mean_pose = [esti_obj_pos, esti_obj_ori]
            dis_std, ang_std = self.compute_std(mean_pose, particle_cloud)
            ###################################
            est_obj_pose = Object_Pose(particle[obj_index].par_name, estimated_object_set[obj_index].obj_id, [x_set/w_set, y_set/w_set, z_set/w_set],  [q[0], q[1], q[2], q[3]], obj_index)
            esti_objs_cloud.append(est_obj_pose)
            dis_std_list.append(dis_std)
            ang_std_list.append(ang_std)
        return esti_objs_cloud, dis_std_list, ang_std_list

    def compute_std(self, mean_pose, particle_cloud):
        mean_pos = copy.deepcopy(mean_pose[0])
        mean_ori = copy.deepcopy(mean_pose[1]) # x,y,z,w
        dis_list = []
        ang_list = []
        for index, particle in enumerate(particle_cloud):
            pos_set = copy.deepcopy(particle[obj_index].pos)
            q = quaternion_correction(particle[obj_index].ori)
            ori_set = copy.deepcopy(q)
            dis_mean_eachPar = compute_pos_err_bt_2_points(pos_set, mean_pos)
            ang_mean_eachPar = compute_ang_err_bt_2_points(ori_set, mean_ori)
            dis_list.append(dis_mean_eachPar)
            ang_list.append(ang_mean_eachPar)
        dis_std = np.std(dis_list)
        ang_std = np.std(ang_list)
        return dis_std, ang_std

    def rotation_4_4_to_transformation_4_4(self, rotation_4_4,pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4


#Class of Constant-velocity Particle Filtering
class CVPFMove():
    def __init__(self, obj_num=0):
        # init internals   
        self.obj_num = obj_num
        self.particle_cloud_CV = copy.deepcopy(initial_parameter.particle_cloud_CV)
        self.particle_no_visual_id_collection_CV = copy.deepcopy(initial_parameter.particle_no_visual_id_collection_CV)
        self.pybullet_env_id_collection_CV = copy.deepcopy(initial_parameter.pybullet_particle_env_collection_CV)
        self.pybullet_sim_fake_robot_id_collection = copy.deepcopy(initial_parameter.fake_robot_id_collection)
        self.pybullet_sim_other_object_id_collection = copy.deepcopy(initial_parameter.other_object_id_collection)
        self.object_estimate_pose_x = []
        self.object_estimate_pose_y = []
        self.object_real_____pose_x = []
        self.object_real_____pose_y = []
        self.rays_id_list = []
        self.ray_list_empty = True

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
            
            
        if (version == "ray" or version == "multiray") and do_obs_update == False:
            self.resample_particles_CV_update(pw_T_obj_obse_objects_pose_list)
            self.set_paticle_in_each_sim_env_CV()
            
        # Compute mean of particles
        object_estimate_pose_CV, dis_std_list, ang_std_list = self.compute_estimate_pos_of_object(self.particle_cloud_CV)
        
        boss_est_pose_CVPF.append(object_estimate_pose_CV)

        # publish pose of particles
        publish_par_pose_info(self.particle_cloud_CV)
        publish_esti_pose_info(object_estimate_pose_CV)
        return object_estimate_pose_CV, dis_std_list, ang_std_list, self.particle_cloud_CV

    def isAnyParticleInContact(self):
        for index, particle in enumerate(self.particle_cloud_CV):
            for obj_index in range(object_num):
                # get pose from particle
                pw_T_par_sim_pw_env = self.pybullet_env_id_collection_CV[index]
#                pw_T_par_sim_id = self.particle_no_visual_id_collection[index][obj_index]
                pw_T_par_sim_id = particle[obj_index].no_visual_par_id
#                sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(self.pybullet_env_id_collection[index], pw_T_par_sim_id)
                sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(pw_T_par_sim_pw_env, pw_T_par_sim_id)
                # check contact 
                pmin, pmax = pw_T_par_sim_pw_env.getAABB(pw_T_par_sim_id)
                collide_ids = pw_T_par_sim_pw_env.getOverlappingObjects(pmin, pmax)
                length = len(collide_ids)
                for t_i in range(length):
                    # print("body id: ",collide_ids[t_i][1])
                    if collide_ids[t_i][1] == 8 or collide_ids[t_i][1] == 9 or collide_ids[t_i][1] == 10 or collide_ids[t_i][1] == 11:
                        return True
        return False
    
    def robot_arm_move_CV(self, position):
        for index, pw_sim_env in enumerate(self.pybullet_env_id_collection_CV):
            fake_rob_id = self.pybullet_sim_fake_robot_id_collection[index]
            num_joints = 9
            for joint_index in range(num_joints):
                if joint_index == 7 or joint_index == 8:
                    pw_sim_env.resetJointState(fake_rob_id,
                                                 joint_index+2,
                                                 targetValue=position[joint_index])
                else:
                    pw_sim_env.resetJointState(fake_rob_id,
                                                 joint_index,
                                                 targetValue=position[joint_index])
                
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
#            obse_obj_quat = Quaternion(x=nois_obj_ori[0],y=nois_obj_ori[1],z=nois_obj_ori[2],w=nois_obj_ori[3]) # w,x,y,z
#            cos_theta_over_2 = obse_obj_quat.w
#            sin_theta_over_2 = math.sqrt(obse_obj_quat.x ** 2 + obse_obj_quat.y ** 2 + obse_obj_quat.z ** 2)
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
                obse_obj_quat = Quaternion(x=obse_obj_ori_corr[0], 
                                           y=obse_obj_ori_corr[1], 
                                           z=obse_obj_ori_corr[2], 
                                           w=obse_obj_ori_corr[3]) # Quaternion(): w,x,y,z
                par_quat = Quaternion(x=par_ori[0], y=par_ori[1], z=par_ori[2], w=par_ori[3])
                err_bt_par_obse = par_quat * obse_obj_quat.inverse
                cos_theta_over_2 = err_bt_par_obse.w
                sin_theta_over_2 = math.sqrt(err_bt_par_obse.x ** 2 + err_bt_par_obse.y ** 2 + err_bt_par_obse.z ** 2)
                theta_over_2 = math.atan2(sin_theta_over_2, cos_theta_over_2)
                theta = theta_over_2 * 2
                weight_ang = self.normal_distribution(theta, mean, boss_sigma_obs_ang)
                weight = weight_xyz * weight_ang
                
                if version == "ray":
                    pw_T_par_sim_pos = [particle_x, particle_y, particle_z]
                    # pybullet_sim_env[index]
                    rayTest_info = p_sim.rayTest(pw_T_cam_tf_pos, pw_T_par_sim_pos)
                    # ray_id = p_sim.addUserDebugLine(pw_T_cam_tf_pos, pw_T_par_sim_pos)
                    # self.rays_id_list[index].append(ray_id)
                    hit_obj_id = rayTest_info[0][0]
                    if hit_obj_id == -1:
                        weight = weight
                    else:
                        weight = weight / 2.0
                elif version == "multiray":
                    # need to change
                    pw_T_parC_pos = copy.deepcopy([particle_x, particle_y, particle_z])
                    pw_T_parC_ori = copy.deepcopy(par_ori) # x, y, z, w
                    pw_T_parC_3_3 = transformations.quaternion_matrix(pw_T_parC_ori)
                    pw_T_parC_4_4 = rotation_4_4_to_transformation_4_4(pw_T_parC_3_3, pw_T_parC_pos)
                    
                    point_list, point_pos_list = generate_point_for_ray(pw_T_parC_pos, pw_T_parC_4_4, obj_index)
                    point_pos_list.append(pw_T_parC_pos)
                    camera_pos_list = []
                    list_length = len(point_pos_list)
                    for point_index in range(list_length):
                        camera_pos_list.append(pw_T_cam_tf_pos)
                    # pybullet_sim_env[index]
                    rayTestBatch_info = p_sim.rayTestBatch(camera_pos_list, point_pos_list)
                    point_hit_num = 0
                    for point_index in range(list_length):
                        hit_obj_id = rayTestBatch_info[point_index][0]
                        if hit_obj_id == -1:
                            point_hit_num = point_hit_num - 1
                        else:
                            point_hit_num = point_hit_num + 1
                    if point_hit_num > 0:
                        weight = weight / 2.0
                    else:
                        weight = weight
            
                particle[obj_index].w = weight
        # old resample function
        # Flag = self.normalize_particles_CV()
        # self.resample_particles_CV()
        # new resample function
        self.resample_particles_CV_update(pw_T_obj_obse_objects_pose_list)
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
            pw_T_parN_ori = transformations.quaternion_from_matrix(pw_T_parN)
            pw_T_parN_ori = quaternion_correction(pw_T_parN_ori)
            # pw_T_parN_ang = pybullet_env.getEulerFromQuaternion(pw_T_parN_ori)
            
            # add noise on particle filter
            normal_x = self.add_noise_2_par(pw_T_parN_pos[0])
            normal_y = self.add_noise_2_par(pw_T_parN_pos[1])
            normal_z = self.add_noise_2_par(pw_T_parN_pos[2])
            
            # quat = copy.deepcopy(pw_T_obj_obse_objects_list[obj_index].ori) # use ori from dope
            quat = copy.deepcopy(pw_T_parN_ori) # x,y,z,w / use ori from CV
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
            
            
            
            if version == "ray" and self.do_obs_update == False:
                pw_T_par_sim_pos = copy.deepcopy([normal_x, normal_y, normal_z])
                # pybullet_sim_env[index]
                rayTest_info = p_sim.rayTest(pw_T_cam_tf_pos, pw_T_par_sim_pos)
                
                hit_obj_id = rayTest_info[0][0]
                if hit_obj_id == -1:
                    weight = 0.1
                else:
                    weight = 0.9
    #            self.particle_cloud[index][obj_index].w = weight
            elif version == "multiray" and self.do_obs_update == False:
                # need to change
                pw_T_parC_pos = copy.deepcopy([normal_x, normal_y, normal_z])
                pw_T_parC_ori = copy.deepcopy(pb_quat) # x, y, z, w
                pw_T_parC_ori = quaternion_correction(pw_T_parC_ori)
                pw_T_parC_3_3 = transformations.quaternion_matrix(pw_T_parC_ori)
                pw_T_parC_4_4 = rotation_4_4_to_transformation_4_4(pw_T_parC_3_3, pw_T_parC_pos)
                
                point_list, point_pos_list = generate_point_for_ray(pw_T_parC_pos, pw_T_parC_4_4, obj_index)
                point_pos_list.append(pw_T_parC_pos)
                camera_pos_list = []
                list_length = len(point_pos_list)
                for point_index in range(list_length):
                    camera_pos_list.append(pw_T_cam_tf_pos)
                # pybullet_sim_env[index]
                rayTestBatch_info = p_sim.rayTestBatch(camera_pos_list, point_pos_list)
                point_hit_num = 0
                for point_index in range(list_length):
                    hit_obj_id = rayTestBatch_info[point_index][0]
                    if hit_obj_id == -1:
                        point_hit_num = point_hit_num - 1
                    else:
                        point_hit_num = point_hit_num + 1
                if point_hit_num > 0:
                    weight = 0.9
                else:
                    weight = 0.1
                        
            self.particle_cloud_CV[index][obj_index].w = weight
            


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
        
    def resample_particles_CV_update(self, pw_T_obj_obse_objects_pose_list):
        local_pick_particle_rate = pick_particle_rate
        if self.do_obs_update == False:
            local_pick_particle_rate = 0.0
        pw_T_obj_obse_objs_pose_list = copy.deepcopy(pw_T_obj_obse_objects_pose_list)
        n_particle = len(self.particle_cloud)
        par_num_on_obse = int(math.ceil(n_particle * local_pick_particle_rate))
        par_num_for_resample = int(n_particle) - int(par_num_on_obse)

        newParticles_list = [[]*self.obj_num for _ in range(n_particle)]
        for obj_index in range(self.obj_num):
            obse_obj_pos = pw_T_obj_obse_objs_pose_list[obj_index].pos
            obse_obj_ori = pw_T_obj_obse_objs_pose_list[obj_index].ori # pybullet x,y,z,w

            particles_w = []
            # newParticles = []
            base_w = 0
            base_w_list = []
            base_w_list.append(base_w)
            particle_array_list = []
            # compute sum of weight
            for particle in self.particle_cloud:
                particles_w.append(particle[obj_index].w)
                base_w = base_w + particle[obj_index].w
                base_w_list.append(base_w)
            w_sum = sum(particles_w)
            r = random.uniform(0, w_sum)

            for index in range(par_num_for_resample):
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
            for index_leftover in range(par_num_on_obse):
                index = index + 1
                particle = Particle(self.particle_cloud[index_leftover][obj_index].par_name,
                                    self.particle_cloud[index][obj_index].visual_par_id,
                                    self.particle_cloud[index][obj_index].no_visual_par_id,
                                    obse_obj_pos,
                                    obse_obj_ori,
                                    1.0/particle_num, 
                                    index,
                                    self.particle_cloud[index_leftover][obj_index].linearVelocity,
                                    self.particle_cloud[index_leftover][obj_index].angularVelocity)
                newParticles_list[index].append(particle)

#                newParticles.append(particle)
        self.particle_cloud = copy.deepcopy(newParticles_list)
        
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
        dis_std_list = []
        ang_std_list = []
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
            ###################################
            esti_obj_pos_x = x_set/w_set
            esti_obj_pos_y = y_set/w_set
            esti_obj_pos_z = z_set/w_set
            esti_obj_pos = [esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z]
            esti_obj_ori_x = q[0]
            esti_obj_ori_y = q[1]
            esti_obj_ori_z = q[2]
            esti_obj_ori_w = q[3]
            esti_obj_ori = [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]
            mean_pose = [esti_obj_pos, esti_obj_ori]
            dis_std, ang_std = self.compute_std(mean_pose, particle_cloud)
            ###################################
            est_obj_pose = Object_Pose(particle[obj_index].par_name,
                                       estimated_object_set[obj_index].obj_id,
                                       [x_set/w_set, y_set/w_set, z_set/w_set],
                                       [q[0], q[1], q[2], q[3]], obj_index)
            esti_objs_cloud.append(est_obj_pose)
            dis_std_list.append(dis_std)
            ang_std_list.append(ang_std)
        return esti_objs_cloud, dis_std_list, ang_std_list
    
    def compute_std(self, mean_pose, particle_cloud):
        mean_pos = copy.deepcopy(mean_pose[0])
        mean_ori = copy.deepcopy(mean_pose[1]) # x,y,z,w
        dis_list = []
        ang_list = []
        for index, particle in enumerate(particle_cloud):
            pos_set = copy.deepcopy(particle[obj_index].pos)
            q = quaternion_correction(particle[obj_index].ori)
            ori_set = copy.deepcopy(q)
            dis_mean_eachPar = compute_pos_err_bt_2_points(pos_set, mean_pos)
            ang_mean_eachPar = compute_ang_err_bt_2_points(ori_set, mean_ori)
            dis_list.append(dis_mean_eachPar)
            ang_list.append(ang_mean_eachPar)
        dis_std = np.std(dis_list)
        ang_std = np.std(ang_list)
        return dis_std, ang_std
    
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
    obj1_ori_quat = quaternion_correction(obj1_ori) # x,y,z,w
    obj2_ori_quat = quaternion_correction(obj2_ori) # x,y,z,w

    #[w, x, y, z]
    obj1_quat = Quaternion(x = obj1_ori_quat[0], y = obj1_ori_quat[1], z = obj1_ori_quat[2], w = obj1_ori_quat[3]) # Quaternion(): w,x,y,z
    obj2_quat = Quaternion(x = obj2_ori_quat[0], y = obj2_ori_quat[1], z = obj2_ori_quat[2], w = obj2_ori_quat[3]) # Quaternion(): w,x,y,z
    diff_bt_o1_o2 = obj2_quat * obj1_quat.inverse
    cos_theta_over_2 = diff_bt_o1_o2.w
    sin_theta_over_2 = math.sqrt(diff_bt_o1_o2.x ** 2 + diff_bt_o1_o2.y ** 2 + diff_bt_o1_o2.z ** 2)
    theta_over_2 = math.atan2(sin_theta_over_2, cos_theta_over_2)
    theta = theta_over_2 * 2
    theta = abs(theta)
    return theta

def compute_diff_bt_two_pose(obj_index, particle_cloud_pub, pw_T_obj_obse_pose_new):
    par_cloud_for_compute = copy.deepcopy(particle_cloud_pub)
    obj_obse_pose_new = copy.deepcopy(pw_T_obj_obse_pose_new)
    obj_obse_pos_new = obj_obse_pose_new[0]
    obj_obse_ori_new = obj_obse_pose_new[1]
    par_dis_list = []
    par_ang_list = []
    par_cloud_length = len(par_cloud_for_compute)
    for par_index in range(par_cloud_length):
        par_pos = par_cloud_for_compute[par_index][obj_index].pos
        par_ori = par_cloud_for_compute[par_index][obj_index].ori

        dis_obseCur_parOld = compute_pos_err_bt_2_points(obj_obse_pos_new, par_pos)
        ang_obseCur_parOld = compute_ang_err_bt_2_points(obj_obse_ori_new, par_ori)
        par_dis_list.append(dis_obseCur_parOld)
        par_ang_list.append(ang_obseCur_parOld)

    minDis_obseCur_parOld = min(par_dis_list)
    minAng_obseCur_parOld = min(par_ang_list)
    return minDis_obseCur_parOld, minAng_obseCur_parOld

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
    # new_angle = p_sim.getEulerFromQuaternion(pb_quat)
    # x_angle = new_angle[0]
    # y_angle = new_angle[1]
    # z_angle = new_angle[2]
    # x_angle = sim_par_cur_ang[0]
    # y_angle = sim_par_cur_ang[1]
    # z_angle = sim_par_cur_ang[2]
    # P_quat = p_sim.getQuaternionFromEuler([x_angle, y_angle, z_angle])
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
    theta = theta_over_2 * 2.0
    while theta >= math.pi:
        theta = theta - 2.0*math.pi
    while theta <= -math.pi:
        theta = theta + 2.0*math.pi
    new_quaternion = [math.sin(theta/2.0)*(new_quat.x/sin_theta_over_2), math.sin(theta/2.0)*(new_quat.y/sin_theta_over_2), math.sin(theta/2.0)*(new_quat.z/sin_theta_over_2), math.cos(theta/2.0)]
    #if theta >= math.pi or theta <= -math.pi:
    #    new_quaternion = [-quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
    #    return new_quaternion
    #return quaternion # x,y,z,w
    return new_quaternion

#def publish_ray_trace_info(particle_cloud_pub):
#    par_pose_list = list(range(particle_num))
#    for par_index in range(particle_num):
#        par_pose = particle_pose()
#        obj_pose_list = []
#        for obj_index in range(object_num):
#            obj_pose = object_pose()
#            obj_info = particle_cloud_pub[par_index][obj_index]
#            obj_pose.name = obj_info.par_name
#            obj_pose.pose.position.x = obj_info.pos[0]
#            obj_pose.pose.position.y = obj_info.pos[1]
#            obj_pose.pose.position.z = obj_info.pos[2]
#            obj_pose_list.append(obj_pose)
#        par_pose.objects = obj_pose_list
#        par_pose_list[par_index] = par_pose
#        
#    par_list.particles = par_pose_list
#    pub_ray_trace.publish(par_list)

def publish_par_pose_info(particle_cloud_pub):
    par_pose_list = list(range(particle_num))
    for par_index in range(particle_num):
        par_pose = particle_pose()
        obj_pose_list = []
        for obj_index in range(object_num):
            obj_pose = object_pose()
            obj_info = particle_cloud_pub[par_index][obj_index]
            obj_pose.name = obj_info.par_name
            obj_pose.pose.position.x = obj_info.pos[0]
            obj_pose.pose.position.y = obj_info.pos[1]
            obj_pose.pose.position.z = obj_info.pos[2]
            obj_pose.pose.orientation.x = obj_info.ori[0]
            obj_pose.pose.orientation.y = obj_info.ori[1]
            obj_pose.pose.orientation.z = obj_info.ori[2]
            obj_pose.pose.orientation.w = obj_info.ori[3]
            obj_pose_list.append(obj_pose)
        par_pose.objects = obj_pose_list
        par_pose_list[par_index] = par_pose
        
    par_list.particles = par_pose_list
    pub_par_pose.publish(par_list)
            
def publish_esti_pose_info(estimated_object_set):
    esti_pose_list = []
    for obj_index in range(object_num):
        esti_pose = object_pose()
        esti_obj_info = estimated_object_set[obj_index]
        esti_pose.name = esti_obj_info.obj_name
        esti_pose.pose.position.x = esti_obj_info.pos[0]
        esti_pose.pose.position.y = esti_obj_info.pos[1]
        esti_pose.pose.position.z = esti_obj_info.pos[2]
        esti_pose.pose.orientation.x = esti_obj_info.ori[0]
        esti_pose.pose.orientation.y = esti_obj_info.ori[1]
        esti_pose.pose.orientation.z = esti_obj_info.ori[2]
        esti_pose.pose.orientation.w = esti_obj_info.ori[3]
        esti_pose_list.append(esti_pose)
    esti_list.objects = esti_pose_list 
    pub_esti_pose.publish(esti_list)

    # need to change
    pub = rospy.Publisher('PBPF_pose', PoseStamped, queue_size = 1)
    pose_PBPF = PoseStamped()
    pose_PBPF.pose.position.x = esti_obj_info.pos[0]
    pose_PBPF.pose.position.y = esti_obj_info.pos[1]
    pose_PBPF.pose.position.z = esti_obj_info.pos[2]
    pose_PBPF.pose.orientation.x = esti_obj_info.ori[0]
    pose_PBPF.pose.orientation.y = esti_obj_info.ori[1]
    pose_PBPF.pose.orientation.z = esti_obj_info.ori[2]
    pose_PBPF.pose.orientation.w = esti_obj_info.ori[3]
    pub.publish(pose_PBPF)

# need to change
def process_esti_pose_from_rostopic(estimated_object_set):
    esti_pose_list = []
    for obj_index in range(object_num):
        esti_obj_info = copy.deepcopy(estimated_object_set[obj_index])
        esti_obj_pos_x = esti_obj_info.pos[0]
        esti_obj_pos_y = esti_obj_info.pos[1]
        esti_obj_pos_z = esti_obj_info.pos[2]
        esti_obj_ori_x = esti_obj_info.ori[0]
        esti_obj_ori_y = esti_obj_info.ori[1]
        esti_obj_ori_z = esti_obj_info.ori[2]
        esti_obj_ori_w = esti_obj_info.ori[3]
        esti_pose = [[esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z], [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]]
        esti_pose_list.append(esti_pose)
    return esti_pose_list

def generate_point_for_ray(pw_T_c_pos, pw_T_parC_4_4, obj_index):
    vector_list = [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
                   [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]]
#                   [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
#                   [1,0.5,0.5], [1,0.5,-0.5], [1,-0.5,0.5], [1,-0.5,-0.5],
#                   [-1,0.5,0.5], [-1,0.5,-0.5], [-1,-0.5,0.5], [-1,-0.5,-0.5],
#                   [0.5,1,0.5], [0.5,1,-0.5], [-0.5,1,0.5], [-0.5,1,-0.5],
#                   [0.5,-1,0.5], [0.5,-1,-0.5], [-0.5,-1,0.5], [-0.5,-1,-0.5],
#                   [0.5,0.5,1], [0.5,-0.5,1], [-0.5,0.5,1], [-0.5,-0.5,1],
#                   [0.5,0.5,-1], [0.5,-0.5,-1], [-0.5,0.5,-1], [-0.5,-0.5,-1]]
    r = math.sqrt(2)
    if object_name_list[obj_index] == "soup":
        vector_list = [[0,0,1], [0,0,-1],
                       [2,2,1], [2,-2,1], [-2,2,1], [-2,-2,1], [r,r,1], [r,-r,1], [-r,r,1], [-r,-r,1],
                       [2,2,0.5], [2,-2,0.5], [-2,2,0.5], [-2,-2,0.5], [r,r,0.5], [r,-r,0.5], [-r,r,0.5], [-r,-r,0.5],
                       [2,2,0], [2,-2,0], [-2,2,0], [-2,-2,0], [r,r,0], [r,-r,0], [-r,r,0], [-r,-r,0],
                       [2,2,-0.5], [2,-2,-0.5], [-2,2,-0.5], [-2,-2,-0.5], [r,r,-0.5], [r,-r,-0.5], [-r,r,-0.5], [-r,-r,-0.5],
                       [2,2,-1], [2,-2,-1], [-2,2,-1], [-2,-2,-1], [r,r,-1], [r,-r,-1], [-r,r,-1], [-r,-r,-1]]
    point_list = []
    point_pos_list = []
    for index in range(len(vector_list)):
        parC_T_p_x_new = vector_list[index][0] * x_w/2
        parC_T_p_y_new = vector_list[index][1] * y_l/2
        parC_T_p_z_new = vector_list[index][2] * z_h/2
        parC_T_p_pos = [parC_T_p_x_new, parC_T_p_y_new, parC_T_p_z_new]
        parC_T_p_ori = [0, 0, 0, 1] # x, y, z, w
        parC_T_p_3_3 = transformations.quaternion_matrix(parC_T_p_ori)
        parC_T_p_4_4 = rotation_4_4_to_transformation_4_4(parC_T_p_3_3, parC_T_p_pos)
        pw_T_p_4_4 = np.dot(pw_T_parC_4_4, parC_T_p_4_4)
        pw_T_p_pos = [pw_T_p_4_4[0][3], pw_T_p_4_4[1][3], pw_T_p_4_4[2][3]]
        pw_T_p_ori = transformations.quaternion_from_matrix(pw_T_p_4_4)
        pw_T_p_pose = Center_T_Point_for_Ray(pw_T_p_pos, pw_T_p_ori, parC_T_p_4_4, index)
        point_list.append(pw_T_p_pose)
        point_pos_list.append(pw_T_p_pos)
    return point_list, point_pos_list


def track_fk_sim_world():
    if show_ray == True:
        p_track_fk_env = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT,GUI_SERVER
    else:
        p_track_fk_env = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT,GUI_SERVER
    p_track_fk_env.setAdditionalSearchPath(pybullet_data.getDataPath())
    track_fk_plane_id = p_track_fk_env.loadURDF("plane.urdf")
    p_track_fk_env.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0.3,0.1,0.2]) 
    track_fk_rob_id = p_track_fk_env.loadURDF(os.path.expanduser("~/project/data/bullet3-master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
                                              [0, 0, 0],
                                              [0, 0, 0, 1],
                                              useFixedBase=1)
    if task_flag == "1":
        track_fk_obst_id = p_track_fk_env.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_obstacle.urdf"),
                                                   pw_T_obst_opti_pos,
                                                   pw_T_obst_opti_ori,
                                                   useFixedBase=1)
    return p_track_fk_env, track_fk_rob_id, track_fk_plane_id

def track_fk_world_rob_mv(p_sim, sim_rob_id, position):
    num_joints = 9
    for joint_index in range(num_joints):
        if joint_index == 7 or joint_index == 8:
            p_sim.resetJointState(sim_rob_id,
                                  joint_index+2,
                                  targetValue=position[joint_index])
        else:
            p_sim.resetJointState(sim_rob_id,
                                  joint_index,
                                  targetValue=position[joint_index])
    
# ctrl-c write down the error file
def signal_handler(sig, frame):
    print("DOPE Jump Rate:", count_DOPE_jumping_time / all_frame)
    sys.exit()

if __name__ == '__main__':
    rospy.init_node('PF_for_obse') # ros node
    signal.signal(signal.SIGINT, signal_handler) # interrupt judgment
    # publish
    pub_ray_trace = rospy.Publisher('/ray_trace_list', particle_list, queue_size = 10)
    ray_trace_list = particle_list()
    pub_par_pose = rospy.Publisher('/par_list', particle_list, queue_size = 10)
    par_list = particle_list()
    pub_esti_pose = rospy.Publisher('/esti_obj_list', estimated_obj_pose, queue_size = 10)
    esti_list = estimated_obj_pose()
    # only for drawing box
    publish_DOPE_pose_flag = True

    with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
        parameter_info = yaml.safe_load(file)
    
    gazebo_flag = parameter_info['gazebo_flag']
    # scene
    task_flag = parameter_info['task_flag'] # parameter_info['task_flag']
    # which algorithm to run
    run_alg_flag = parameter_info['run_alg_flag'] # PBPF/CVPF
    # update mode (pose/time)
    update_style_flag = parameter_info['update_style_flag'] # time/pose
    # observation model
    pick_particle_rate = parameter_info['pick_particle_rate']
    # the flag is used to determine whether the robot touches the particle in the simulation
    simRobot_touch_par_flag = 0
    object_num = parameter_info['object_num']
    robot_num = 1
    check_dope_work_flag_init = 0
    if task_flag == "4": 
        other_obj_num = 1 # parameter_info['other_obj_num']
    else:
        other_obj_num = 0 # parameter_info['other_obj_num']

    if update_style_flag == "pose":
        particle_num = parameter_info['particle_num']
    elif update_style_flag == "time":
        if run_alg_flag == "PBPF":
            particle_num = parameter_info['particle_num']
        elif run_alg_flag == "CVPF":
            particle_num = parameter_info['particle_num']
    object_name_list = parameter_info['object_name_list']
    version = parameter_info['version'] # old/ray/multiray
    ray_point_num = parameter_info['ray_point_num']
    if run_alg_flag == 'CVPF':
        particle_num = 140
    print("This is "+update_style_flag+" update in scene"+task_flag)    
    # some parameters
    d_thresh = 0.005
    a_thresh = 0.01
    d_thresh_obse = 0.15
    a_thresh_obse = math.pi * 2 / 3.0
    d_thresh_CV = 0.0002
    a_thresh_CV = 0.0010
    flag_record = 0
    flag_record_obse = 0
    flag_record_PBPF = 0
    flag_record_CVPF = 0
    flag_update_num_CV = 0
    flag_update_num_PB = 0
    change_sim_time = 1.0/90
    if run_alg_flag == "PBPF":
        boss_pf_update_interval_in_real = 0.16
    elif run_alg_flag == "CVPF":
        boss_pf_update_interval_in_real = 0.02
    pf_update_rate = rospy.Rate(1.0/boss_pf_update_interval_in_real)
    # # error in xyz axis obse before recalibrating
    # boss_sigma_obs_x = 0.03973017808163751 / 2.0
    # boss_sigma_obs_y = 0.01167211468503462 / 2.0
    # boss_sigma_obs_z = 0.02820930183351492 / 2.0
    # # new obse error
    # boss_sigma_obs_x = 0.032860982 * 2.0
    # boss_sigma_obs_y = 0.012899399 * 1.5
    # boss_sigma_obs_z = 0.01
    # boss_sigma_obs_ang_init = 0.0216773873 * 2.0

    # Motion model Noise
    pos_noise = 0.001 * 5.0
    # ang_noise = 0.05 * 1.0
    ang_noise = 0.05 * 3.0
    pos_noise = 0.005
    ang_noise = 0.05 
    motion_noise = True
    show_ray = True
    # pos_noise = 0.0
    # ang_noise = 0.0
    # motion_noise = True

    # Standard deviation of computing the weight
    # boss_sigma_obs_ang = 0.216773873
    # boss_sigma_obs_ang = 0.0216773873
    # boss_sigma_obs_ang = 0.0216773873 * 4
    # boss_sigma_obs_ang = 0.0216773873 * 20
    boss_sigma_obs_ang = 0.0216773873 * 60
    # boss_sigma_obs_pos = 0.038226405
    # boss_sigma_obs_pos = 0.004
    boss_sigma_obs_pos = 0.25 # 0.02 need to increase

    mass_mean = 0.380 # 0.380
    mass_sigma = 0.5
    friction_mean = 0.1
    friction_sigma = 0.3
    restitution_mean = 0.9
    restitution_sigma = 0.2
    count_DOPE_jumping_time = 0
    all_frame = 0
    
    PBPF_time_cosuming_list = []
    
    pw_T_obst_opti_pos = [0.7188993998723022, 0.2767650526046564, 0.1258681365201122]
    pw_T_obst_opti_ori = [ 7.07173367e-01, -6.19453133e-03, 4.42925544e-04, 7.07012914e-01]

    # multi-objects/robot list
    pw_T_rob_sim_pose_list_alg = []
    pw_T_obj_obse_obj_list_alg = []
    pw_T_obj_obse_oto_list_alg = []
    # need to change
    dis_std_list = [d_thresh_obse]
    ang_std_list = [a_thresh_obse]
    # build an object of class "Ros_Listener"
    ros_listener = Ros_Listener()
    create_scene = Create_Scene(object_num, robot_num, other_obj_num)
    listener = tf.TransformListener()
    time.sleep(0.5)
    
    pw_T_rob_sim_pose_list_alg = create_scene.initialize_robot()
    pw_T_rob_sim_4_4 = pw_T_rob_sim_pose_list_alg[0].trans_matrix
    
    pw_T_obj_obse_obj_list_alg, trans_ob, rot_ob = create_scene.initialize_object()

    for obj_index in range(other_obj_num):
        pw_T_obj_obse_oto_list_alg = create_scene.initialize_base_of_cheezit()

    initial_parameter = InitialSimulationModel(object_num, robot_num, other_obj_num, particle_num, 
                                               pw_T_rob_sim_pose_list_alg, 
                                               pw_T_obj_obse_obj_list_alg,
                                               pw_T_obj_obse_oto_list_alg,
                                               update_style_flag, change_sim_time)
    # get estimated object
    if run_alg_flag == "PBPF":
        estimated_object_set, particle_cloud_pub = initial_parameter.initial_and_set_simulation_env()
    if run_alg_flag == "CVPF":
        estimated_object_set, particle_cloud_pub = initial_parameter.initial_and_set_simulation_env_CV()
        boss_est_pose_CVPF.append(estimated_object_set) # [esti_obj1, esti_obj2]
    
    # publish particles/estimated object
    publish_par_pose_info(particle_cloud_pub)
    publish_esti_pose_info(estimated_object_set)
    estimated_object_set_old = copy.deepcopy(estimated_object_set)
    estimated_object_set_old_list = process_esti_pose_from_rostopic(estimated_object_set_old)

    if version == "ray" or version == "multiray":
        while True:
            try:
                (trans_camera, rot_camera) = listener.lookupTransform('/panda_link0', '/RealSense', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        rob_T_cam_tf_pos = list(trans_camera)
        rob_T_cam_tf_ori = list(rot_camera)
        rob_T_cam_tf_3_3 = transformations.quaternion_matrix(rob_T_cam_tf_ori)
        rob_T_cam_tf_4_4 = rotation_4_4_to_transformation_4_4(rob_T_cam_tf_3_3, rob_T_cam_tf_pos)
        pw_T_cam_tf = np.dot(pw_T_rob_sim_4_4, rob_T_cam_tf_4_4)
        pw_T_cam_tf_pos = [pw_T_cam_tf[0][3], pw_T_cam_tf[1][3], pw_T_cam_tf[2][3]]
        

    # run the simulation
    Flag = True
    # compute pose of robot arm
    
    # get pose of the end-effector of the robot arm from joints of robot arm 
    p_sim, sim_rob_id, sim_plane_id = track_fk_sim_world()
    track_fk_world_rob_mv(p_sim, sim_rob_id, ros_listener.current_joint_values)

    rob_link_9_pose_old = p_sim.getLinkState(sim_rob_id, 9) # position = rob_link_9_pose_old[0], quaternion = rob_link_9_pose_old[1]
    rob_T_obj_obse_pos_old = list(trans_ob)
    rob_T_obj_obse_ori_old = list(rot_ob)

    print("Welcome to Our Approach !")
    PBPF_alg = PBPFMove(object_num) # PF_alg
    CVPF_alg = CVPFMove(object_num) 

    while not rospy.is_shutdown():
        #panda robot moves in the visualization window
        temp_pw_T_obj_obse_objs_list = []
        track_fk_world_rob_mv(p_sim, sim_rob_id, ros_listener.current_joint_values)
        for obj_index in range(object_num):
            # need to change
            object_name = object_name_list[obj_index]
            
            if object_name == "cracker":
                x_w = 0.159
                y_l = 0.21243700408935547
                z_h = 0.06
            elif object_name == "soup":
                x_w = 0.032829689025878906
                y_l = 0.032829689025878906
                z_h = 0.099
            
            
            use_gazebo = ""
            if gazebo_flag == True:
                use_gazebo = '_noise'
            # get obse data
            obse_is_fresh = True
            obse_is_jumping = False
            try:
                latest_obse_time = listener.getLatestCommonTime('/panda_link0', '/'+object_name+use_gazebo)
                # print("rospy.get_time():")
                # print(rospy.get_time())
                # print("latest_obse_time.to_sec():")
                # print(latest_obse_time.to_sec())
                # old_obse_time = latest_obse_time.to_sec()
                # if (rospy.get_time() - latest_obse_time.to_sec()) < 0.1:
                #     (trans_ob,rot_ob) = listener.lookupTransform('/panda_link0', '/'+object_name+use_gazebo, rospy.Time(0))
                #     obse_is_fresh = True
                #     print("obse is FRESH")

                if check_dope_work_flag_init == 0:
                    check_dope_work_flag_init = 1
                    old_obse_time = latest_obse_time.to_sec()
                # print("latest_obse_time.to_sec():")
                # print(latest_obse_time.to_sec())
                # print("difference:", latest_obse_time.to_sec() - old_obse_time)
                if (latest_obse_time.to_sec() > old_obse_time):
                    (trans_ob,rot_ob) = listener.lookupTransform('/panda_link0', '/'+object_name+use_gazebo, rospy.Time(0))
                    obse_is_fresh = True
                    # print("obse is FRESH")
                else:
                    # obse has not been updating for a while
                    obse_is_fresh = False
                    # print("obse is NOT fresh")
                old_obse_time = latest_obse_time.to_sec()
                # break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("can not find tf")
            rob_T_obj_obse_pos = list(trans_ob)
            rob_T_obj_obse_ori = list(rot_ob)
            rob_T_obj_obse_3_3 = transformations.quaternion_matrix(rob_T_obj_obse_ori)
            rob_T_obj_obse_4_4 = rotation_4_4_to_transformation_4_4(rob_T_obj_obse_3_3,rob_T_obj_obse_pos)

#            if gazebo_flag == True:
#                robpw_T_robga_4_4 = [[1., 0., 0.,    0.],
#                                     [0., 1., 0.,    0.],
#                                     [0., 0., 1., -0.06],
#                                     [0., 0., 0.,    1.]]
#                robpw_T_robga_4_4 = np.array(robpw_T_robga_4_4)                
#                rob_T_obj_obse_4_4 = np.dot(robpw_T_robga_4_4, rob_T_obj_obse_4_4)
            
            
            pw_T_obj_obse = np.dot(pw_T_rob_sim_4_4, rob_T_obj_obse_4_4)
            pw_T_obj_obse_pos = [pw_T_obj_obse[0][3],pw_T_obj_obse[1][3],pw_T_obj_obse[2][3]]
            pw_T_obj_obse_ori = transformations.quaternion_from_matrix(pw_T_obj_obse)

            # need to change when we run alg in multi-object tracking scene
            # in the futrue we need to use "for obj_index in range(object_num):"
            pw_T_obj_obse_pos_new = copy.deepcopy(pw_T_obj_obse_pos)
            pw_T_obj_obse_ori_new = copy.deepcopy(pw_T_obj_obse_ori)
            pw_T_esti_obj_pose_old = copy.deepcopy(estimated_object_set_old_list[obj_index])
            pw_T_esti_obj_pos_old = copy.deepcopy(pw_T_esti_obj_pose_old[0])
            pw_T_esti_obj_ori_old = copy.deepcopy(pw_T_esti_obj_pose_old[1])

            dis_obseCur_estiOld = compute_pos_err_bt_2_points(pw_T_obj_obse_pos_new, pw_T_esti_obj_pos_old)
            ang_obseCur_estiOld = compute_ang_err_bt_2_points(pw_T_obj_obse_ori_new, pw_T_esti_obj_ori_old)
            pw_T_obj_obse_pose_new = [pw_T_obj_obse_pos_new, pw_T_obj_obse_ori_new]

            minDis_obseCur_parOld, minAng_obseCur_parOld = compute_diff_bt_two_pose(obj_index, particle_cloud_pub, pw_T_obj_obse_pose_new)            
            
            all_frame = all_frame + 1
            if minDis_obseCur_parOld > 0.10 or minAng_obseCur_parOld > math.pi * 1 / 2.0:
                # print("DOPE becomes crazy")
                count_DOPE_jumping_time = count_DOPE_jumping_time + 1
                obse_is_fresh = False
                obse_is_jumping = True
            # if dis_obseCur_estiOld > dis_std_list[obj_index]*3 or ang_obseCur_estiOld > ang_std_list[obj_index]*3:
            # if dis_obseCur_estiOld > 0.30:# or ang_obseCur_estiOld > math.pi * 1 / 2.0:
            #     # print("DOPE becomes crazy")
            #     count_DOPE_jumping_time = count_DOPE_jumping_time + 1
            #     obse_is_fresh = False
            #     obse_is_jumping = True

            # only for drawing BOX/ need to change
            if publish_DOPE_pose_flag == True:
                pub_DOPE = rospy.Publisher('DOPE_pose', PoseStamped, queue_size = 1)
                pose_DOPE = PoseStamped()
                pose_DOPE.pose.position.x = pw_T_obj_obse_pos[0]
                pose_DOPE.pose.position.y = pw_T_obj_obse_pos[1]
                pose_DOPE.pose.position.z = pw_T_obj_obse_pos[2]
                pose_DOPE.pose.orientation.x = pw_T_obj_obse_ori[0]
                pose_DOPE.pose.orientation.y = pw_T_obj_obse_ori[1]
                pose_DOPE.pose.orientation.z = pw_T_obj_obse_ori[2]
                pose_DOPE.pose.orientation.w = pw_T_obj_obse_ori[3]
                pub_DOPE.publish(pose_DOPE)

            pw_T_obj_obse_name = object_name
            pw_T_obj_obse_id = 0
            obse_object = Object_Pose(pw_T_obj_obse_name, pw_T_obj_obse_id, pw_T_obj_obse_pos, pw_T_obj_obse_ori, index=obj_index)
            temp_pw_T_obj_obse_objs_list.append(obse_object)
            
        pw_T_obj_obse_objects_list = copy.deepcopy(temp_pw_T_obj_obse_objs_list)
        
        # compute distance between old robot and cur robot (position and angle)
        rob_link_9_pose_cur = p_sim.getLinkState(sim_rob_id, 9)
        rob_link_9_ang_cur = p_sim.getEulerFromQuaternion(rob_link_9_pose_cur[1])
        
        dis_robcur_robold = compute_pos_err_bt_2_points(rob_link_9_pose_cur[0], rob_link_9_pose_old[0])
        
        # only for drawing box
        obse_obj_pos_draw = copy.deepcopy(pw_T_obj_obse_objects_list[0].pos)
        obse_obj_ori_draw = copy.deepcopy(pw_T_obj_obse_objects_list[0].ori) # pybullet x,y,z,w
        pose_DOPE = PoseStamped()
        pose_DOPE.pose.position.x = obse_obj_pos_draw[0]
        pose_DOPE.pose.position.y = obse_obj_pos_draw[1]
        pose_DOPE.pose.position.z = obse_obj_pos_draw[2]
        pose_DOPE.pose.orientation.x = obse_obj_ori_draw[0]
        pose_DOPE.pose.orientation.y = obse_obj_ori_draw[1]
        pose_DOPE.pose.orientation.z = obse_obj_ori_draw[2]
        pose_DOPE.pose.orientation.w = obse_obj_ori_draw[3]
        # print(pose_DOPE)
        pub_DOPE.publish(pose_DOPE)

        # update according to the pose
        if update_style_flag == "pose":
            # PBPF algorithm
            if run_alg_flag == "PBPF":
                if (dis_robcur_robold > d_thresh):
                    # judgement for any particles contact
                    if PBPF_alg.isAnyParticleInContact():
                        simRobot_touch_par_flag = 1
                        t_begin_PBPF = time.time()
                        flag_update_num_PB = flag_update_num_PB + 1
                        pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                        # execute PBPF algorithm movement
                        estimated_object_set, dis_std_list, ang_std_list, particle_cloud_pub = PBPF_alg.update_particle_filter_PB(ros_listener.current_joint_values, # joints of robot arm
                                                                                  pw_T_obj_obse_objects_pose_list,
                                                                                  do_obs_update=obse_is_fresh) # flag for judging obse work
                        rob_link_9_pose_old = copy.deepcopy(rob_link_9_pose_cur)

                        # print("Average time of updating: ",np.mean(PBPF_alg.times))
                        t_finish_PBPF = time.time()
                        PBPF_time_cosuming_list.append(t_finish_PBPF - t_begin_PBPF)
                        # print("Time consuming:", t_finish_PBPF - t_begin_PBPF)
                        simRobot_touch_par_flag = 0
                    else:
                        # also update the pose of the robot arm in the simulation when particles are not touched
                        PBPF_alg.motion_update_PB_parallelised(initial_parameter.pybullet_particle_env_collection,
                                                               initial_parameter.fake_robot_id_collection,
                                                               ros_listener.current_joint_values)
#                else:
#                    PBPF_alg.motion_update_PB_parallelised(initial_parameter.pybullet_particle_env_collection,
#                                                             initial_parameter.fake_robot_id_collection,
#                                                             ros_listener.current_joint_values)
            # CVPF algorithm
            if run_alg_flag == "CVPF":
                # if (dis_betw_cur_and_old_CV > d_thresh_CV) or (ang_betw_cur_and_old_CV > a_thresh_CV) or (dis_robcur_robold_CV > d_thresh_CV):
                if (dis_robcur_robold > d_thresh_CV):
                    if CVPF_alg.isAnyParticleInContact():
                        flag_update_num_CV = flag_update_num_CV + 1
                        boss_obs_pose_CVPF.append(pw_T_obj_obse_objects_list)
                        # execute CVPF algorithm movement
                        pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                        estimated_object_set, dis_std_list, ang_std_list, particle_cloud_pub = CVPF_alg.update_particle_filter_CV(pw_T_obj_obse_objects_pose_list, # [obse_obj1_pose, obse_obj2_pose]
                                                                                  do_obs_update=obse_is_fresh) # flag for judging obse work
                        rob_link_9_pose_old = copy.deepcopy(rob_link_9_pose_cur)
                    else:
                        CVPF_alg.robot_arm_move_CV(ros_listener.current_joint_values) # joints of robot arm
#                else:
#                    CVPF_alg.robot_arm_move_CV(ros_listener.current_joint_values) # joints of robot arm
                    
        # update according to the time
        elif update_style_flag == "time":
            while True:
                # PBPF algorithm
                if run_alg_flag == "PBPF":
                    if PBPF_alg.isAnyParticleInContact() and (dis_robcur_robold > 0.002):
                        simRobot_touch_par_flag = 1
                        t_begin_PBPF = time.time()
                        flag_update_num_PB = flag_update_num_PB + 1
                        pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                        # execute PBPF algorithm movement
                        estimated_object_set, dis_std_list, ang_std_list, particle_cloud_pub = PBPF_alg.update_particle_filter_PB(ros_listener.current_joint_values, # joints of robot arm
                                                                                  pw_T_obj_obse_objects_pose_list, # [obse_obj1_pose, obse_obj2_pose]
                                                                                  do_obs_update=obse_is_fresh) # flag for judging obse work
                        rob_link_9_pose_old = copy.deepcopy(rob_link_9_pose_cur)
                        t_finish_PBPF = time.time()
                        PBPF_time_cosuming_list.append(t_finish_PBPF - t_begin_PBPF)
                        simRobot_touch_par_flag = 0
                    else:
                        PBPF_alg.motion_update_PB_parallelised(initial_parameter.pybullet_particle_env_collection,
                                                               initial_parameter.fake_robot_id_collection,
                                                               ros_listener.current_joint_values)
                # CVPF algorithm
                if run_alg_flag == "CVPF":
                    # if CVPF_alg.isAnyParticleInContact():
                    flag_update_num_CV = flag_update_num_CV + 1
                    boss_obs_pose_CVPF.append(pw_T_obj_obse_objects_list)
                    # execute CVPF algorithm movement
                    pw_T_obj_obse_objects_pose_list = copy.deepcopy(pw_T_obj_obse_objects_list)
                    estimated_object_set, dis_std_list, ang_std_list, particle_cloud_pub = CVPF_alg.update_particle_filter_CV(pw_T_obj_obse_objects_pose_list,
                                                                              do_obs_update=obse_is_fresh) # flag for judging obse work
                    # else:
                    #     CVPF_alg.robot_arm_move_CV(ros_listener.current_joint_values) # joints of robot arm
                        
                estimated_object_set_old = copy.deepcopy(estimated_object_set)
                estimated_object_set_old_list = process_esti_pose_from_rostopic(estimated_object_set_old)
                pf_update_rate.sleep()
                break    
        t_end_while = time.time()
        if Flag is False:
            break

    


