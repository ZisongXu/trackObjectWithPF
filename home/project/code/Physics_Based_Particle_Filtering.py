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


compute_error_flag = False
# panda data frame to record the error and to compare them
# pos
if compute_error_flag == True:
    boss_obse_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    boss_PBPF_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    boss_CVPF_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    boss_err_pos_df = pd.DataFrame(columns=['step','time','pos','alg'],index=[])
    # ang
    boss_obse_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
    boss_PBPF_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
    boss_CVPF_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
    boss_err_ang_df = pd.DataFrame(columns=['step','time','ang','alg'],index=[])
    
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

# CVPF Pose list (motion model)
boss_obs_pose_CVPF = []
boss_est_pose_CVPF = []



#Class of Physics-based Particle Filtering
class PFMove():
    def __init__(self,robot_id=None,real_robot_id=None,object_id=None):
        # initialize internal parameters
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
    def real_robot_control_PB(self,opti_obj_pos_cur, opti_obj_ori_cur, real_robot_joint_pos,
                              nois_obj_pos_cur, nois_obj_ori_cur, do_obs_update):
        # begin to run the PBPF algorithm
        self.update_particle_filter_PB(self.pybullet_env_id_collection, # simulation environment per particle
                                       self.pybullet_sim_fake_robot_id_collection, # fake robot id per sim_env
                                       real_robot_joint_pos, # execution actions of the fake robot
                                       opti_obj_pos_cur, # ground truth pos [x, y, z]
                                       opti_obj_ori_cur, # ground truth ori [x, y, z, w]
                                       nois_obj_pos_cur, # DOPE value pos [x, y, z]
                                       nois_obj_ori_cur, # DOPE value ori [x, y, z, w]
                                       do_obs_update) # flag for judging DOPE work
    
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
                                  opti_obj_pos_cur, opti_obj_ori_cur,
                                  nois_obj_pos_cur, nois_obj_ori_cur, do_obs_update):
        global flag_record_dope
        global flag_record_PBPF
        global flag_record
        global estPB_form_previous
        global estDO_form_previous
        self.times = []
        t1 = time.time()
        # motion model
        self.motion_update_PB_parallelised(pybullet_sim_env, fake_robot_id, real_robot_joint_pos)
        t2 = time.time()
        self.times.append(t2-t1)
        # observation model
        if do_obs_update:
            self.observation_update_PB(nois_obj_pos_cur, nois_obj_ori_cur)
        # Compute mean of particles
        object_estimate_pose = self.compute_estimate_pos_of_object(self.particle_cloud)
        estimated_object_pos = [object_estimate_pose[0], object_estimate_pose[1], object_estimate_pose[2]]
        estimated_object_ori = [object_estimate_pose[3], object_estimate_pose[4], object_estimate_pose[5], object_estimate_pose[6]]
        # display estimated object
        if visualisation_flag == True and visualisation_mean == True:
            self.display_estimated_object_in_visual_model(estimated_object_id, estimated_object_pos, estimated_object_ori)
        # display particles
        if visualisation_particle_flag == True:
            self.display_particle_in_visual_model_PB(self.particle_cloud)
        # self.draw_contrast_figure(estimated_object_pos,observation)
        # compute error and write down to the file
        err_opti_dope_pos = compute_pos_err_bt_2_points(nois_obj_pos_cur,opti_obj_pos_cur)
        err_opti_dope_ang = compute_ang_err_bt_2_points(nois_obj_ori_cur,opti_obj_ori_cur)
        err_opti_dope_ang = angle_correction(err_opti_dope_ang)
        err_opti_PBPF_pos = compute_pos_err_bt_2_points(estimated_object_pos,opti_obj_pos_cur)
        err_opti_PBPF_ang = compute_ang_err_bt_2_points(estimated_object_ori,opti_obj_ori_cur)
        err_opti_PBPF_ang = angle_correction(err_opti_PBPF_ang)
        if publish_PBPF_pose_flag == True:
            pub = rospy.Publisher('PBPF_pose', PoseStamped, queue_size = 1)
            pose_PBPF = PoseStamped()
            pose_PBPF.pose.position.x = estimated_object_pos[0]
            pose_PBPF.pose.position.y = estimated_object_pos[1]
            pose_PBPF.pose.position.z = estimated_object_pos[2]
            pose_PBPF.pose.orientation.x = estimated_object_ori[0]
            pose_PBPF.pose.orientation.y = estimated_object_ori[1]
            pose_PBPF.pose.orientation.z = estimated_object_ori[2]
            pose_PBPF.pose.orientation.w = estimated_object_ori[3]
            pub.publish(pose_PBPF)
        if publish_DOPE_pose_flag == True:
            pub_DOPE = rospy.Publisher('DOPE_pose', PoseStamped, queue_size = 1)
            pose_DOPE = PoseStamped()
            pose_DOPE.pose.position.x = nois_obj_pos_cur[0]
            pose_DOPE.pose.position.y = nois_obj_pos_cur[1]
            pose_DOPE.pose.position.z = nois_obj_pos_cur[2]
            pose_DOPE.pose.orientation.x = nois_obj_ori_cur[0]
            pose_DOPE.pose.orientation.y = nois_obj_ori_cur[1]
            pose_DOPE.pose.orientation.z = nois_obj_ori_cur[2]
            pose_DOPE.pose.orientation.w = nois_obj_ori_cur[3]
            pub_DOPE.publish(pose_DOPE)
        # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
        if publish_Opti_pose_flag == True and optitrack_working_flag == True:
            pub_opti = rospy.Publisher('Opti_pose', PoseStamped, queue_size = 1)
            pose_opti = PoseStamped()
            pose_opti.pose.position.x = opti_obj_pos_cur[0]
            pose_opti.pose.position.y = opti_obj_pos_cur[1]
            pose_opti.pose.position.z = opti_obj_pos_cur[2]
            pose_opti.pose.orientation.x = opti_obj_ori_cur[0]
            pose_opti.pose.orientation.y = opti_obj_ori_cur[1]
            pose_opti.pose.orientation.z = opti_obj_ori_cur[2]
            pose_opti.pose.orientation.w = opti_obj_ori_cur[3]
            pub_opti.publish(pose_opti)
        if compute_error_flag == True:
            t_before_record = time.time()
            boss_obse_err_pos_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
            boss_obse_err_ang_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
            boss_err_pos_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
            boss_err_ang_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
            flag_record = flag_record + 1
            flag_record_dope = flag_record_dope + 1
            boss_PBPF_err_pos_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_PBPF_pos, 'PBPF']
            boss_PBPF_err_ang_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_PBPF_ang, 'PBPF']
            boss_err_pos_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_PBPF_pos, 'PBPF']
            boss_err_ang_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_PBPF_ang, 'PBPF']
            flag_record = flag_record + 1
            flag_record_PBPF = flag_record_PBPF + 1
            estPB_from_pre_time = time.time()
            boss_estPB_pos_x_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[0], 'estPB']
            boss_estPB_pos_y_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[1], 'estPB']
            boss_estPB_pos_z_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[2], 'estPB']
            boss_estPB_ori_x_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[0], 'estPB']
            boss_estPB_ori_y_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[1], 'estPB']
            boss_estPB_ori_z_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[2], 'estPB']
            boss_estPB_ori_w_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[3], 'estPB']
            estPB_form_previous = estPB_form_previous + 1
            boss_estDO_pos_x_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_pos_cur[0], 'estDO']
            boss_estDO_pos_y_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_pos_cur[1], 'estDO']
            boss_estDO_pos_z_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_pos_cur[2], 'estDO']
            boss_estDO_ori_x_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_ori_cur[0], 'estDO']
            boss_estDO_ori_y_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_ori_cur[1], 'estDO']
            boss_estDO_ori_z_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_ori_cur[2], 'estDO']
            boss_estDO_ori_w_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, nois_obj_ori_cur[3], 'estDO']
            estDO_form_previous = estDO_form_previous + 1
        return
    
    # judge if any particles are contact
    def isAnyParticleInContact(self):
        for index, particle in enumerate(self.particle_cloud):
            # get pose from particle
            sim_par_cur_pos,sim_par_cur_ori = self.get_item_pos(self.pybullet_env_id_collection[index], initial_parameter.particle_no_visual_id_collection[index])
            # reset pose of object in pybullet vis to the pose
            p_visualisation.resetBasePositionAndOrientation(contact_particle_id,
                                                            sim_par_cur_pos,
                                                            sim_par_cur_ori)
            # check contact 
            pmin,pmax = p_visualisation.getAABB(contact_particle_id)
            collide_ids = p_visualisation.getOverlappingObjects(pmin,pmax)
            length = len(collide_ids)
            for t_i in range(length):
                # print("body id: ",collide_ids[t_i][1])
                if collide_ids[t_i][1] == 8 or collide_ids[t_i][1] == 9 or collide_ids[t_i][1] == 10 or collide_ids[t_i][1] == 11:
                    return True
            # print("check collision")
            # p_visualisation.stepSimulation()
            # contacts = p_visualisation.getContactPoints(bodyA=real_robot_id, bodyB=contact_particle_id)
            # for contact in contacts:
            #     contact_dis = contact[8]
            #     if contact_dis < 0.001:
            #         return True
        return False
    
    # update particle cloud particle angle
    def update_partcile_cloud_pose_PB(self, index, x, y, z, ori, linearVelocity, angularVelocity):
        self.particle_cloud[index].pos = [x, y, z]
        self.particle_cloud[index].ori = copy.deepcopy(ori)
        self.particle_cloud[index].linearVelocity = linearVelocity
        self.particle_cloud[index].angularVelocity = angularVelocity
        
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
            
    def function_to_parallelise(self, index, pybullet_env,fake_robot_id, real_robot_joint_pos):
        # ensure that each update of particles in the simulation inherits the velocity of the previous update
        pybullet_env.resetBaseVelocity(initial_parameter.particle_no_visual_id_collection[index],
                                       self.particle_cloud[index].linearVelocity,
                                       self.particle_cloud[index].angularVelocity)
        # change particle parameters
        self.change_obj_parameters(pybullet_env, initial_parameter.particle_no_visual_id_collection[index])
        # execute the control
        if update_style_flag == "pose":
            self.pose_sim_robot_move(index, pybullet_env, fake_robot_id, real_robot_joint_pos)
        elif update_style_flag == "time":
            # change simulation time
            pf_update_interval_in_sim = boss_pf_update_interval_in_real / change_sim_time
            # make sure all particles are updated
            for time_index in range(int(pf_update_interval_in_sim)):
                self.set_real_robot_JointPosition(pybullet_env,fake_robot_id[index],real_robot_joint_pos)
                pybullet_env.stepSimulation()
        ### ori: x,y,z,w
        # get velocity of each particle
        linearVelocity, angularVelocity = pybullet_env.getBaseVelocity(initial_parameter.particle_no_visual_id_collection[index])
        sim_par_cur_pos, sim_par_cur_ori = self.get_item_pos(pybullet_env,initial_parameter.particle_no_visual_id_collection[index])
        # add noise on pose of each particle
        normal_x, normal_y, normal_z, P_quat = self.add_noise_pose(sim_par_cur_pos, sim_par_cur_ori)
        pybullet_env.resetBasePositionAndOrientation(initial_parameter.particle_no_visual_id_collection[index],
                                                     [normal_x, normal_y, normal_z],
                                                     P_quat)
        # check collision
        while True:
            pybullet_env.stepSimulation()
            flag = 0
            contacts = pybullet_env.getContactPoints(bodyA=fake_robot_id[index], bodyB=initial_parameter.particle_no_visual_id_collection[index])
            # pmin,pmax = pybullet_simulation_env.getAABB(particle_no_visual_id)
            # collide_ids = pybullet_simulation_env.getOverlappingObjects(pmin,pmax)
            # length = len(collide_ids)
            for contact in contacts:
                contact_dis = contact[8]
                if contact_dis < -0.001:
                    #print("detected contact during initialization. BodyA: %d, BodyB: %d, LinkOfA: %d, LinkOfB: %d", contact[1], contact[2], contact[3], contact[4])
                    normal_x, normal_y, normal_z, P_quat = self.add_noise_pose(sim_par_cur_pos, sim_par_cur_ori)
                    pybullet_env.resetBasePositionAndOrientation(initial_parameter.particle_no_visual_id_collection[index],
                                                                 [normal_x, normal_y, normal_z],
                                                                 P_quat)
                    flag = 1
                    break
            if flag == 0:
                break
        self.update_partcile_cloud_pose_PB(index, normal_x, normal_y, normal_z, P_quat, linearVelocity, angularVelocity)
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
    def observation_update_PB(self, nois_obj_pos_cur, nois_obj_ori_cur):
        nois_obj_x = nois_obj_pos_cur[0]
        nois_obj_y = nois_obj_pos_cur[1]
        nois_obj_z = nois_obj_pos_cur[2]
        nois_obj_x_ori = nois_obj_ori_cur[0]
        nois_obj_y_ori = nois_obj_ori_cur[1]
        nois_obj_z_ori = nois_obj_ori_cur[2]
        nois_obj_w_ori = nois_obj_ori_cur[3]
        nois_obj_ori = [nois_obj_x_ori, nois_obj_y_ori, nois_obj_z_ori, nois_obj_w_ori] # pybullet x,y,z,w
        # make sure theta between -pi and pi
        nois_obj_ori = quaternion_correction(nois_obj_ori_cur)
#        nois_obj_quat = Quaternion(x=nois_obj_ori[0],y=nois_obj_ori[1],z=nois_obj_ori[2],w=nois_obj_ori[3]) # w,x,y,z
#        cos_theta_over_2 = nois_obj_quat.w
#        sin_theta_over_2 = math.sqrt(nois_obj_quat.x ** 2 + nois_obj_quat.y ** 2 + nois_obj_quat.z ** 2)
#        theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
#        theta = theta_over_2 * 2
#        if theta >= math.pi or theta <= -math.pi:
#            nois_obj_ori = [-nois_obj_x_ori, -nois_obj_y_ori, -nois_obj_z_ori, -nois_obj_w_ori]
            
        for index,particle in enumerate(self.particle_cloud): # particle angle
            particle_x = particle.pos[0]
            particle_y = particle.pos[1]
            particle_z = particle.pos[2]
            mean = 0
            # position weight
            dis_x = abs(particle_x - nois_obj_x)
            dis_y = abs(particle_y - nois_obj_y)
            dis_z = abs(particle_z - nois_obj_z)
            dis_xyz = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
            weight_xyz = self.normal_distribution(dis_xyz, mean, boss_sigma_obs_pos)
            # rotation weight
            par_ori = quaternion_correction(particle.ori)
            nois_obj_quat = Quaternion(x=nois_obj_ori[0], y=nois_obj_ori[1], z=nois_obj_ori[2], w=nois_obj_ori[3]) # Quaternion(): w,x,y,z
            par_quat = Quaternion(x=par_ori[0],y=par_ori[1],z=par_ori[2],w=par_ori[3])
            err_bt_par_dope = par_quat * nois_obj_quat.inverse
            cos_theta_over_2 = err_bt_par_dope.w
            sin_theta_over_2 = math.sqrt(err_bt_par_dope.x ** 2 + err_bt_par_dope.y ** 2 + err_bt_par_dope.z ** 2)
            theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
            theta = theta_over_2 * 2
            weight_ang = self.normal_distribution(theta, mean, boss_sigma_obs_ang)
            weight = weight_xyz * weight_ang
            particle.w = weight
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
    def change_obj_parameters(self,pybullet_env,par_id):
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
        sigma = boss_sigma_obs_x
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
        particles_w = []
        newParticles = []
        base_w = 0
        base_w_list = []
        base_w_list.append(base_w)
        particle_array_list = []
        n_particle = len(self.particle_cloud)
        for particle in self.particle_cloud:
            particles_w.append(particle.w)
            base_w = base_w + particle.w
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
            particle = Particle(self.particle_cloud[i].pos,
                                self.particle_cloud[i].ori,
                                1.0/particle_num, 
                                index,
                                self.particle_cloud[i].linearVelocity,
                                self.particle_cloud[i].angularVelocity)
            newParticles.append(particle)
        self.particle_cloud = copy.deepcopy(newParticles)

    def compute_position(self, position, base_w_list):
        for index in range(1, len(base_w_list)):
            if position <= base_w_list[index] and position > base_w_list[index - 1]:
                return index - 1
            else:
                continue
               
    def set_paticle_in_each_sim_env(self): # particle angle
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection):
            pybullet_env.resetBasePositionAndOrientation(self.particle_no_visual_id_collection[index],
                                                         self.particle_cloud[index].pos,
                                                         self.particle_cloud[index].ori)
        return

    def display_particle_in_visual_model_PB(self, particle_cloud): # particle angle
        for index, particle in enumerate(particle_cloud):
            p_visualisation.resetBasePositionAndOrientation(self.particle_with_visual_id_collection[index],
                                                            particle.pos,
                                                            particle.ori)

    def display_estimated_object_in_visual_model(self, estimated_object_id, esti_obj_pos, esti_obj_ori):
        p_visualisation.resetBasePositionAndOrientation(estimated_object_id,
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

    def compute_estimate_pos_of_object(self, particle_cloud): # need to change
        x_set = 0
        y_set = 0
        z_set = 0
        w_set = 0
        quaternions = []
        qws = []
        for index, particle in enumerate(particle_cloud):
            x_set = x_set + particle.pos[0] * particle.w
            y_set = y_set + particle.pos[1] * particle.w
            z_set = z_set + particle.pos[2] * particle.w
            q = quaternion_correction(particle.ori)
            qws.append(particle.w)
            quaternions.append([q[0], q[1], q[2], q[3]])
            w_set = w_set + particle.w
        q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
        return x_set/w_set, y_set/w_set, z_set/w_set, q[0], q[1], q[2], q[3]

    def compute_transformation_matrix(self, opti_T_rob_opti_pos, opti_T_rob_opti_ori, opti_T_obj_opti_pos, opti_T_obj_opti_ori):
        robot_transformation_matrix = transformations.quaternion_matrix(opti_T_rob_opti_ori)
        ow_T_robot = self.rotation_4_4_to_transformation_4_4(robot_transformation_matrix,opti_T_rob_opti_pos)
        object_transformation_matrix = transformations.quaternion_matrix(opti_T_obj_opti_ori)
        ow_T_object = self.rotation_4_4_to_transformation_4_4(object_transformation_matrix,opti_T_obj_opti_pos)
        robot_T_ow = np.linalg.inv(ow_T_robot)
        robot_T_object = np.dot(robot_T_ow,ow_T_object)
        return robot_T_object

    def rotation_4_4_to_transformation_4_4(self, rotation_4_4,pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4


#Class of Constant-velocity Particle Filtering
class PFMoveCV():
    def __init__(self,robot_id=None,real_robot_id=None,object_id=None):
        # init internals   
        self.particle_cloud_CV = copy.deepcopy(initial_parameter.particle_cloud_CV)
        self.particle_no_visual_id_collection_CV = copy.deepcopy(initial_parameter.particle_no_visual_id_collection_CV)
        self.pybullet_env_id_collection_CV = copy.deepcopy(initial_parameter.pybullet_particle_env_collection_CV)
        self.particle_with_visual_id_collection_CV = copy.deepcopy(initial_parameter.particle_with_visual_id_collection_CV)
        self.object_estimate_pose_x = []
        self.object_estimate_pose_y = []
        self.object_real_____pose_x = []
        self.object_real_____pose_y = []

    #new structure
    def real_robot_control_CV(self, opti_obj_pos_cur, opti_obj_ori_cur, nois_obj_pos_cur, nois_obj_ori_cur, do_obs_update):
        self.update_particle_filter_CV(opti_obj_pos_cur,
                                       opti_obj_ori_cur,
                                       nois_obj_pos_cur,
                                       nois_obj_ori_cur,
                                       do_obs_update)

    def compute_pos_err_bt_2_points(self,pos1,pos2):
        x_d = pos1[0]-pos2[0]
        y_d = pos1[1]-pos2[1]
        z_d = pos1[2]-pos2[2]
        distance = math.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)
        return distance

    # executed_control
    def update_particle_filter_CV(self, opti_obj_pos_cur, opti_obj_ori_cur, nois_obj_pos_cur, nois_obj_ori_cur, do_obs_update):
        global flag_record_CVPF
        global flag_record
        # motion model
        self.motion_update_CV(nois_obj_ori_cur)
        # observation model
        if do_obs_update:
            self.observation_update_CV(nois_obj_pos_cur, nois_obj_ori_cur)
        # Compute mean of particles
        estimated_object_pose_CV = self.compute_estimate_pos_of_object(self.particle_cloud_CV)
        estimated_object_pos_CV = [estimated_object_pose_CV[0], estimated_object_pose_CV[1], estimated_object_pose_CV[2]] # particle angle
        estimated_object_ori_CV = [estimated_object_pose_CV[3], estimated_object_pose_CV[4], estimated_object_pose_CV[5], estimated_object_pose_CV[6]]
        boss_est_pose_CVPF.append(estimated_object_pose_CV)
        # display estimated object
        if visualisation_flag == True and visualisation_mean == True:
            self.display_estimated_object_in_visual_model(estimated_object_id_CV, estimated_object_pos_CV, estimated_object_ori_CV)
        # display particles
        if visualisation_particle_flag == True:
            self.display_particle_in_visual_model_CV(self.particle_cloud_CV)
        # compute error and write down to the file
        err_opti_CVPF_pos = compute_pos_err_bt_2_points(estimated_object_pos_CV, opti_obj_pos_cur)
        err_opti_CVPF_ang = compute_ang_err_bt_2_points(estimated_object_ori_CV, opti_obj_ori_cur)
        err_opti_CVPF_ang = angle_correction(err_opti_CVPF_ang)
        if compute_error_flag == True:
            t_before_record = time.time()
            boss_CVPF_err_pos_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_CVPF_pos, 'CVPF']
            boss_CVPF_err_ang_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_CVPF_ang, 'CVPF']
            boss_err_pos_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_CVPF_pos, 'CVPF']
            boss_err_ang_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_CVPF_ang, 'CVPF']
            flag_record = flag_record + 1
            flag_record_CVPF = flag_record_CVPF + 1
        return

    def isAnyParticleInContact(self):
        for index, particle in enumerate(self.particle_cloud_CV):
            # get pose from particle
            sim_par_cur_pos,sim_par_cur_ori = self.get_item_pos(self.pybullet_env_id_collection_CV[index], initial_parameter.particle_no_visual_id_collection_CV[index])
            # reset pose of object in pybullet vis to the pose
            p_visualisation.resetBasePositionAndOrientation(contact_particle_id,
                                                            sim_par_cur_pos,
                                                            sim_par_cur_ori)
            # check contact 
            pmin,pmax = p_visualisation.getAABB(contact_particle_id)
            collide_ids = p_visualisation.getOverlappingObjects(pmin,pmax)
            length = len(collide_ids)
            for t_i in range(length):
                # print("body id: ",collide_ids[t_i][1])
                if collide_ids[t_i][1] == 8 or collide_ids[t_i][1] == 9 or collide_ids[t_i][1] == 10 or collide_ids[t_i][1] == 11:
                    return True
            # p_visualisation.stepSimulation()
            # contacts = p_visualisation.getContactPoints(bodyA=real_robot_id, bodyB=contact_particle_id)
            # for contact in contacts:
            #     contact_dis = contact[8]
            #     if contact_dis > 0.001:
            #         return True
                
        return False

    def motion_update_CV(self, nois_obj_ori_cur):
        # t0, t1: use observation data (obs0, obs1) to update motion
        if flag_update_num_CV < 2:
            length = len(boss_obs_pose_CVPF)
            obs_curr_pose = copy.deepcopy(boss_obs_pose_CVPF[length-1])
            obs_last_pose = copy.deepcopy(boss_obs_pose_CVPF[length-2])
            obs_curr_pos = obs_curr_pose[0]
            obs_curr_ori = obs_curr_pose[1]
            obs_last_pos = obs_last_pose[0]
            obs_last_ori = obs_last_pose[1]
            obsO_T_obsN = self.compute_transformation_matrix(obs_last_pos, obs_last_ori, obs_curr_pos, obs_curr_ori)
            parO_T_parN = copy.deepcopy(obsO_T_obsN)
            self.update_particle_in_motion_model_CV(parO_T_parN, nois_obj_ori_cur)
        # after t1: use (est0, est1) to update motion
        else:
            length = len(boss_est_pose_CVPF)
            est_curr_pose = copy.deepcopy(boss_est_pose_CVPF[length-1])
            est_last_pose = copy.deepcopy(boss_est_pose_CVPF[length-2])
            est_curr_pos = [est_curr_pose[0], est_curr_pose[1], est_curr_pose[2]]
            est_curr_ori = [est_curr_pose[3], est_curr_pose[4], est_curr_pose[5], est_curr_pose[6]]
            est_last_pos = [est_last_pose[0],est_last_pose[1], est_last_pose[2]]
            est_last_ori = [est_last_pose[3], est_last_pose[4], est_last_pose[5], est_last_pose[6]]
            estO_T_estN = self.compute_transformation_matrix(est_last_pos, est_last_ori, est_curr_pos, est_curr_ori)
            parO_T_parN = copy.deepcopy(estO_T_estN)
            self.update_particle_in_motion_model_CV(parO_T_parN, nois_obj_ori_cur)
        return

    def observation_update_CV(self,nois_obj_pos_cur, nois_obj_ori_cur):
        nois_obj_x = nois_obj_pos_cur[0]
        nois_obj_y = nois_obj_pos_cur[1]
        nois_obj_z = nois_obj_pos_cur[2]
        nois_obj_x_ori = nois_obj_ori_cur[0]
        nois_obj_y_ori = nois_obj_ori_cur[1]
        nois_obj_z_ori = nois_obj_ori_cur[2]
        nois_obj_w_ori = nois_obj_ori_cur[3]
        nois_obj_ori = [nois_obj_x_ori, nois_obj_y_ori, nois_obj_z_ori, nois_obj_w_ori]
        
        # make sure theta between -pi and pi
        nois_obj_ori = quaternion_correction(nois_obj_ori_cur)
#        nois_obj_quat = Quaternion(x=nois_obj_ori[0],y=nois_obj_ori[1],z=nois_obj_ori[2],w=nois_obj_ori[3]) # w,x,y,z
#        cos_theta_over_2 = nois_obj_quat.w
#        sin_theta_over_2 = math.sqrt(nois_obj_quat.x ** 2 + nois_obj_quat.y ** 2 + nois_obj_quat.z ** 2)
#        theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
#        theta = theta_over_2 * 2
#        if theta >= math.pi or theta <= -math.pi:
#            nois_obj_ori = [-nois_obj_x_ori, -nois_obj_y_ori, -nois_obj_z_ori, -nois_obj_w_ori]
       
        for index,particle in enumerate(self.particle_cloud_CV):
            particle_x = particle.pos[0]
            particle_y = particle.pos[1]
            particle_z = particle.pos[2]
            mean = 0
            # position weight
            dis_x = abs(particle_x - nois_obj_x)
            dis_y = abs(particle_y - nois_obj_y)
            dis_z = abs(particle_z - nois_obj_z)
            dis_xyz = math.sqrt(dis_x ** 2 + dis_y ** 2 + dis_z ** 2)
            weight_xyz = self.normal_distribution(dis_xyz, mean, boss_sigma_obs_pos)
            # rotation weight
            # nois_obj_ang = [nois_obj_pose[3],nois_obj_pose[4],nois_obj_pose[5]]
            # nois_obj_ori = p_visualisation.getQuaternionFromEuler(nois_obj_ang) # pybullet x,y,z,w

            par_ori = quaternion_correction(particle.ori)
            nois_obj_quat = Quaternion(x=nois_obj_ori[0],y=nois_obj_ori[1],z=nois_obj_ori[2],w=nois_obj_ori[3]) # w,x,y,z
            par_quat = Quaternion(x=par_ori[0],y=par_ori[1],z=par_ori[2],w=par_ori[3])
            err_bt_par_dope = par_quat * nois_obj_quat.inverse
            cos_theta_over_2 = err_bt_par_dope.w
            sin_theta_over_2 = math.sqrt(err_bt_par_dope.x ** 2 + err_bt_par_dope.y ** 2 + err_bt_par_dope.z ** 2)
            theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
            theta = theta_over_2 * 2
            weight_ang = self.normal_distribution(theta, mean, boss_sigma_obs_ang)
            weight = weight_xyz * weight_ang
            particle.w = weight
        # old resample function
        # Flag = self.normalize_particles_CV()
        # self.resample_particles_CV()
        # new resample function
        self.resample_particles_CV_update()
        self.set_paticle_in_each_sim_env_CV()
        return

    def update_particle_in_motion_model_CV(self, parO_T_parN, nois_obj_ori_cur):
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection_CV):
            pw_T_parO_pos = copy.deepcopy(self.particle_cloud_CV[index].pos)
            pw_T_parO_ori = copy.deepcopy(self.particle_cloud_CV[index].ori)
            pw_T_parO_3_3 = transformations.quaternion_matrix(pw_T_parO_ori)
            pw_T_parO = self.rotation_4_4_to_transformation_4_4(pw_T_parO_3_3,pw_T_parO_pos)
            pw_T_parN = np.dot(pw_T_parO,parO_T_parN)
            pw_T_parN_pos = [pw_T_parN[0][3], pw_T_parN[1][3], pw_T_parN[2][3]]
            # pw_T_parN_ori = transformations.quaternion_from_matrix(pw_T_parN)
            # pw_T_parN_ang = pybullet_env.getEulerFromQuaternion(pw_T_parN_ori)

            # add noise on particle filter
            normal_x = self.add_noise_2_par(pw_T_parN_pos[0])
            normal_y = self.add_noise_2_par(pw_T_parN_pos[1])
            normal_z = self.add_noise_2_par(pw_T_parN_pos[2])

            quat = copy.deepcopy(nois_obj_ori_cur) # x,y,z,w
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

            self.particle_cloud_CV[index].pos = [normal_x, normal_y, normal_z]
            self.particle_cloud_CV[index].ori = copy.deepcopy(pb_quat)


    def get_item_pos(self,pybullet_env,item_id):
        item_info = pybullet_env.getBasePositionAndOrientation(item_id)
        return item_info[0],item_info[1]

    def add_noise_2_par(self,current_pos):
        mean = current_pos
        sigma = boss_sigma_obs_x
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
        particles_w = []
        newParticles = []
        base_w = 0
        base_w_list = []
        base_w_list.append(base_w)
        particle_array_list = []
        n_particle = len(self.particle_cloud_CV)
        for particle in self.particle_cloud_CV:
            particles_w.append(particle.w)
            base_w = base_w + particle.w
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
            particle = Particle(self.particle_cloud_CV[i].pos,
                                self.particle_cloud_CV[i].ori,
                                1.0/particle_num, index)
            newParticles.append(particle)
        self.particle_cloud_CV = copy.deepcopy(newParticles)
        
    def compute_position_CV(self, position, base_w_list):
        for index in range(1, len(base_w_list)):
            if position <= base_w_list[index] and position > base_w_list[index - 1]:
                return index - 1
            else:
                continue
            
    def set_paticle_in_each_sim_env_CV(self):
        for index, pybullet_env in enumerate(self.pybullet_env_id_collection_CV):
            visual_particle_pos = self.particle_cloud_CV[index].pos
            visual_particle_ori = self.particle_cloud_CV[index].ori
            pybullet_env.resetBasePositionAndOrientation(self.particle_no_visual_id_collection_CV[index],
                                                         visual_particle_pos,
                                                         visual_particle_ori)
        return

    def display_particle_in_visual_model_CV(self, particle_cloud): # particle angle
        for index, particle in enumerate(particle_cloud):
            p_visualisation.resetBasePositionAndOrientation(self.particle_with_visual_id_collection_CV[index],
                                                            particle.pos,
                                                            particle.ori)

    def display_estimated_object_in_visual_model(self, estimated_object_id_CV, esti_obj_pos, esti_obj_ori):
        p_visualisation.resetBasePositionAndOrientation(estimated_object_id_CV,
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
        x_set = 0
        y_set = 0
        z_set = 0
        w_set = 0
        quaternions = []
        qws = []
        for index, particle in enumerate(particle_cloud): # particle angle
            x_set = x_set + particle.pos[0] * particle.w
            y_set = y_set + particle.pos[1] * particle.w
            z_set = z_set + particle.pos[2] * particle.w
            q = quaternion_correction(particle.ori)
            qws.append(particle.w)
            quaternions.append([q[0], q[1], q[2], q[3]])
            w_set = w_set + particle.w
        q = weightedAverageQuaternions(np.array(quaternions), np.array(qws))
        return x_set/w_set, y_set/w_set, z_set/w_set, q[0], q[1], q[2], q[3]

    def compute_transformation_matrix(self, a_pos,a_ori,b_pos,b_ori):
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
def compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, opti_T_obj_opti_pos, opti_T_obj_opti_ori):
    robot_transformation_matrix = transformations.quaternion_matrix(opti_T_rob_opti_ori)
    ow_T_robot = rotation_4_4_to_transformation_4_4(robot_transformation_matrix, opti_T_rob_opti_pos)
    object_transformation_matrix = transformations.quaternion_matrix(opti_T_obj_opti_ori)
    ow_T_object = rotation_4_4_to_transformation_4_4(object_transformation_matrix, opti_T_obj_opti_pos)
    robot_T_ow = np.linalg.inv(ow_T_robot)
    robot_T_object = np.dot(robot_T_ow, ow_T_object)
    return robot_T_object
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
    sigma = boss_sigma_obs_x
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
    # write the error file
    # if rospy.is_shutdown():
    if update_style_flag == "pose":
        if task_flag == "1":
            file_name_obse_pos = 'pose_scene1_obse_err_pos.csv'
            file_name_PBPF_pos = 'pose_scene1_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'pose_scene1_CVPF_err_pos.csv'
            file_name_obse_ang = 'pose_scene1_obse_err_ang.csv'
            file_name_PBPF_ang = 'pose_scene1_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'pose_scene1_CVPF_err_ang.csv'
        elif task_flag == "2":
            file_name_obse_pos = 'pose_scene2_obse_err_pos.csv'
            file_name_PBPF_pos = 'pose_scene2_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'pose_scene2_CVPF_err_pos.csv'
            file_name_obse_ang = 'pose_scene2_obse_err_ang.csv'
            file_name_PBPF_ang = 'pose_scene2_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'pose_scene2_CVPF_err_ang.csv'
        elif task_flag == "3":
            file_name_obse_pos = 'pose_scene3_obse_err_pos.csv'
            file_name_PBPF_pos = 'pose_scene3_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'pose_scene3_CVPF_err_pos.csv'
            file_name_obse_ang = 'pose_scene3_obse_err_ang.csv'
            file_name_PBPF_ang = 'pose_scene3_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'pose_scene3_CVPF_err_ang.csv'
        elif task_flag == "4":
            file_name_obse_pos = 'pose_scene4_obse_err_pos.csv'
            file_name_PBPF_pos = 'pose_scene4_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'pose_scene4_CVPF_err_pos.csv'
            file_name_obse_ang = 'pose_scene4_obse_err_ang.csv'
            file_name_PBPF_ang = 'pose_scene4_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'pose_scene4_CVPF_err_ang.csv'
    elif update_style_flag == "time":
        if task_flag == "1":
            file_name_obse_pos = 'time_scene1_obse_err_pos.csv'
            file_name_PBPF_pos = 'time_scene1_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'time_scene1_CVPF_err_pos.csv'
            file_name_obse_ang = 'time_scene1_obse_err_ang.csv'
            file_name_PBPF_ang = 'time_scene1_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'time_scene1_CVPF_err_ang.csv'
        elif task_flag == "2":
            file_name_obse_pos = 'time_scene2_obse_err_pos.csv'
            file_name_PBPF_pos = 'time_scene2_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'time_scene2_CVPF_err_pos.csv'
            file_name_obse_ang = 'time_scene2_obse_err_ang.csv'
            file_name_PBPF_ang = 'time_scene2_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'time_scene2_CVPF_err_ang.csv'
        elif task_flag == "3":
            file_name_obse_pos = 'time_scene3_obse_err_pos.csv'
            file_name_PBPF_pos = 'time_scene3_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'time_scene3_CVPF_err_pos.csv'
            file_name_obse_ang = 'time_scene3_obse_err_ang.csv'
            file_name_PBPF_ang = 'time_scene3_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'time_scene3_CVPF_err_ang.csv'
        elif task_flag == "4":
            file_name_obse_pos = 'time_scene4_obse_err_pos.csv'
            file_name_PBPF_pos = 'time_scene4_PBPF_err_pos.csv'
            file_name_CVPF_pos = 'time_scene4_CVPF_err_pos.csv'
            file_name_obse_ang = 'time_scene4_obse_err_ang.csv'
            file_name_PBPF_ang = 'time_scene4_PBPF_err_ang.csv'
            file_name_CVPF_ang = 'time_scene4_CVPF_err_ang.csv'
    # boss_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_pos,index=0,header=0,mode='a')
    # boss_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_ang,index=0,header=0,mode='a')
    if run_PBPF_flag == True:
        boss_obse_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_obse_pos,index=0,header=0,mode='a')
        boss_obse_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_obse_ang,index=0,header=0,mode='a')
        print("write obser file")
        boss_PBPF_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_PBPF_pos,index=0,header=0,mode='a')
        boss_PBPF_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_PBPF_ang,index=0,header=0,mode='a')
        print("write PBPF file")
        print("PB: Update frequency is: " + str(flag_update_num_PB))
        print("max time:", max(PBPF_time_cosuming_list))
    if run_CVPF_flag == True:
        boss_CVPF_err_pos_df.to_csv('error_file/'+str(file_time)+file_name_CVPF_pos,index=0,header=0,mode='a')
        boss_CVPF_err_ang_df.to_csv('error_file/'+str(file_time)+file_name_CVPF_ang,index=0,header=0,mode='a')
        print("write CVPF file")
        print("CV: Update frequency is: " + str(flag_update_num_CV))
    print("file_time:", file_time)
    # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
    if write_opti_pose_flag == True:
        print("write_opti_pos")
        boss_opti_pos_x_df.to_csv('opti_pos_x.csv')
        boss_opti_pos_y_df.to_csv('opti_pos_y.csv')
        boss_opti_pos_z_df.to_csv('opti_pos_z.csv')
        boss_opti_ori_x_df.to_csv('opti_ori_x.csv')
        boss_opti_ori_y_df.to_csv('opti_ori_y.csv')
        boss_opti_ori_z_df.to_csv('opti_ori_z.csv')
        boss_opti_ori_w_df.to_csv('opti_ori_w.csv')
    if write_estPB_pose_flag == True:
        print("write_opti_pos")
        boss_estPB_pos_x_df.to_csv('estPB_pos_x.csv')
        boss_estPB_pos_y_df.to_csv('estPB_pos_y.csv')
        boss_estPB_pos_z_df.to_csv('estPB_pos_z.csv')
        boss_estPB_ori_x_df.to_csv('estPB_ori_x.csv')
        boss_estPB_ori_y_df.to_csv('estPB_ori_y.csv')
        boss_estPB_ori_z_df.to_csv('estPB_ori_z.csv')
        boss_estPB_ori_w_df.to_csv('estPB_ori_w.csv')
    if write_estDO_pose_flag == True:
        print("write_estDO_pos")
        boss_estPB_pos_x_df.to_csv('estDO_pos_x.csv')
        boss_estPB_pos_y_df.to_csv('estDO_pos_y.csv')
        boss_estPB_pos_z_df.to_csv('estDO_pos_z.csv')
        boss_estPB_ori_x_df.to_csv('estDO_ori_x.csv')
        boss_estPB_ori_y_df.to_csv('estDO_ori_y.csv')
        boss_estPB_ori_z_df.to_csv('estDO_ori_z.csv')
        boss_estPB_ori_w_df.to_csv('estDO_ori_w.csv')
    sys.exit()
    
if __name__ == '__main__':
    opti_from_pre_time_begin = time.time() # program run initial time
    rospy.init_node('PF_for_dope') # ros node
    signal.signal(signal.SIGINT, signal_handler) # interrupt judgment
    # the flag of publish pose
    publish_PBPF_pose_flag = False
    publish_DOPE_pose_flag = False
    publish_Opti_pose_flag = False
    # the flag of visualisation
    visualisation_all = True
    visualisation_flag = True # DOPE and OptiTrack
    visualisation_mean = False
    visualisation_particle_flag = True
    # the flag of object judgment
    observation_cheating_flag = False
    object_flag = "cracker" # cracker/soup
    # OptiTrack works fine flag
    optitrack_working_flag = True
    publish_opti_pose_for_inter_flag = False
    # write pose file
    write_opti_pose_flag = False
    write_estPB_pose_flag = False
    write_estDO_pose_flag = False
    write_estCV_pose_flag = False
    first_write_flag = 0
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
    object_num = 2
    if update_style_flag == "pose":
        particle_num = 150
    elif update_style_flag == "time":
        if run_PBPF_flag == True:
            particle_num = 70
        elif run_CVPF_flag == True:
            particle_num = 50
    print("This is "+update_style_flag+" update in scene"+task_flag)    
    # some parameters
    d_thresh = 0.005
    a_thresh = 0.01
    d_thresh_CV = 0.0002
    a_thresh_CV = 0.0010
    flag_record = 0
    flag_record_dope = 0
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
    # error in xyz axis DOPE before recalibrating
    boss_sigma_obs_x = 0.03973017808163751 / 2.0
    boss_sigma_obs_y = 0.01167211468503462 / 2.0
    boss_sigma_obs_z = 0.02820930183351492 / 2.0
    # new dope error
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
    opti_form_previous = 0
    estPB_form_previous = 0
    estDO_form_previous = 0
    
    PBPF_time_cosuming_list = []
    
    # multi-objects/robots list
    pw_T_rob_sim_pose = [] 
    pw_T_robs_sim_pose = []
    pw_T_robs_sim_4_4 = []
    rob_T_obj_dope_pose = []
    rob_T_objs_dope_pose = []
    rob_T_objs_dope_4_4 = []
    
    # visualisation_model
    if visualisation_all == True:
        p_visualisation = bc.BulletClient(connection_mode=p.GUI_SERVER)#DIRECT,GUI_SERVER
    elif visualisation_all == False:
        p_visualisation = bc.BulletClient(connection_mode=p.DIRECT)#DIRECT,GUI_SERVER
    p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
    p_visualisation.setGravity(0, 0, -9.81)
    if task_flag == "4":
        p_visualisation.resetDebugVisualizerCamera(cameraDistance=0.5,cameraYaw=90,cameraPitch=-10,cameraTargetPosition=[0.5,0.1,0.2])
    else:
        p_visualisation.resetDebugVisualizerCamera(cameraDistance=1,cameraYaw=180,cameraPitch=-85,cameraTargetPosition=[0.3,0.1,0.2])
        
    
    plane_id = p_visualisation.loadURDF("plane.urdf")   
    # build an object of class "Ros_listener"
    ros_listener = Ros_listener(optitrack_working_flag, object_flag)
    # get object pose in robot world info from DOPE
    listener = tf.TransformListener()
    # robot pose in sim world (pybullet)
    pw_T_rob_sim_pos = [0.0, 0.0, 0.026]
    pw_T_rob_sim_ori = [0,0,0,1]
    pw_T_rob_sim_3_3 = transformations.quaternion_matrix(pw_T_rob_sim_ori)
    pw_T_rob_sim_4_4 = rotation_4_4_to_transformation_4_4(pw_T_rob_sim_3_3, pw_T_rob_sim_pos)
#    pw_T_rob_sim_pose.extend([pw_T_rob_sim_pos, pw_T_rob_sim_ori])
#    pw_T_robs_sim_pose.append(pw_T_rob_sim_pose)
#    pw_T_robs_sim_4_4.append(pw_T_rob_sim_4_4)
    
    if observation_cheating_flag == False:
        print("before while loop")
        while True:
            try:
                if object_flag == "cracker":
                    (trans,rot) = listener.lookupTransform('/panda_link0', '/cracker', rospy.Time(0))
                if object_flag == "soup":
                    (trans,rot) = listener.lookupTransform('/panda_link0', '/soup', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        rob_T_obj_dope_pos = list(trans)
        rob_T_obj_dope_ori = list(rot)
        rob_T_obj_dope_3_3 = transformations.quaternion_matrix(rob_T_obj_dope_ori)
        rob_T_obj_dope_4_4 = rotation_4_4_to_transformation_4_4(rob_T_obj_dope_3_3, rob_T_obj_dope_pos)
#        rob_T_obj_dope_pose.extend([rob_T_obj_dope_pos, rob_T_obj_dope_ori])
#        rob_T_objs_dope_pose.append(rob_T_obj_dope_pose)
#        rob_T_objs_dope_4_4.append(rob_T_obj_dope_4_4)
        print("after while loop")
        # give some time to update the data
        time.sleep(0.5)
        if optitrack_working_flag == True:
            opti_T_rob_opti_pos = ros_listener.robot_pos
            opti_T_rob_opti_ori = ros_listener.robot_ori
            opti_T_obj_opti_pos = ros_listener.object_pos
            opti_T_obj_opti_ori = ros_listener.object_ori
            if task_flag == "4":
                base_of_cheezit_pos = ros_listener.base_pos
                base_of_cheezit_ori = ros_listener.base_ori
        
            # compute transformation matrix (OptiTrack)
            rob_T_obj_opti_4_4 = compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, opti_T_obj_opti_pos, opti_T_obj_opti_ori)
            pw_T_obj_opti_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_obj_opti_4_4)
            pw_T_obj_opti_pos = [pw_T_obj_opti_4_4[0][3], pw_T_obj_opti_4_4[1][3], pw_T_obj_opti_4_4[2][3]]
            pw_T_obj_opti_ori = transformations.quaternion_from_matrix(pw_T_obj_opti_4_4)
            # load the groud truth object
            if visualisation_flag == True and object_flag == "cracker":
                # test_euler = p_visualisation.getEulerFromQuaternion(pw_T_obj_opti_ori)
                # print(pw_T_obj_opti_ori)
                # pw_T_obj_opti_ori = p_visualisation.getQuaternionFromEuler(test_euler) # pybullet x,y,z,w
                # print(pw_T_obj_opti_ori)
                optitrack_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_real_obj_with_visual_hor.urdf"),
                                                              pw_T_obj_opti_pos,
                                                              pw_T_obj_opti_ori)
            if visualisation_flag == True and object_flag == "soup":
                optitrack_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/camsoup_real_obj_with_visual_small_hor.urdf"),
                                                              pw_T_obj_opti_pos,
                                                              pw_T_obj_opti_ori)    
            # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
            if compute_error_flag == True:
                opti_from_pre_time = time.time()
                boss_opti_pos_x_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[0], 'opti']
                boss_opti_pos_y_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[1], 'opti']
                boss_opti_pos_z_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[2], 'opti']
                boss_opti_ori_x_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[0], 'opti']
                boss_opti_ori_y_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[1], 'opti']
                boss_opti_ori_z_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[2], 'opti']
                boss_opti_ori_w_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[3], 'opti']
                opti_form_previous = opti_form_previous + 1
            # compute and load the pose of optitrack_base (only task 4)
            pw_T_base_pos = [0,0,0]
            pw_T_base_ori = [0,0,0,1]
            if task_flag == "4":
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
    
        # compute pose of DOPE object in sim world (pybullet)
        pw_T_obj_dope = np.dot(pw_T_rob_sim_4_4, rob_T_obj_dope_4_4)
        pw_T_obj_dope_pos = [pw_T_obj_dope[0][3], pw_T_obj_dope[1][3], pw_T_obj_dope[2][3]]
        pw_T_obj_dope_ori = transformations.quaternion_from_matrix(pw_T_obj_dope)
        # load the DOPE object
        if visualisation_flag == True and object_flag == "cracker":
            dope_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_dope_obj_with_visual_hor.urdf"),
                                                      pw_T_obj_dope_pos,
                                                      pw_T_obj_dope_ori)
        if visualisation_flag == True and object_flag == "soup":
            dope_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/soup_dope_obj_with_visual_hor.urdf"),
                                                      pw_T_obj_dope_pos,
                                                      pw_T_obj_dope_ori)
    elif observation_cheating_flag == True:
        # load the groud truth object
        pw_T_objs_opti_pose_list = []
        pw_T_obj1_opti_pose = []
        pw_T_obj2_opti_pose = []
        pw_T_objs_dope_pose_list = []
        # for obj_num_index in range(object_num):
        pw_T_obj1_opti_pos = [0.4472889147344443, 0.08677179678403951, 0.0821006075425945]
        pw_T_obj1_opti_ori = [0.52338279, 0.47884367, 0.52129429, -0.47437481]
        pw_T_obj1_opti_pose.append(pw_T_obj1_opti_pos)
        pw_T_obj1_opti_pose.append(pw_T_obj1_opti_ori)
        pw_T_objs_opti_pose_list.append(pw_T_obj1_opti_pose)
        pw_T_obj2_opti_pos = [0.4472889147344443, 0.15677179678403951, 0.05]
        pw_T_obj2_opti_ori = [1.0, 0.0, 0.0, 1.0]
        pw_T_obj2_opti_pose.append(pw_T_obj2_opti_pos)
        pw_T_obj2_opti_pose.append(pw_T_obj2_opti_ori)
        pw_T_objs_opti_pose_list.append(pw_T_obj2_opti_pose)
        for obj_num_index in range(object_num):
            pw_T_objs_dope_pose = []
            pw_T_objs_dope_pos, pw_T_objs_dope_ori = add_noise_pose(pw_T_objs_opti_pose_list[obj_num_index][0], pw_T_objs_opti_pose_list[obj_num_index][1])
            pw_T_objs_dope_pose.append(pw_T_objs_dope_pos)
            pw_T_objs_dope_pose.append(pw_T_objs_dope_ori)
            pw_T_objs_dope_pose_list.append(pw_T_objs_dope_pose)
        # load the Opti/DOPE object
        if visualisation_flag == True:
            optitrack_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_real_obj_hor.urdf"),
                                                           pw_T_objs_opti_pose_list[0][0],
                                                           pw_T_objs_opti_pose_list[0][1])
            optitrack_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/soup_real_obj_hor.urdf"),
                                                           pw_T_objs_opti_pose_list[1][0],
                                                           pw_T_objs_opti_pose_list[1][1])
            dope_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_dope_obj_with_visual_hor.urdf"),
                                                      pw_T_objs_dope_pose_list[0][0],
                                                      pw_T_objs_dope_pose_list[0][1])
            dope_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/soup_dope_obj_with_visual_hor.urdf"),
                                                      pw_T_objs_dope_pose_list[1][0],
                                                      pw_T_objs_dope_pose_list[1][1])
        pw_T_base_pos = [0,0,0]
        pw_T_base_ori = [0,0,0,1]
    # initialization pose of DOPE
    pw_T_obj_dope_pose = [[pw_T_obj_dope_pos[0], pw_T_obj_dope_pos[1], pw_T_obj_dope_pos[2]],
                          [pw_T_obj_dope_ori[0], pw_T_obj_dope_ori[1], pw_T_obj_dope_ori[2], pw_T_obj_dope_ori[3]]]

    boss_obs_pose_CVPF.append(pw_T_obj_dope_pose)
    # compute error
    if optitrack_working_flag == True:
        err_opti_dope_pos = compute_pos_err_bt_2_points(pw_T_obj_opti_pos, pw_T_obj_dope_pos)
        err_opti_dope_ang = compute_ang_err_bt_2_points(pw_T_obj_opti_ori, pw_T_obj_dope_ori)
        err_opti_dope_ang = angle_correction(err_opti_dope_ang)
    elif optitrack_working_flag == False:
        err_opti_dope_pos = compute_pos_err_bt_2_points(ros_listener.fake_opti_pos, pw_T_obj_dope_pos)
        err_opti_dope_ang = compute_ang_err_bt_2_points(ros_listener.fake_opti_ori, pw_T_obj_dope_ori)
        err_opti_dope_ang = angle_correction(err_opti_dope_ang)
    # initial visualisation world model
    # build an object of class "InitialRealworldModel"
    if observation_cheating_flag == False:
        real_world_object = InitialRealworldModel(ros_listener.current_joint_values, object_flag, p_visualisation)
    elif observation_cheating_flag == True:
        real_world_object = InitialRealworldModel(0, object_flag, p_visualisation)
    #initialize the real robot in the pybullet
    real_robot_id = real_world_object.initial_robot(robot_pos=pw_T_rob_sim_pos, robot_orientation=pw_T_rob_sim_ori)
    # initialize the real object in the pybullet
    # real_object_id = real_world_object.initial_target_object(object_pos = pw_T_obj_opti_pos, object_orientation = pw_T_obj_opti_ori)
    if optitrack_working_flag == True:
        contact_particle_id = real_world_object.initial_contact_object(object_pos=pw_T_obj_opti_pos, object_orientation=pw_T_obj_opti_ori)
    elif optitrack_working_flag == False:
        contact_particle_id = real_world_object.initial_contact_object(object_pos=pw_T_obj_dope_pos, object_orientation=pw_T_obj_dope_ori)
    # build an object of class "Franka_robot"
    franka_robot = Franka_robot(real_robot_id, p_visualisation)
    # initialize sim world (particles)
    # initial_parameter = InitialSimulationModel(particle_num, pw_T_rob_sim_pos, pw_T_rob_sim_ori, dope_obj_pos_init, dope_obj_ori_init)
    initial_parameter = InitialSimulationModel(object_num, particle_num, pw_T_rob_sim_pos, pw_T_rob_sim_ori, pw_T_obj_dope_pos, pw_T_obj_dope_ori,
                                               pw_T_base_pos, pw_T_base_ori,
                                               boss_sigma_obs_x, boss_sigma_obs_y, boss_sigma_obs_z, boss_sigma_obs_ang_init, p_visualisation,
                                               update_style_flag, change_sim_time, task_flag, object_flag)
    initial_parameter.initial_particle()
    # get estimated object
    estimated_object_set = initial_parameter.initial_and_set_simulation_env(ros_listener.current_joint_values)
    estimated_object_pos = [estimated_object_set[0], estimated_object_set[1], estimated_object_set[2]]
    estimated_object_ori = [estimated_object_set[3], estimated_object_set[4], estimated_object_set[5], estimated_object_set[6]]
    # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
    if compute_error_flag == True:
        estPB_from_pre_time = time.time()
        boss_estPB_pos_x_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[0], 'estPB']
        boss_estPB_pos_y_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[1], 'estPB']
        boss_estPB_pos_z_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_pos[2], 'estPB']
        boss_estPB_ori_x_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[0], 'estPB']
        boss_estPB_ori_y_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[1], 'estPB']
        boss_estPB_ori_z_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[2], 'estPB']
        boss_estPB_ori_w_df.loc[estPB_form_previous] = [estPB_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, estimated_object_ori[3], 'estPB']
        estPB_form_previous = estPB_form_previous + 1
        boss_estDO_pos_x_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_dope_pos[0], 'estDO']
        boss_estDO_pos_y_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_dope_pos[1], 'estDO']
        boss_estDO_pos_z_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_dope_pos[2], 'estDO']
        boss_estDO_ori_x_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_dope_ori[0], 'estDO']
        boss_estDO_ori_y_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_dope_ori[1], 'estDO']
        boss_estDO_ori_z_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_dope_ori[2], 'estDO']
        boss_estDO_ori_w_df.loc[estDO_form_previous] = [estDO_form_previous, estPB_from_pre_time - opti_from_pre_time_begin, pw_T_obj_dope_ori[3], 'estDO']
        estDO_form_previous = estDO_form_previous + 1
    # publish pose
    if publish_PBPF_pose_flag == True:
        pub = rospy.Publisher('PBPF_pose', PoseStamped, queue_size = 1)
        pose_PBPF = PoseStamped()
        pose_PBPF.pose.position.x = estimated_object_pos[0]
        pose_PBPF.pose.position.y = estimated_object_pos[1]
        pose_PBPF.pose.position.z = estimated_object_pos[2]
        pose_PBPF.pose.orientation.x = estimated_object_ori[0]
        pose_PBPF.pose.orientation.y = estimated_object_ori[1]
        pose_PBPF.pose.orientation.z = estimated_object_ori[2]
        pose_PBPF.pose.orientation.w = estimated_object_ori[3]
        pub.publish(pose_PBPF)
        # rospy.loginfo(pose_PBPF)
    if publish_DOPE_pose_flag == True:
        pub_DOPE = rospy.Publisher('DOPE_pose', PoseStamped, queue_size = 1)
        pose_DOPE = PoseStamped()
        pose_DOPE.pose.position.x = pw_T_obj_dope_pos[0]
        pose_DOPE.pose.position.y = pw_T_obj_dope_pos[1]
        pose_DOPE.pose.position.z = pw_T_obj_dope_pos[2]
        pose_DOPE.pose.orientation.x = pw_T_obj_dope_ori[0]
        pose_DOPE.pose.orientation.y = pw_T_obj_dope_ori[1]
        pose_DOPE.pose.orientation.z = pw_T_obj_dope_ori[2]
        pose_DOPE.pose.orientation.w = pw_T_obj_dope_ori[3]
        # print(pose_DOPE)
        pub_DOPE.publish(pose_DOPE)
        # rospy.loginfo(pose_DOPE)
    if optitrack_working_flag == True:
        # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
        if publish_Opti_pose_flag == True:
            pub_opti = rospy.Publisher('Opti_pose', PoseStamped, queue_size = 1)
            pose_opti = PoseStamped()
            pose_opti.pose.position.x = pw_T_obj_opti_pos[0]
            pose_opti.pose.position.y = pw_T_obj_opti_pos[1]
            pose_opti.pose.position.z = pw_T_obj_opti_pos[2]
            pose_opti.pose.orientation.x = pw_T_obj_opti_ori[0]
            pose_opti.pose.orientation.y = pw_T_obj_opti_ori[1]
            pose_opti.pose.orientation.z = pw_T_obj_opti_ori[2]
            pose_opti.pose.orientation.w = pw_T_obj_opti_ori[3]
            pub_opti.publish(pose_opti)
    
    boss_est_pose_CVPF.append(estimated_object_set)
    initial_parameter.initial_and_set_simulation_env_CV(ros_listener.current_joint_values)
    # display particles
    if visualisation_particle_flag == True:
        if run_PBPF_flag == True:
            initial_parameter.display_particle()
        if run_CVPF_flag == True:
            initial_parameter.display_particle_CV()
    # load object in the sim world
    if visualisation_flag == True and object_flag == "cracker" and visualisation_mean == True:
        if run_PBPF_flag == True:
            estimated_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_est_obj_with_visual_PB_hor.urdf"),
                                                           estimated_object_pos,
                                                           estimated_object_ori)
        if run_CVPF_flag == True:
            estimated_object_id_CV = p_visualisation.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_est_obj_with_visual_CV_hor.urdf"),
                                                              estimated_object_pos,
                                                              estimated_object_ori)
    if visualisation_flag == True and object_flag == "soup" and visualisation_mean == True:
        if run_PBPF_flag == True:
            estimated_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/camsoup_est_obj_with_visual_small_PB_hor.urdf"),
                                                           estimated_object_pos,
                                                           estimated_object_ori)
        if run_CVPF_flag == True:
            estimated_object_id_CV = p_visualisation.loadURDF(os.path.expanduser("~/project/object/soup/camsoup_est_obj_with_visual_small_CV_hor.urdf"),
                                                              estimated_object_pos,
                                                              estimated_object_ori)
    # compute error
    if optitrack_working_flag == True:
        err_opti_esti_pos = compute_pos_err_bt_2_points(estimated_object_pos, pw_T_obj_opti_pos)
        err_opti_esti_ang = compute_ang_err_bt_2_points(estimated_object_ori, pw_T_obj_opti_ori)
        err_opti_esti_ang = angle_correction(err_opti_esti_ang)
    elif optitrack_working_flag == False:
        err_opti_esti_pos = compute_pos_err_bt_2_points(estimated_object_pos,ros_listener.fake_opti_pos)
        err_opti_esti_ang = compute_ang_err_bt_2_points(estimated_object_ori,ros_listener.fake_opti_ori)
        err_opti_esti_ang = angle_correction(err_opti_esti_ang)
    # initial_parameter.particle_cloud #parameter of particle
    # initial_parameter.pybullet_particle_env_collection #env of simulation
    # initial_parameter.fake_robot_id_collection #id of robot in simulation
    # initial_parameter.particle_no_visual_id_collection #id of particle in simulation
    
    # run the simulation
    Flag = True
    # compute DOPE object old pose
    # dope_obj_pos_old = copy.deepcopy(pw_T_obj_dope_pos)
    # dope_obj_ori_old = copy.deepcopy(pw_T_obj_dope_ori)
    # dope_obj_pos_old_CV = copy.deepcopy(pw_T_obj_dope_pos)
    # dope_obj_ori_old_CV = copy.deepcopy(pw_T_obj_dope_ori)
    # compute pose of robot arm
    rob_link_9_pose_old_PB = p_visualisation.getLinkState(real_robot_id,9)
    rob_link_9_pose_old_CV = p_visualisation.getLinkState(real_robot_id,9)
    rob_link_9_ang_old_PB = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_old_PB[1])
    rob_link_9_ang_old_CV = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_old_CV[1])
    rob_pose_init = copy.deepcopy(rob_link_9_pose_old_PB)
    print("Welcome to Our Approach !")
    robot1 = PFMove()
    robot2 = PFMoveCV()
    
    while not rospy.is_shutdown():
        #panda robot moves in the visualization window
        franka_robot.fanka_robot_move(ros_listener.current_joint_values)
        #get pose info from DOPE
        dope_is_fresh = True
        if observation_cheating_flag == False:
            try:
                if object_flag == "cracker":
                    latest_dope_time = listener.getLatestCommonTime('/panda_link0', '/cracker')
                if object_flag == "soup":
                    latest_dope_time = listener.getLatestCommonTime('/panda_link0', '/soup')
                #print("latest_dope_time: ",latest_dope_time.to_sec())
                #print("rospy.get_time: ",rospy.get_time())
                if (rospy.get_time() - latest_dope_time.to_sec()) < 0.1:
                    if object_flag == "cracker":
                        (trans,rot) = listener.lookupTransform('/panda_link0', '/cracker', rospy.Time(0))
                    if object_flag == "soup":
                        (trans,rot) = listener.lookupTransform('/panda_link0', '/soup', rospy.Time(0))
                    dope_is_fresh = True
                    # print("dope is FRESH")
                else:
                    # DOPE has not been updating for a while
                    dope_is_fresh = False
                    print("dope is NOT fresh")
                # break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("can not find tf")
            rob_T_obj_dope_pos = list(trans)
            rob_T_obj_dope_ori = list(rot)
            rob_T_obj_dope_3_3 = transformations.quaternion_matrix(rob_T_obj_dope_ori)
            rob_T_obj_dope_4_4 = rotation_4_4_to_transformation_4_4(rob_T_obj_dope_3_3,rob_T_obj_dope_pos)
            pw_T_obj_dope = np.dot(pw_T_rob_sim_4_4, rob_T_obj_dope_4_4)
            pw_T_obj_dope_pos = [pw_T_obj_dope[0][3],pw_T_obj_dope[1][3],pw_T_obj_dope[2][3]]
            pw_T_obj_dope_ori = transformations.quaternion_from_matrix(pw_T_obj_dope)
            pw_T_obj_dope_ang = p_visualisation.getEulerFromQuaternion(pw_T_obj_dope_ori)
            dope_obj_pos_cur = copy.deepcopy(pw_T_obj_dope_pos)
            dope_obj_ang_cur = copy.deepcopy(pw_T_obj_dope_ang)
            dope_obj_ori_cur = copy.deepcopy(pw_T_obj_dope_ori)
            dope_obj_pose_cur = [[dope_obj_pos_cur[0], dope_obj_pos_cur[1], dope_obj_pos_cur[2]],
                                 [dope_obj_ori_cur[0], dope_obj_ori_cur[1], dope_obj_ori_cur[2], dope_obj_ori_cur[3]]]
            
        # elif observation_cheating_flag == True
            
        
        # display DOPE object in visual model
        if visualisation_flag == True:
            display_real_object_in_visual_model(dope_object_id,dope_obj_pos_cur,dope_obj_ori_cur)
        # get ground truth pose of robot and object
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
            pw_T_obj_pos_opti = copy.deepcopy(pw_T_obj_opti_pos)
            pw_T_obj_ori_opti = copy.deepcopy(pw_T_obj_opti_ori)
            # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
            if compute_error_flag == True:
                opti_from_pre_time = time.time()
                boss_opti_pos_x_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[0], 'opti']
                boss_opti_pos_y_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[1], 'opti']
                boss_opti_pos_z_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_pos[2], 'opti']
                boss_opti_ori_x_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[0], 'opti']
                boss_opti_ori_y_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[1], 'opti']
                boss_opti_ori_z_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[2], 'opti']
                boss_opti_ori_w_df.loc[opti_form_previous] = [opti_form_previous, opti_from_pre_time - opti_from_pre_time_begin, pw_T_obj_opti_ori[3], 'opti']
                opti_form_previous = opti_form_previous + 1
        # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
        if publish_opti_pose_for_inter_flag == True:
            pub_opti = rospy.Publisher('Opti_pose', PoseStamped, queue_size = 1)
            pose_opti = PoseStamped()
            pose_opti.pose.position.x = pw_T_obj_opti_pos[0]
            pose_opti.pose.position.y = pw_T_obj_opti_pos[1]
            pose_opti.pose.position.z = pw_T_obj_opti_pos[2]
            pose_opti.pose.orientation.x = pw_T_obj_opti_ori[0]
            pose_opti.pose.orientation.y = pw_T_obj_opti_ori[1]
            pose_opti.pose.orientation.z = pw_T_obj_opti_ori[2]
            pose_opti.pose.orientation.w = pw_T_obj_opti_ori[3]
            pub_opti.publish(pose_opti)
        # compute distance between old DOPE obj and cur DOPE obj (position and angle)
        # dis_betw_cur_and_old = compute_pos_err_bt_2_points(dope_obj_pos_cur, dope_obj_pos_old)
        # ang_betw_cur_and_old = compute_ang_err_bt_2_points(dope_obj_ori_cur, dope_obj_ori_old)
        # dis_betw_cur_and_old_CV = compute_pos_err_bt_2_points(dope_obj_pos_cur, dope_obj_pos_old_CV)
        # ang_betw_cur_and_old_CV = compute_ang_err_bt_2_points(dope_obj_ori_cur, dope_obj_ori_old_CV)
        # compute distance between old robot arm and cur robot arm (position and angle)
        rob_link_9_pose_cur_PB = p_visualisation.getLinkState(real_robot_id, 9)
        rob_link_9_pose_cur_CV = p_visualisation.getLinkState(real_robot_id, 9)
        rob_link_9_ang_cur_PB = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_cur_PB[1])
        rob_link_9_ang_cur_CV = p_visualisation.getEulerFromQuaternion(rob_link_9_pose_cur_CV[1])
        dis_robcur_robold_PB = compute_pos_err_bt_2_points(rob_link_9_pose_cur_PB[0], rob_link_9_pose_old_PB[0])
        dis_robcur_robold_CV = compute_pos_err_bt_2_points(rob_link_9_pose_cur_CV[0], rob_link_9_pose_old_CV[0])
        # update according to the pose
        if update_style_flag == "pose":
            # PBPF algorithm
            if run_PBPF_flag == True:
                if (dis_robcur_robold_PB > d_thresh):
                    # judgement for any particles contact
                    if robot1.isAnyParticleInContact():
                        if first_write_flag == 0 and compute_error_flag == True:
                            # record the error
                            t_begin = time.time()
                            t_before_record = time.time()
                            boss_obse_err_pos_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
                            boss_obse_err_ang_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
                            boss_err_pos_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
                            boss_err_ang_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
                            flag_record = flag_record + 1
                            flag_record_dope = flag_record_dope + 1
                            boss_PBPF_err_pos_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_PBPF_err_ang_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            flag_record = flag_record + 1
                            flag_record_PBPF = flag_record_PBPF + 1
                            boss_CVPF_err_pos_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_CVPF_err_ang_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            flag_record = flag_record + 1
                            flag_record_CVPF = flag_record_CVPF + 1
                            first_write_flag = 1
                        simRobot_touch_par_flag = 1
                        t_begin_PBPF = time.time()
                        flag_update_num_PB = flag_update_num_PB + 1
                        # when OptiTrack does not work, record the previous OptiTrack pose in the rosbag
                        if optitrack_working_flag == True:
                            opti_obj_pos_cur = copy.deepcopy(pw_T_obj_opti_pos) # get pos of real object
                            opti_obj_ori_cur = copy.deepcopy(pw_T_obj_opti_ori)
                        elif optitrack_working_flag == False:
                            opti_obj_pos_cur = ros_listener.fake_opti_pos
                            opti_obj_ori_cur = ros_listener.fake_opti_ori
                        nois_obj_pos_cur = copy.deepcopy(dope_obj_pos_cur)
                        nois_obj_ori_cur = copy.deepcopy(dope_obj_ori_cur)
                        # execute PBPF algorithm movement
                        robot1.real_robot_control_PB(opti_obj_pos_cur, # ground truth pos [x, y, z]
                                                     opti_obj_ori_cur, # ground truth ori [x, y, z, w]
                                                     ros_listener.current_joint_values, # joints of robot arm
                                                     nois_obj_pos_cur, # DOPE value pos [x, y, z]
                                                     nois_obj_ori_cur, # DOPE value ori [x, y, z, w]
                                                     do_obs_update=dope_is_fresh) # flag for judging DOPE work        
                        rob_link_9_pose_old_PB = copy.deepcopy(rob_link_9_pose_cur_PB)
                        if visualisation_flag == True and optitrack_working_flag == True:
                            display_real_object_in_visual_model(optitrack_object_id, pw_T_obj_opti_pos, pw_T_obj_opti_ori)
                        elif visualisation_flag == True and optitrack_working_flag == False:
                            display_real_object_in_visual_model(optitrack_object_id,ros_listener.fake_opti_pos,ros_listener.fake_opti_ori)
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
                    if first_write_flag == 0 and compute_error_flag == True:
                        # record the error
                        t_begin = time.time()
                        t_before_record = time.time()
                        boss_obse_err_pos_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
                        boss_obse_err_ang_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
                        boss_err_pos_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
                        boss_err_ang_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
                        flag_record = flag_record + 1
                        flag_record_dope = flag_record_dope + 1
                        boss_PBPF_err_pos_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                        boss_PBPF_err_ang_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                        boss_err_pos_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                        boss_err_ang_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                        flag_record = flag_record + 1
                        flag_record_PBPF = flag_record_PBPF + 1
                        boss_CVPF_err_pos_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                        boss_CVPF_err_ang_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                        boss_err_pos_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                        boss_err_ang_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                        flag_record = flag_record + 1
                        flag_record_CVPF = flag_record_CVPF + 1
                        first_write_flag = 1
                    flag_update_num_CV = flag_update_num_CV + 1
                    boss_obs_pose_CVPF.append(dope_obj_pose_cur)
                    if optitrack_working_flag == True:
                        opti_obj_pos_cur_CV = copy.deepcopy(pw_T_obj_opti_pos) #get pos of real object
                        opti_obj_ori_cur_CV = copy.deepcopy(pw_T_obj_opti_ori)
                    elif optitrack_working_flag == False:
                        opti_obj_pos_cur_CV = ros_listener.fake_opti_pos
                        opti_obj_ori_cur_CV = ros_listener.fake_opti_ori
                    nois_obj_pos_cur_CV = copy.deepcopy(dope_obj_pos_cur)
                    nois_obj_ang_cur_CV = copy.deepcopy(dope_obj_ang_cur)
                    nois_obj_ori_cur_CV = copy.deepcopy(dope_obj_ori_cur)
                    # execute CVPF algorithm movement
                    robot2.real_robot_control_CV(opti_obj_pos_cur_CV, # ground truth pos [x, y, z]
                                                 opti_obj_ori_cur_CV, # ground truth ori [x, y, z, w]
                                                 nois_obj_pos_cur_CV, # DOPE value pos [x, y, z]
                                                 nois_obj_ori_cur_CV, # DOPE value ori [x, y, z, w]
                                                 do_obs_update=dope_is_fresh) # flag for judging DOPE work
                    rob_link_9_pose_old_CV = copy.deepcopy(rob_link_9_pose_cur_CV)
                    if visualisation_flag == True:
                        display_real_object_in_visual_model(optitrack_object_id, pw_T_obj_opti_pos, pw_T_obj_opti_ori)
        # update according to the time
        elif update_style_flag == "time":
            while True:
                # PBPF algorithm
                if run_PBPF_flag == True:
                    if robot1.isAnyParticleInContact():
                        if first_write_flag == 0 and compute_error_flag == True:
                            # record the error
                            t_begin = time.time()
                            t_before_record = time.time()
                            boss_obse_err_pos_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
                            boss_obse_err_ang_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
                            boss_err_pos_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
                            boss_err_ang_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
                            flag_record = flag_record + 1
                            flag_record_dope = flag_record_dope + 1
                            boss_PBPF_err_pos_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_PBPF_err_ang_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            flag_record = flag_record + 1
                            flag_record_PBPF = flag_record_PBPF + 1
                            boss_CVPF_err_pos_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_CVPF_err_ang_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            flag_record = flag_record + 1
                            flag_record_CVPF = flag_record_CVPF + 1
                            first_write_flag = 1
                        simRobot_touch_par_flag = 1
                        t_begin_PBPF = time.time()
                        flag_update_num_PB = flag_update_num_PB + 1
                        if optitrack_working_flag == True:
                            opti_obj_pos_cur = copy.deepcopy(pw_T_obj_opti_pos) # get pos of real object
                            opti_obj_ori_cur = copy.deepcopy(pw_T_obj_opti_ori)
                        elif optitrack_working_flag == False:
                            opti_obj_pos_cur = ros_listener.fake_opti_pos
                            opti_obj_ori_cur = ros_listener.fake_opti_ori
                        nois_obj_pos_cur = copy.deepcopy(dope_obj_pos_cur)
                        nois_obj_ori_cur = copy.deepcopy(dope_obj_ori_cur)
                        # execute PBPF algorithm movement
                        robot1.real_robot_control_PB(opti_obj_pos_cur, # ground truth pos [x, y, z]
                                                     opti_obj_ori_cur, # ground truth ori [x, y, z, w]
                                                     ros_listener.current_joint_values, # joints of robot arm
                                                     nois_obj_pos_cur, # DOPE value pos [x, y, z]
                                                     nois_obj_ori_cur, # DOPE value ori [x, y, z, w]
                                                     do_obs_update=dope_is_fresh) # flag for judging DOPE work
                        if visualisation_flag == True and optitrack_working_flag == True:
                            display_real_object_in_visual_model(optitrack_object_id, pw_T_obj_opti_pos, pw_T_obj_opti_ori)
                        elif visualisation_flag == True and optitrack_working_flag == False:
                            display_real_object_in_visual_model(optitrack_object_id, ros_listener.fake_opti_pos, ros_listener.fake_opti_ori)
                        t_finish_PBPF = time.time()
                        PBPF_time_cosuming_list.append(t_finish_PBPF - t_begin_PBPF)
                        simRobot_touch_par_flag = 0
                    else:
                        robot1.motion_update_PB_parallelised(initial_parameter.pybullet_particle_env_collection,
                                                             initial_parameter.fake_robot_id_collection,
                                                             ros_listener.current_joint_values)
                # CVPF algorithm
                if run_CVPF_flag == True:
                    if robot2.isAnyParticleInContact():
                        if first_write_flag == 0 and compute_error_flag == True:
                            # record the error
                            t_begin = time.time()
                            t_before_record = time.time()
                            boss_obse_err_pos_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
                            boss_obse_err_ang_df.loc[flag_record_dope] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
                            boss_err_pos_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_pos, 'dope']
                            boss_err_ang_df.loc[flag_record] = [flag_record_dope, t_before_record - t_begin, err_opti_dope_ang, 'dope']
                            flag_record = flag_record + 1
                            flag_record_dope = flag_record_dope + 1
                            boss_PBPF_err_pos_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_PBPF_err_ang_df.loc[flag_record_PBPF] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_pos, 'PBPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_PBPF, t_before_record - t_begin, err_opti_esti_ang, 'PBPF']
                            flag_record = flag_record + 1
                            flag_record_PBPF = flag_record_PBPF + 1
                            boss_CVPF_err_pos_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_CVPF_err_ang_df.loc[flag_record_CVPF] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            boss_err_pos_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_pos, 'CVPF']
                            boss_err_ang_df.loc[flag_record] = [flag_record_CVPF, t_before_record - t_begin, err_opti_esti_ang, 'CVPF']
                            flag_record = flag_record + 1
                            flag_record_CVPF = flag_record_CVPF + 1
                            first_write_flag = 1
                        flag_update_num_CV = flag_update_num_CV + 1
                        boss_obs_pose_CVPF.append(dope_obj_pose_cur)
                        if optitrack_working_flag == True:
                            opti_obj_pos_cur_CV = copy.deepcopy(pw_T_obj_opti_pos) #get pos of real object
                            opti_obj_ori_cur_CV = copy.deepcopy(pw_T_obj_opti_ori)
                        elif optitrack_working_flag == False:
                            opti_obj_pos_cur_CV = ros_listener.fake_opti_pos
                            opti_obj_ori_cur_CV = ros_listener.fake_opti_ori
                        nois_obj_pos_cur_CV = copy.deepcopy(dope_obj_pos_cur)
                        nois_obj_ang_cur_CV = copy.deepcopy(dope_obj_ang_cur)
                        nois_obj_ori_cur_CV = copy.deepcopy(dope_obj_ori_cur)
                        # execute CVPF algorithm movement
                        robot2.real_robot_control_CV(opti_obj_pos_cur_CV, # ground truth pos [x, y, z]
                                                     opti_obj_ori_cur_CV, # ground truth ori [x, y, z, w]
                                                     nois_obj_pos_cur_CV, # DOPE value pos [x, y, z]
                                                     nois_obj_ori_cur_CV, # DOPE value ori [x, y, z, w]
                                                     do_obs_update=dope_is_fresh) # flag for judging DOPE work
                        if visualisation_flag == True and optitrack_working_flag == True:
                            display_real_object_in_visual_model(optitrack_object_id, pw_T_obj_opti_pos, pw_T_obj_opti_ori)
                pf_update_rate.sleep()
                break    
        t_end_while = time.time()
        if Flag is False:
            break
    p_visualisation.disconnect()


