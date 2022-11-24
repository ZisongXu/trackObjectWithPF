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

#Class of initialize the real world model
class Visualisation_World():
    def __init__(self, object_num=0, rob_num=1, other_obj_num=0, particle_num=0):
        self.object_num = object_num
        self.rob_num = rob_num
        self.other_obj_num = other_obj_num
        self.particle_num = particle_num
        self.p_visualisation = 0
        self.create_scene = Create_Scene(object_num, rob_num, other_obj_num)
        self.ros_listener = Ros_Listener()
        self.listener = tf.TransformListener()
        self.visualisation_all = True
        self.gazebo_flag = True
        self.pw_T_rob_sim_pose_list = []
        self.pw_T_target_obj_obse_pose_lsit = []
        self.pw_T_target_obj_opti_pose_lsit = []
        self.pw_T_other_obj_opti_pose_list = []
        
        self.objects_name_list = ["cracker", "soup"]
        
    def initialize_visual_world_pybullet_env(self, task_flag):
        pw_T_rob_sim_pose_list = self.create_scene.initialize_robot()
        pw_T_target_obj_obse_pose_lsit, trans, rot = self.create_scene.initialize_object()
        pw_T_target_obj_opti_pose_lsit, pw_T_other_obj_opti_pose_list = self.create_scene.initialize_ground_truth_objects()
        
        if self.visualisation_all == True:
            p_visualisation = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT, GUI_SERVER
        else:
            p_visualisation = bc.BulletClient(connection_mode=p.DIRECT) # DIRECT, GUI_SERVER
        self.p_visualisation = p_visualisation
        p_visualisation.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_visualisation.setGravity(0, 0, -9.81)
        p_visualisation.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=180, cameraPitch=-85, cameraTargetPosition=[0.3,0.1,0.2])        
        plane_id = p_visualisation.loadURDF("plane.urdf")
        
        # load objects in the pybullet world
        for obj_index in range(self.object_num):
            obse_obj_name = pw_T_target_obj_obse_pose_lsit[obj_index].obj_name
            obse_obj_pos = pw_T_target_obj_obse_pose_lsit[obj_index].pos
            obse_obj_ori = pw_T_target_obj_obse_pose_lsit[obj_index].ori
            if self.gazebo_flag == True:
                obse_obj_name = "gazebo_" + obse_obj_name
            obse_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+obse_obj_name+"/"+obse_obj_name+"_obse_obj_with_visual_hor.urdf"),
                                                      obse_obj_pos,
                                                      obse_obj_ori)
            pw_T_target_obj_obse_pose_lsit[obj_index].obj_id = obse_object_id
            opti_obj_name = pw_T_target_obj_opti_pose_lsit[obj_index].obj_name
            opti_obj_pos = pw_T_target_obj_opti_pose_lsit[obj_index].pos
            opti_obj_ori = pw_T_target_obj_opti_pose_lsit[obj_index].ori
            if self.gazebo_flag == True:
                opti_obj_name = "gazebo_" + opti_obj_name
            opti_object_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+opti_obj_name+"/"+opti_obj_name+"_real_obj_with_visual_hor.urdf"),
                                                      opti_obj_pos,
                                                      opti_obj_ori)
            pw_T_target_obj_opti_pose_lsit[obj_index].obj_id = opti_object_id
        
        # load other objects in the pybullet world
        for obj_index in range(self.other_obj_num):
            other_obj_name = pw_T_other_obj_opti_pose_list[obj_index].obj_name
            other_obj_pos = pw_T_other_obj_opti_pose_list[obj_index].pos
            other_obj_ori = pw_T_other_obj_opti_pose_list[obj_index].ori
            if self.gazebo_flag == True:
                obse_obj_name = "gazebo_" + other_obj_name
            optitrack_base_id = p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+other_obj_name+"/base_of_cracker.urdf"),
                                                         other_obj_pos,
                                                         other_obj_ori)
            pw_T_other_obj_opti_pose_list[obj_index].obj_id = optitrack_base_id
        
        # load robot in the pybullet world
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
        for i in range(240):
            p_visualisation.stepSimulation()
            
        self.pw_T_rob_sim_pose_list = pw_T_rob_sim_pose_list
        self.pw_T_target_obj_obse_pose_lsit = pw_T_target_obj_obse_pose_lsit
        self.pw_T_target_obj_opti_pose_lsit = pw_T_target_obj_opti_pose_lsit
        self.pw_T_other_obj_opti_pose_list = pw_T_other_obj_opti_pose_list
        
        return p_visualisation, pw_T_rob_sim_pose_list, pw_T_target_obj_obse_pose_lsit, pw_T_target_obj_opti_pose_lsit, pw_T_other_obj_opti_pose_list
    
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
        obj_id = object_info_list.obj_id
        pybullet_simulation_env.resetBasePositionAndOrientation(obj_id,
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
        if self.gazebo_flag == True:
            obj_par_name = "gazebo_" + obj_par_name
        visualize_particle_Id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+obj_par_name+"/"+obj_par_name+"_par_with_visual_PB_hor.urdf"),
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
                if self.gazebo_flag == True:
                    obj_par_name = "gazebo_" + obj_par_name
                visualize_particle_Id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+obj_par_name+"/"+obj_par_name+"_par_with_visual_CV_hor.urdf"),
                                                                      obj_par_pos,
                                                                      obj_par_ori)
                obj_id_list.append(visualize_particle_Id)
                particle[obj_index].visual_par_id = visualize_particle_Id
                
    
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
        if self.gazebo_flag == True:
            esti_obj_name = "gazebo_" + esti_obj_name
        estimated_object_id = self.p_visualisation.loadURDF(os.path.expanduser("~/project/object/"+esti_obj_name+"/"+esti_obj_name+"_est_obj_with_visual_PB_hor.urdf"),
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
        
        
if __name__ == '__main__':
#    par_obj_id = [[]*2 for _ in range(50)]
#    print(par_obj_id)
#    input("stop")
    rospy.init_node('visualization_world') # ros node
    time.sleep(0.5)
    object_num = 1
    robot_num = 1
    other_obj_num = 0
    particle_num = 50
    init_par_flag = 0
    init_esti_flag = 0
    display_par_flag = True
    display_esti_flag = False
    object_list = ["cracker", "soup"]
    
    visual_world = Visualisation_World(object_num, robot_num, other_obj_num, particle_num)
    visual_world.initialize_visual_world_pybullet_env("task1")
    
    listener_tf = visual_world.listener
    p_visual = visual_world.p_visualisation
    pw_T_rob_sim_pose_list_param = visual_world.pw_T_rob_sim_pose_list
    pw_T_target_obj_obse_pose_lsit_param = visual_world.pw_T_target_obj_obse_pose_lsit
    pw_T_target_obj_opti_pose_lsit_param = visual_world.pw_T_target_obj_opti_pose_lsit
    pw_T_other_obj_opti_pose_list_param = visual_world.pw_T_other_obj_opti_pose_list
    
    if visual_world.gazebo_flag == True:
        panda_pose = visual_world.ros_listener.listen_2_robot_pose()
    
    par_obj_id = [[]*object_num for _ in range(particle_num)]
    esti_obj_id = [0] * object_num
    
    while True:
        # synchronize robot arm changes
        joint_states = visual_world.ros_listener.current_joint_values
        for rob_index in range(robot_num):
            rob_id = pw_T_rob_sim_pose_list_param[rob_index].obj_id
            pw_T_rob_sim_4_4 = pw_T_rob_sim_pose_list_param[rob_index].trans_matrix
            visual_world.set_real_robot_JointPosition(p_visual, rob_id, joint_states)
            
        for obj_index in range(object_num):
            # display ground truth (grtu)
            if visual_world.gazebo_flag == True:
                model_pose, model_pose_added_noise = visual_world.ros_listener.listen_2_object_pose(object_list[obj_index])
                
                gazebo_T_obj_pos = model_pose[0]
                gazebo_T_obj_ori = model_pose[1]
                gazebo_T_obj_pos_added_noise = model_pose_added_noise[0]
                gazebo_T_obj_ori_added_noise = model_pose_added_noise[1]
                gazebo_T_rob_pos = panda_pose[0]
                gazebo_T_rob_ori = panda_pose[1]
                
                opti_T_rob_opti_pos = copy.deepcopy(gazebo_T_rob_pos)
                opti_T_rob_opti_ori = copy.deepcopy(gazebo_T_rob_ori)
                opti_T_obj_opti_pos = copy.deepcopy(gazebo_T_obj_pos)
                opti_T_obj_opti_ori = copy.deepcopy(gazebo_T_obj_ori)
                opti_T_obj_obse_pos = copy.deepcopy(gazebo_T_obj_pos_added_noise)
                opti_T_obj_obse_ori = copy.deepcopy(gazebo_T_obj_ori_added_noise)
            else:
                opti_T_rob_opti_pos = visual_world.ros_listener.listen_2_robot_pose()[0]
                opti_T_rob_opti_ori = visual_world.ros_listener.listen_2_robot_pose()[1]
                opti_T_obj_opti_pos = visual_world.ros_listener.listen_2_object_pose(object_list[obj_index])[0]
                opti_T_obj_opti_ori = visual_world.ros_listener.listen_2_object_pose(object_list[obj_index])[1]
            # get ground truth data 
            rob_T_obj_opti_4_4 = compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, opti_T_obj_opti_pos, opti_T_obj_opti_ori)
            pw_T_obj_opti_4_4 = np.dot(pw_T_rob_sim_4_4, rob_T_obj_opti_4_4)
            pw_T_obj_opti_pos = [pw_T_obj_opti_4_4[0][3], pw_T_obj_opti_4_4[1][3], pw_T_obj_opti_4_4[2][3]]
            pw_T_obj_opti_ori = transformations.quaternion_from_matrix(pw_T_obj_opti_4_4)
            # update pose
            pw_T_target_obj_opti_pose_lsit_param[obj_index].pos = pw_T_obj_opti_pos
            pw_T_target_obj_opti_pose_lsit_param[obj_index].ori = pw_T_obj_opti_ori
            visual_world.display_object_in_visual_model(p_visual, pw_T_target_obj_opti_pose_lsit_param[obj_index])
            
            # get observation data
            if visual_world.gazebo_flag == True:
                obse_is_fresh = True
                rob_T_obj_obse_4_4 = compute_transformation_matrix(opti_T_rob_opti_pos, opti_T_rob_opti_ori, opti_T_obj_obse_pos, opti_T_obj_obse_ori)
            else:
                obse_is_fresh = True
                try:
                    latest_obse_time = listener_tf.getLatestCommonTime('/panda_link0', '/'+object_list[obj_index])
                    if (rospy.get_time() - latest_obse_time.to_sec()) < 0.1:
                        (trans,rot) = listener_tf.lookupTransform('/panda_link0', '/'+object_list[obj_index], rospy.Time(0))
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
            # update pose
            pw_T_target_obj_obse_pose_lsit_param[obj_index].pos = pw_T_obj_obse_pos
            pw_T_target_obj_obse_pose_lsit_param[obj_index].ori = pw_T_obj_obse_ori
            visual_world.display_object_in_visual_model(p_visual, pw_T_target_obj_obse_pose_lsit_param[obj_index])
            
        # display other objects
        for obj_index in range(other_obj_num):
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
        
        # display particles
        if display_par_flag == True:
            for obj_index in range(object_num):
                particles_states_list = visual_world.ros_listener.listen_2_pars_states()
                if len(particles_states_list.particles) == 0:
                    par_list_not_pub = 0
                else:
                    if init_par_flag == 0:
                        init_par_flag = 1
                        for par_index in range(particle_num):
                            visual_world.init_display_particle(particles_states_list.particles[par_index].objects[obj_index])
                            obj_visual_id = particles_states_list.particles[par_index].objects[obj_index].id
                            par_obj_id[par_index].append(obj_visual_id)
                    else:
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
                        init_esti_flag = 1
                        visual_world.init_display_estimated_object(esti_obj_states_list.objects[obj_index])
                        esti_obj_id[obj_index] = esti_obj_states_list.objects[obj_index].id
                    else:
                        esti_obj_states_list.objects[obj_index].id = esti_obj_id[obj_index]
                        visual_world.display_estimated_object_in_visual_model(esti_obj_states_list.objects[obj_index])
                    
        p_visual.stepSimulation()
        

#par_obj_id = [[]*object_num for _ in range(particle_num)]
#    esti_obj_id = [0] * object_num
        
        
        
        
