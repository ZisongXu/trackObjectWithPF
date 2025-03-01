#!/usr/bin/python3
import rospy
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3
from PBPF.msg import object_pose, particle_pose, particle_list, estimated_obj_pose
from gazebo_msgs.msg import ModelStates
from vision_msgs.msg import Detection3DArray


import tf
import tf.transformations as transformations
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

import copy
import random
import math
import numpy as np

from pyquaternion import Quaternion
import yaml
import os
#Class of franka robot listen to info from ROS
class Ros_Listener():
    def __init__(self, model=None):
        with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
            self.parameter_info = yaml.safe_load(file)
        self.gazebo_flag = self.parameter_info['gazebo_flag']
        self.object_name_list = self.parameter_info['object_name_list']
        self.object_num = self.parameter_info['object_num']
        self.current_joint_values = [0,0,0,0,0,0,0,0,0]
        self.pos_added_noise = []
        self.ori_added_noise = []
        self.model_pose_added_noise = []
        self.rob_T_obj_obse_4_4_list = []
        self.bridge = CvBridge()

        rospy.Subscriber('/joint_states', JointState, self.joint_values_callback, queue_size=1)
        self.joint_subscriber = JointState()

        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_image_callback, queue_size=1)
        # rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_image_callback, queue_size=1)
        self.depth_image_subscriber = Image()
        
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)
        self.model_states = ModelStates()
        
        rospy.Subscriber('/mocap/rigid_bodies/pandaRobot/pose', PoseStamped, self.robot_pose_callback, queue_size=1)
        self.robot_pose = PoseStamped()
        
        rospy.Subscriber('/mocap/rigid_bodies/cracker_opti/pose', PoseStamped, self.object_pose_callback_cracker, queue_size=1)
        self.object_cracker_pose = PoseStamped()
        
        rospy.Subscriber('/mocap/rigid_bodies/soup_opti/pose', PoseStamped, self.object_pose_callback_soup, queue_size=1)
        self.object_soup_pose = PoseStamped()
        
        rospy.Subscriber('/mocap/rigid_bodies/soup_opti2/pose', PoseStamped, self.object_pose_callback_soup2, queue_size=1)
        self.object_soup2_pose = PoseStamped()

        rospy.Subscriber('/mocap/rigid_bodies/Ketchup_opti/pose', PoseStamped, self.object_pose_callback_Ketchup, queue_size=1)
        self.object_Ketchup_pose = PoseStamped()

        rospy.Subscriber('/mocap/rigid_bodies/Milk_opti/pose', PoseStamped, self.object_pose_callback_Milk, queue_size=1)
        self.object_Milk_pose = PoseStamped()

        rospy.Subscriber('/mocap/rigid_bodies/Mustard_opti/pose', PoseStamped, self.object_pose_callback_Mustard, queue_size=1)
        self.object_Mustard_pose = PoseStamped()

        rospy.Subscriber('/mocap/rigid_bodies/Mayo_opti/pose', PoseStamped, self.object_pose_callback_Mayo, queue_size=1)
        self.object_Mayo_pose = PoseStamped()

        rospy.Subscriber('/mocap/rigid_bodies/Parmesan_opti/pose', PoseStamped, self.object_pose_callback_Parmesan, queue_size=1)
        self.object_Parmesan_pose = PoseStamped()

        rospy.Subscriber('/mocap/rigid_bodies/SaDressing_opti/pose', PoseStamped, self.object_pose_callback_SaladDressing, queue_size=1)
        self.object_SaladDressing_pose = PoseStamped()
        
        rospy.Subscriber('/mocap/rigid_bodies/gelation_opti/pose', PoseStamped, self.object_pose_callback_gelation, queue_size=1)
        self.object_gelation_pose = PoseStamped()

        rospy.Subscriber('/mocap/rigid_bodies/baseofcheezit/pose', PoseStamped, self.base_of_cheezit_callback, queue_size=1)
        self.base_pose = PoseStamped()

        rospy.Subscriber('/mocap/rigid_bodies/pringles_opti/pose', PoseStamped, self.other_object_pose_callback_pringles, queue_size=1)
        self.other_object_pringles_pose = PoseStamped()

        rospy.Subscriber('/mocap/rigid_bodies/smallObstacle/pose', PoseStamped, self.smallObstacle_callback, queue_size=1)
        self.smallObstacle = PoseStamped()
        
        rospy.Subscriber('/mocap/rigid_bodies/bigObstacle/pose', PoseStamped, self.bigObstacle_callback, queue_size=1)
        self.bigObstacle = PoseStamped()
        
        rospy.Subscriber('/Opti_pose', PoseStamped, self.fake_optipose_callback, queue_size=10)
        self.fake_opti_pose = PoseStamped()
        
        rospy.Subscriber('/esti_obj_list', estimated_obj_pose, self.esti_obj_states_callback, queue_size=10)
        self.esti_obj_states_list = estimated_obj_pose()
        
        rospy.Subscriber('/par_list', particle_list, self.particles_states_callback, queue_size=10)
        self.particles_states_list = particle_list()

        # rospy.Subscriber('/dope/detected_objects', Detection3DArray, self.detected_objects, queue_size=10)
        # self.detection_flag = Detection3DArray()

        # rospy.spin()
        
    # def detected_objects(self, detection_state):
    #     detection_info = detection_state.detections
    #     length_detection = len(detection_info)
    #     if length_detection == 0:
    #         self.detection_flag = False
    #     else:
    #         self.detection_flag =  True

    def depth_image_callback(self, depth_image_data):
        # print(len(depth_image_data.data))
        # print(depth_image_data.width)
        # print(depth_image_data.height)
        # print("================================")
        cv_image = self.bridge.imgmsg_to_cv2(depth_image_data,"16UC1")
        # cv_image = (cv_image).astype(np.uint16)
        self.depth_image = depth_image_data



    def model_states_callback(self, model_states):
        gzb_T_obj_obse_4_4 = [[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]]
        gzb_T_obj_obse_4_4 = np.array(gzb_T_obj_obse_4_4)  
        gzb_T_rob_obse_4_4 = [[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]]
        gzb_T_rob_obse_4_4 = np.array(gzb_T_rob_obse_4_4)  
        gzb_T_fish_obse_4_4 = [[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]]
        gzb_T_fish_obse_4_4 = np.array(gzb_T_rob_obse_4_4)
        name_lenght = len(model_states.name)
        gzb_T_obj_obse_4_4_list = []
        for name_index in range(name_lenght):
            if model_states.name[name_index] == "panda":
#                self.pos_added_noise, self.ori_added_noise = self.add_noise_pose(self.model_pos, self.model_ori)
#                self.model_pose_added_noise = [self.pos_added_noise, self.ori_added_noise]
                panda_name = model_states.name[name_index]
                panda_pos = model_states.pose[name_index].position
                panda_ori = model_states.pose[name_index].orientation
                self.panda_pos = [panda_pos.x, panda_pos.y, panda_pos.z]
                self.panda_ori = [panda_ori.x, panda_ori.y, panda_ori.z, panda_ori.w]
                self.panda_pose = [self.panda_pos, self.panda_ori]
                
                gzb_T_rob_obse_3_3 = transformations.quaternion_matrix(self.panda_ori)
                gzb_T_rob_obse_4_4 = self.rotation_4_4_to_transformation_4_4(gzb_T_rob_obse_3_3, self.panda_ori)
                robpos_T_pandalink0_4_4 = [[1., 0., 0.,    0.],
                                           [0., 1., 0.,    0.],
                                           [0., 0., 1.,  0.06],
                                           [0., 0., 0.,    1.]]
                robpos_T_pandalink0_4_4 = np.array(robpos_T_pandalink0_4_4)                
                gazebo_T_pandalink0_opti_4_4 = np.dot(gzb_T_rob_obse_4_4, robpos_T_pandalink0_4_4)
                pandalink0_T_gzb_obse_4_4 = np.linalg.inv(gazebo_T_pandalink0_opti_4_4)   
                
            for obj_num in range(self.object_num):    
                if model_states.name[name_index] == self.object_name_list[obj_num]:
                    model_name = model_states.name[name_index]
                    model_pos = model_states.pose[name_index].position
                    model_ori = model_states.pose[name_index].orientation
                    self.model_pos = [model_pos.x, model_pos.y, model_pos.z]
                    self.model_ori = [model_ori.x, model_ori.y, model_ori.z, model_ori.w]
                    self.model_pose = [self.model_pos, self.model_ori]
                    
                    gzb_T_obj_obse_3_3 = transformations.quaternion_matrix(self.model_ori)
                    gzb_T_obj_obse_4_4 = self.rotation_4_4_to_transformation_4_4(gzb_T_obj_obse_3_3, self.model_pos)
                    gzb_T_obj_obse_4_4_list.append(gzb_T_obj_obse_4_4)
    #                gzb_T_obj_opti_4_4 = np.dot(robpw_T_robga_4_4, rob_T_obj_opti_4_4)
                
                
#        pandalink0_T_gzb_obse_4_4 = np.linalg.inv(gzb_T_rob_obse_4_4)
        for obj_num in range(self.object_num):
            pandalink0_T_obj_obse_4_4 = np.dot(pandalink0_T_gzb_obse_4_4, gzb_T_obj_obse_4_4_list[obj_num])
            # print(pandalink0_T_obj_obse_4_4)
            self.rob_T_obj_obse_4_4_list.append(pandalink0_T_obj_obse_4_4)

    def listen_2_test_matrix(self):
        return self.rob_T_obj_obse_4_4_list

    def listen_2_object_pose(self, object_flag):
        # print("what you input is:", object_flag)
        if object_flag == "cracker":
            # print("==============")
            # print("cracker: In the Ros_Listener")
            # print(self.object_cracker_pose)
            # print("==============")
            if self.gazebo_flag == True:
                return self.model_pose, self.model_pose_added_noise
            return self.object_cracker_pose
        elif object_flag == "soup":
            # print("==============")
            # print("soup: In the Ros_Listener")
            # print(self.object_soup_pose)
            # print("==============")
            return self.object_soup_pose
        elif object_flag == "soup2":
            # print("==============")
            # print("soup: In the Ros_Listener")
            # print(self.object_soup_pose)
            # print("==============")
            return self.object_soup2_pose
        elif object_flag == "gelatin":
            # print("==============")
            # print("gelation: In the Ros_Listener")
            # print(self.object_gelation_pose)
            # print("==============")
            return self.object_gelation_pose
        elif object_flag == "Ketchup":
            # print("==============")
            # print("Ketchup: In the Ros_Listener")
            # print(self.object_Ketchup_pose)
            # print("==============")
            return self.object_Ketchup_pose
        elif object_flag == "Milk":
            # print("==============")
            # print("Milk: In the Ros_Listener")
            # print(self.object_Milk_pose)
            # print("==============")
            return self.object_Milk_pose
        elif object_flag == "Mustard":
            # print("==============")
            # print("Mustard: In the Ros_Listener")
            # print(self.object_Mustard_pose)
            # print("==============")
            return self.object_Mustard_pose
        elif object_flag == "Mayo":
            # print("==============")
            # print("Mayo: In the Ros_Listener")
            # print(self.object_Mayo_pose)
            # print("==============")
            return self.object_Mayo_pose
        elif object_flag == "Parmesan":
            # print("==============")
            # print("Parmesan: In the Ros_Listener")
            # print(self.object_Parmesan_pose)
            # print("==============")
            return self.object_Parmesan_pose
        elif object_flag == "SaladDressing":
            # print("==============")
            # print("SaladDressing: In the Ros_Listener")
            # print(self.object_SaladDressing_pose)
            # print("==============")
            return self.object_SaladDressing_pose
        elif object_flag == "base":
            return self.base_pose
        elif object_flag == "smallobstacle":
            return self.smallObstacle_pose
        elif object_flag == "bigbstacle":
            return self.bigObstacle_pose
        elif object_flag == "pringles":
            return self.other_object_pringles_pose
    
    def listen_2_pars_states(self):
        return self.particles_states_list
    
    def listen_2_estis_states(self):
        return self.esti_obj_states_list
    
    def listen_2_gazebo_robot_pose(self):
        return self.panda_pose
    
    def listen_2_robot_pose(self):
        if self.gazebo_flag == True:
            return self.panda_pose
        return self.robot_pose
    
    def joint_values_callback(self, msg):
        self.current_joint_values = list(msg.position)    
    
    def robot_pose_callback(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.robot_pos = [x_pos,y_pos,z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.robot_ori = [x_ori,y_ori,z_ori,w_ori]
        self.robot_pose = [self.robot_pos, self.robot_ori]
    
    def object_pose_callback_cracker(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_cracker_pose = [self.object_pos, self.object_ori]

    def object_pose_callback_gelation(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_gelation_pose = [self.object_pos, self.object_ori]

    def object_pose_callback_soup(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_soup_pose = [self.object_pos, self.object_ori]

    def object_pose_callback_soup2(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_soup2_pose = [self.object_pos, self.object_ori]
    
    def object_pose_callback_Ketchup(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_Ketchup_pose = [self.object_pos, self.object_ori]
    
    def object_pose_callback_Milk(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_Milk_pose = [self.object_pos, self.object_ori]
    
    def object_pose_callback_Mustard(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_Mustard_pose = [self.object_pos, self.object_ori]
    
    def object_pose_callback_Mayo(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_Mayo_pose = [self.object_pos, self.object_ori]
    
    def object_pose_callback_Parmesan(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_Parmesan_pose = [self.object_pos, self.object_ori]

    def object_pose_callback_SaladDressing(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.object_pos = [x_pos, y_pos, z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.object_ori = [x_ori, y_ori, z_ori, w_ori]
        self.object_SaladDressing_pose = [self.object_pos, self.object_ori]
        
    def base_of_cheezit_callback(self, data):
        # pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.base_pos = [x_pos, y_pos, z_pos]
        # ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.base_ori = [x_ori, y_ori, z_ori, w_ori]
        self.base_pose = [self.base_pos, self.base_ori]
        # print("self.base_pose:", self.base_pose)

    def other_object_pose_callback_pringles(self, data):
        # pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.other_object_pringles_pos = [x_pos, y_pos, z_pos]
        # ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.other_object_pringles_ori = [x_ori, y_ori, z_ori, w_ori]
        self.other_object_pringles_pose = [self.other_object_pringles_pos, self.other_object_pringles_ori]

    def smallObstacle_callback(self, data):
        # pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.smallObstacle_pos = [x_pos, y_pos, z_pos]
        # ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.smallObstacle_ori = [x_ori, y_ori, z_ori, w_ori]
        self.smallObstacle_pose = [self.smallObstacle_pos, self.smallObstacle_ori]
        # print("self.smallObstacle_pose:", self.smallObstacle_pose)

    def bigObstacle_callback(self, data):
        # pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.bigObstacle_pos = [x_pos, y_pos, z_pos]
        # ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.bigObstacle_ori = [x_ori, y_ori, z_ori, w_ori]
        self.bigObstacle_pose = [self.bigObstacle_pos, self.bigObstacle_ori]
        # print("self.bigObstacle_pose:", self.bigObstacle_pose)
        
    def fake_optipose_callback(self, data):
        # pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.fake_opti_pos = [x_pos, y_pos, z_pos]
        # ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.fake_opti_ori = [x_ori, y_ori, z_ori, w_ori]
        self.fake_opti_pose = [self.fake_opti_pos, self.fake_opti_ori]

    def particles_states_callback(self, pars_states_list):
        self.particles_states_list = pars_states_list
    
    def esti_obj_states_callback(self, esti_objs_list):
        self.esti_obj_states_list = esti_objs_list
        
    def add_noise_pose(self, sim_par_cur_pos, sim_par_cur_ori):
        normal_x = self.add_noise_2_par(sim_par_cur_pos[0])
        normal_y = self.add_noise_2_par(sim_par_cur_pos[1])
        normal_z = self.add_noise_2_par(sim_par_cur_pos[2])
        pos_added_noise = [normal_x, normal_y, normal_z]
        # add noise on ang of each particle
        quat = copy.deepcopy(sim_par_cur_ori)# x,y,z,w
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
        ori_added_noise = [new_quat[1],new_quat[2],new_quat[3],new_quat[0]]
        return pos_added_noise, ori_added_noise
    
    def add_noise_2_par(self, current_pos):
        mean = current_pos
        pos_noise_sigma = 0.01
        sigma = pos_noise_sigma
        new_pos_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_pos_is_added_noise
    
    def add_noise_2_ang(self, cur_angle):
        mean = cur_angle
        ang_noise_sigma = 0.1
        sigma = ang_noise_sigma
        new_angle_is_added_noise = self.take_easy_gaussian_value(mean, sigma)
        return new_angle_is_added_noise
    
    def take_easy_gaussian_value(self, mean, sigma):
        normal = random.normalvariate(mean, sigma)
        return normal
    
    def rotation_4_4_to_transformation_4_4(self, rotation_4_4, pos):
        rotation_4_4[0][3] = pos[0]
        rotation_4_4[1][3] = pos[1]
        rotation_4_4[2][3] = pos[2]
        return rotation_4_4