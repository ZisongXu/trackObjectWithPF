# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""
from __future__ import print_function
from doctest import FAIL_FAST

import cv2
import message_filters
import numpy as np
import resource_retriever
import rospy
import tf.transformations
from PIL import Image
from PIL import ImageDraw
from cv_bridge import CvBridge
# from dope.inference.cuboid import Cuboid3d
# from dope.inference.cuboid_pnp_solver import CuboidPNPSolver
# from dope.inference.detector import ModelData, ObjectDetector
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image as ImageSensor_msg
# from std_msgs.msg import String
# from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
# from visualization_msgs.msg import Marker, MarkerArray
import tf
import tf.transformations as transformations
import time
import os
import yaml

# with open(os.path.expanduser("~/catkin_ws/src/PBPF/config/parameter_info.yaml"), 'r') as file:
#     parameter_info = yaml.safe_load(file)

# object_name_list = parameter_info['object_name_list']
object_num = 1

# the flag is used to determine whether the robot touches the particle in the simulation
simRobot_touch_par_flag = 0

robot_num = 1
class Ros_listener_PFPE():
    def __init__(self):
        # pose_PFPE = rospy.Subscriber('/esti_obj_list', estimated_obj_pose, self.pose_PFPE_callback, queue_size=1)

        pose_PFPE = rospy.Subscriber('PBPF_pose', PoseStamped, self.pose_PFPE_callback, queue_size=1)
        pose_DOPE = rospy.Subscriber('DOPE_pose', PoseStamped, self.pose_DOPE_callback, queue_size=1)
        # pose_Opti = rospy.Subscriber('Opti_pose', PoseStamped, self.pose_Opti_callback, queue_size=1)
        self.current_joint_values = [-1.57,0.0,0.0,-2.8,1.7,1.57,1.1]
        self.PFPE_pos = [ 0.139080286026,
                         -0.581342339516,
                         0.0238141193986]
        #x,y,z,w
        self.PFPE_ori = [ 0.707254290581,
                          0.0115503482521,
                         -0.0140119809657,
                         -0.706726074219]
        
        self.DOPE_pos = [ 0.139080286026,
                         -0.581342339516,
                         0.0238141193986]
        #x,y,z,w
        self.DOPE_ori = [ 0.707254290581,
                          0.0115503482521,
                         -0.0140119809657,
                         -0.706726074219]
        
        self.Opti_pos = [ 0.139080286026,
                         -0.581342339516,
                         0.0238141193986]
        #x,y,z,w
        self.Opti_ori = [ 0.707254290581,
                          0.0115503482521,
                         -0.0140119809657,
                         -0.706726074219]

        rospy.spin

    def pose_PFPE_callback(self, data):
        # need to change
        for obj_index in range(object_num):
            # esti_obj_pos_x = data.objects[obj_index].pose.position.x
            # esti_obj_pos_y = data.objects[obj_index].pose.position.y
            # esti_obj_pos_z = data.objects[obj_index].pose.position.z
            
            # esti_obj_ori_x = data.objects[obj_index].pose.orientation.x
            # esti_obj_ori_y = data.objects[obj_index].pose.orientation.y
            # esti_obj_ori_z = data.objects[obj_index].pose.orientation.z
            # esti_obj_ori_w = data.objects[obj_index].pose.orientation.w

            #pos
            x_pos = data.pose.position.x
            y_pos = data.pose.position.y
            z_pos = data.pose.position.z
            # self.PFPE_pos = [esti_obj_pos_x, esti_obj_pos_y, esti_obj_pos_z]
            self.PFPE_pos = [x_pos, y_pos, z_pos]
            #ori
            x_ori = data.pose.orientation.x
            y_ori = data.pose.orientation.y
            z_ori = data.pose.orientation.z
            w_ori = data.pose.orientation.w
            # self.PFPE_ori = [esti_obj_ori_x, esti_obj_ori_y, esti_obj_ori_z, esti_obj_ori_w]
            self.PFPE_ori = [x_ori, y_ori, z_ori, w_ori]
    def pose_DOPE_callback(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.DOPE_pos = [x_pos,y_pos,z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.DOPE_ori = [x_ori,y_ori,z_ori,w_ori]
    def pose_Opti_callback(self, data):
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.Opti_pos = [x_pos,y_pos,z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.Opti_ori = [x_ori,y_ori,z_ori,w_ori]
        # print(self.Opti_pos)
        # print(self.Opti_ori)
        
class Ros_listener_OPTI():
    def __init__(self):
        pose_Opti = rospy.Subscriber('Opti_pose', PoseStamped, self.pose_Opti_callback, queue_size=1)
        # print("I am heree")
        self.Opti_pos = [ 0.139080286026,
                         -0.581342339516,
                         0.0238141193986]
        #x,y,z,w
        self.Opti_ori = [ 0.707254290581,
                          0.0115503482521,
                         -0.0140119809657,
                         -0.706726074219]
        rospy.spin

    def pose_Opti_callback(self, data):
        # print("I am hereeeeeeee")
        #pos
        x_pos = data.pose.position.x
        y_pos = data.pose.position.y
        z_pos = data.pose.position.z
        self.Opti_pos = [x_pos,y_pos,z_pos]
        #ori
        x_ori = data.pose.orientation.x
        y_ori = data.pose.orientation.y
        z_ori = data.pose.orientation.z
        w_ori = data.pose.orientation.w
        self.Opti_ori = [x_ori,y_ori,z_ori,w_ori]
        # print(self.Opti_pos)
        # print(self.Opti_ori)
        
        
class Ros_listener():
    def __init__(self):
        # Start ROS publishers
        self.pub_rgb_dope_points = \
            rospy.Publisher(
                rospy.get_param('~topic_publishing') + "/rgb_points",
                ImageSensor_msg,
                queue_size=10
            )

        self.cv_bridge = CvBridge()
        image_sub = message_filters.Subscriber('/camera/color/image_raw', ImageSensor_msg)
        
        ts = message_filters.TimeSynchronizer([image_sub], 1)
        ts.registerCallback(self.image_callback)
        rospy.spin()
    
    def image_callback(self, image_msg):
        img = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
        height, width, _ = img.shape
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        im = im.convert('RGB')
        draw = Draw(im)
        if object_soup_flag == True:
            length = 0.1000 / 2
            width = 0.023829689025878906
            height = 0.023829689025878906
        elif object_cheezit_flag == True:
            length = 0.1067185
            width = 0.0305
            # height = 0.08
            height = 0.065
        point_0 = [-1 * height, 1 * length, 1 * width]
        point_1 = [ 1 * height, 1 * length, 1 * width]
        point_2 = [ 1 * height,-1 * length, 1 * width]
        point_3 = [-1 * height,-1 * length, 1 * width]
        point_4 = [-1 * height, 1 * length,-1 * width]
        point_5 = [ 1 * height, 1 * length,-1 * width]
        point_6 = [ 1 * height,-1 * length,-1 * width]
        point_7 = [-1 * height,-1 * length,-1 * width]
        rvecs = np.array([[0.0], 
                          [0.0], 
                          [0.0]])
        tvecs = np.array([[0.0], 
                          [0.0], 
                          [0.0]])
        if DOPE_flag == True:
            x_par = 0
            y_par = 0
            color = (100, 149, 237) # dark blue
            color = (255, 255, 0) # yellow
            # color = (138, 43, 226) # purple
            # color = (0, 255, 0) # green
            # color = (0, 255, 255) # light blue
            # color = (255, 0, 0) # red
            # while True:
            #     try:
            #         (trans,rot) = listener.lookupTransform('/RealSense', '/cracker', rospy.Time(0))
            #         break
            #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #         continue
            # camera_T_obj_dope_pos = list(trans)
            # camera_T_obj_dope_ori = list(rot)
            # camera_T_obj_dope_3_3 = transformations.quaternion_matrix(camera_T_obj_dope_ori)
            # camera_T_obj_dope_4_4 = rotation_4_4_to_transformation_4_4(camera_T_obj_dope_3_3,camera_T_obj_dope_pos)

            pw_T_obj_DOPE_pos = PFPE_listener.DOPE_pos
            pw_T_obj_DOPE_ori = PFPE_listener.DOPE_ori
            # pybullet_robot_pos
            # # pybullet_robot_ori
            # print(pw_T_obj_DOPE_pos)
            # print(pw_T_obj_DOPE_ori)
            while True:
                try:
                    if optitrack_working_flag == True:
                        (trans,rot) = listener.lookupTransform('/RealSense', '/pandaRobot', rospy.Time(0))
                    else:
                        (trans,rot) = listener.lookupTransform('/ar_tracking_camera_frame', '/panda_link0', rospy.Time(0))
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            camera_T_rob_dope_pos = list(trans)
            camera_T_rob_dope_ori = list(rot)
            camera_T_rob_dope_3_3 = transformations.quaternion_matrix(camera_T_rob_dope_ori)
            camera_T_rob_dope_4_4 = rotation_4_4_to_transformation_4_4(camera_T_rob_dope_3_3,camera_T_rob_dope_pos)
            pw_T_obj_DOPE_3_3 = transformations.quaternion_matrix(pw_T_obj_DOPE_ori)
            pw_T_obj_DOPE_4_4 = rotation_4_4_to_transformation_4_4(pw_T_obj_DOPE_3_3,pw_T_obj_DOPE_pos)
            pw_T_rob_DOPE_3_3 = transformations.quaternion_matrix(pybullet_robot_ori)
            pw_T_rob_DOPE_4_4 = rotation_4_4_to_transformation_4_4(pw_T_rob_DOPE_3_3,pybullet_robot_ori)
            rob_T_pw_DOPE_4_4 = np.linalg.inv(pw_T_rob_DOPE_4_4)
            camera_T_pw_4_4 = np.dot(camera_T_rob_dope_4_4, rob_T_pw_DOPE_4_4)
            camera_T_obj_dope_4_4 = np.dot(camera_T_pw_4_4, pw_T_obj_DOPE_4_4)
            
            point_ori = [0,0,0,1]
            point_3_3 = transformations.quaternion_matrix(point_ori)
            camera_T_point_0_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_0)
            camera_T_point_1_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_1)
            camera_T_point_2_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_2)
            camera_T_point_3_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_3)
            camera_T_point_4_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_4)
            camera_T_point_5_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_5)
            camera_T_point_6_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_6)
            camera_T_point_7_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_7)
            camera_T_points = []
            camera_T_points.append(camera_T_point_0_pos)
            camera_T_points.append(camera_T_point_1_pos)
            camera_T_points.append(camera_T_point_2_pos)
            camera_T_points.append(camera_T_point_3_pos)
            camera_T_points.append(camera_T_point_4_pos)
            camera_T_points.append(camera_T_point_5_pos)
            camera_T_points.append(camera_T_point_6_pos)
            camera_T_points.append(camera_T_point_7_pos)
            
            results_cv2_points = []
            for i in range(len(camera_T_points)):
                results_cv2_point = cv2.projectPoints(np.array(camera_T_points[i]),
                                                    rvecs,
                                                    tvecs,
                                                    np.array(_camera_intrinsic_matrix),
                                                    np.array(_dist_coeffs))
                results_cv2_points.append(tuple([results_cv2_point[0][0][0][0]+x_par,results_cv2_point[0][0][0][1]+y_par]))
            points_DOPE = [results_cv2_points[0], 
                    results_cv2_points[1], 
                    results_cv2_points[2], 
                    results_cv2_points[3], 
                    results_cv2_points[4], 
                    results_cv2_points[5], 
                    results_cv2_points[6], 
                    results_cv2_points[7]]
            t_middle = time.time()
            # print(t_middle - t_begin)
            draw.draw_cube(points_DOPE, color)
            
        if PFPE_flag == True:
            x_par = 0
            y_par = 0
            color = (0, 255, 0) # green
            pw_T_obj_PFPE_pos = PFPE_listener.PFPE_pos
            pw_T_obj_PFPE_ori = PFPE_listener.PFPE_ori
            # pybullet_robot_pos
            # pybullet_robot_ori
            # print(pw_T_obj_PFPE_pos)
            # print(pw_T_obj_PFPE_ori)
            while True:
                try:
                    if optitrack_working_flag == True:
                        (trans,rot) = listener.lookupTransform('/RealSense', '/pandaRobot', rospy.Time(0))
                    else:
                        (trans,rot) = listener.lookupTransform('/ar_tracking_camera_frame', '/panda_link0', rospy.Time(0))
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            camera_T_rob_dope_pos = list(trans)
            camera_T_rob_dope_ori = list(rot)
            camera_T_rob_dope_3_3 = transformations.quaternion_matrix(camera_T_rob_dope_ori)
            camera_T_rob_dope_4_4 = rotation_4_4_to_transformation_4_4(camera_T_rob_dope_3_3,camera_T_rob_dope_pos)
            pw_T_obj_PFPE_3_3 = transformations.quaternion_matrix(pw_T_obj_PFPE_ori)
            pw_T_obj_PFPE_4_4 = rotation_4_4_to_transformation_4_4(pw_T_obj_PFPE_3_3,pw_T_obj_PFPE_pos)
            pw_T_rob_PFPE_3_3 = transformations.quaternion_matrix(pybullet_robot_ori)
            pw_T_rob_PFPE_4_4 = rotation_4_4_to_transformation_4_4(pw_T_rob_PFPE_3_3,pybullet_robot_ori)
            rob_T_pw_PFPE_4_4 = np.linalg.inv(pw_T_rob_PFPE_4_4)
            camera_T_pw_4_4 = np.dot(camera_T_rob_dope_4_4, rob_T_pw_PFPE_4_4)
            camera_T_obj_dope_4_4 = np.dot(camera_T_pw_4_4, pw_T_obj_PFPE_4_4)
            
            point_ori = [0,0,0,1]
            point_3_3 = transformations.quaternion_matrix(point_ori)
            camera_T_point_0_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_0)
            camera_T_point_1_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_1)
            camera_T_point_2_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_2)
            camera_T_point_3_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_3)
            camera_T_point_4_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_4)
            camera_T_point_5_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_5)
            camera_T_point_6_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_6)
            camera_T_point_7_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_7)
            camera_T_points = []
            camera_T_points.append(camera_T_point_0_pos)
            camera_T_points.append(camera_T_point_1_pos)
            camera_T_points.append(camera_T_point_2_pos)
            camera_T_points.append(camera_T_point_3_pos)
            camera_T_points.append(camera_T_point_4_pos)
            camera_T_points.append(camera_T_point_5_pos)
            camera_T_points.append(camera_T_point_6_pos)
            camera_T_points.append(camera_T_point_7_pos)
            
            results_cv2_points = []
            for i in range(len(camera_T_points)):
                results_cv2_point = cv2.projectPoints(np.array(camera_T_points[i]),
                                                    rvecs,
                                                    tvecs,
                                                    np.array(_camera_intrinsic_matrix),
                                                    np.array(_dist_coeffs))
                results_cv2_points.append(tuple([results_cv2_point[0][0][0][0]+x_par,results_cv2_point[0][0][0][1]+y_par]))
            points_PFPE = [results_cv2_points[0], 
                    results_cv2_points[1], 
                    results_cv2_points[2], 
                    results_cv2_points[3], 
                    results_cv2_points[4], 
                    results_cv2_points[5], 
                    results_cv2_points[6], 
                    results_cv2_points[7]]
            draw.draw_cube(points_PFPE, color)
        if Opti_flag == True:
            x_par = 0
            y_par = 0
            color = (0, 255, 255) # light blue
            pw_T_obj_Opti_pos = OPTI_listener.Opti_pos
            pw_T_obj_Opti_ori = OPTI_listener.Opti_ori
            # pybullet_robot_pos
            # pybullet_robot_ori
            # print("pw_T_obj_PFPE_pos:", pw_T_obj_Opti_pos)
            # print("pw_T_obj_PFPE_ori:", pw_T_obj_Opti_ori)
            while True:
                try:
                    if optitrack_working_flag == True:
                        (trans,rot) = listener.lookupTransform('/RealSense', '/pandaRobot', rospy.Time(0))
                    else:
                        (trans,rot) = listener.lookupTransform('/ar_tracking_camera_frame', '/panda_link0', rospy.Time(0))
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
            camera_T_rob_dope_pos = list(trans)
            camera_T_rob_dope_ori = list(rot)
            camera_T_rob_dope_3_3 = transformations.quaternion_matrix(camera_T_rob_dope_ori)
            camera_T_rob_dope_4_4 = rotation_4_4_to_transformation_4_4(camera_T_rob_dope_3_3,camera_T_rob_dope_pos)
            pw_T_obj_Opti_3_3 = transformations.quaternion_matrix(pw_T_obj_Opti_ori)
            pw_T_obj_Opti_4_4 = rotation_4_4_to_transformation_4_4(pw_T_obj_Opti_3_3,pw_T_obj_Opti_pos)
            pw_T_rob_Opti_3_3 = transformations.quaternion_matrix(pybullet_robot_ori)
            pw_T_rob_Opti_4_4 = rotation_4_4_to_transformation_4_4(pw_T_rob_Opti_3_3,pybullet_robot_ori)
            rob_T_pw_Opti_4_4 = np.linalg.inv(pw_T_rob_Opti_4_4)
            camera_T_pw_4_4 = np.dot(camera_T_rob_dope_4_4, rob_T_pw_Opti_4_4)
            camera_T_obj_dope_4_4 = np.dot(camera_T_pw_4_4, pw_T_obj_Opti_4_4)
            
            point_ori = [0,0,0,1]
            point_3_3 = transformations.quaternion_matrix(point_ori)
            camera_T_point_0_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_0)
            camera_T_point_1_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_1)
            camera_T_point_2_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_2)
            camera_T_point_3_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_3)
            camera_T_point_4_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_4)
            camera_T_point_5_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_5)
            camera_T_point_6_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_6)
            camera_T_point_7_pos = point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point_7)
            camera_T_points = []
            camera_T_points.append(camera_T_point_0_pos)
            camera_T_points.append(camera_T_point_1_pos)
            camera_T_points.append(camera_T_point_2_pos)
            camera_T_points.append(camera_T_point_3_pos)
            camera_T_points.append(camera_T_point_4_pos)
            camera_T_points.append(camera_T_point_5_pos)
            camera_T_points.append(camera_T_point_6_pos)
            camera_T_points.append(camera_T_point_7_pos)
            
            results_cv2_points = []
            for i in range(len(camera_T_points)):
                results_cv2_point = cv2.projectPoints(np.array(camera_T_points[i]),
                                                    rvecs,
                                                    tvecs,
                                                    np.array(_camera_intrinsic_matrix),
                                                    np.array(_dist_coeffs))
                results_cv2_points.append(tuple([results_cv2_point[0][0][0][0]+x_par,results_cv2_point[0][0][0][1]+y_par]))
            points_OPTI = [results_cv2_points[0], 
                    results_cv2_points[1], 
                    results_cv2_points[2], 
                    results_cv2_points[3], 
                    results_cv2_points[4], 
                    results_cv2_points[5], 
                    results_cv2_points[6], 
                    results_cv2_points[7]]
            draw.draw_cube(points_OPTI, color)

        rgb_points_img = CvBridge().cv2_to_imgmsg(np.array(im)[..., ::-1], "bgr8")
        # rgb_points_img.header = camera_info.header
        self.pub_rgb_dope_points.publish(rgb_points_img)
        
class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)

    def draw_line(self, point1, point2, line_color, line_width=5):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            # print(point1)
            # print(point2)
            self.draw.line([point1, point2], fill=line_color, width=line_width)
#            self.draw.draw_lines([point1, point2], fill=line_color, width=line_width)
            
    def draw_dot(self, point, point_color, point_radius):
        """Draws dot (filled circle) on image"""
        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius
            ]
            self.draw.ellipse(xy,
                              fill=point_color,
                              outline=point_color
                              )

    def draw_cube(self, points, color=(255, 0, 0)):
        """
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        """
        
        # draw front
        self.draw_line(points[0], points[1], color)
        self.draw_line(points[1], points[2], color)
        self.draw_line(points[3], points[2], color)
        self.draw_line(points[3], points[0], color)

        # draw back
        self.draw_line(points[4], points[5], color)
        self.draw_line(points[6], points[5], color)
        self.draw_line(points[6], points[7], color)
        self.draw_line(points[4], points[7], color)

        # draw sides
        self.draw_line(points[0], points[4], color)
        self.draw_line(points[7], points[3], color)
        self.draw_line(points[5], points[1], color)
        self.draw_line(points[2], points[6], color)

        # draw dots
        self.draw_dot(points[0], point_color=color, point_radius=4)
        self.draw_dot(points[1], point_color=color, point_radius=4)

        # draw x on the top
        self.draw_line(points[0], points[5], color)
        self.draw_line(points[1], points[4], color)
        
#cv_bridge = CvBridge()
#img = cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")

#draw = Draw()
#draw.draw_cube(points2d, self.draw_colors[m])

def rotation_4_4_to_transformation_4_4(rotation_4_4,pos):
    rotation_4_4[0][3] = pos[0]
    rotation_4_4[1][3] = pos[1]
    rotation_4_4[2][3] = pos[2]
    return rotation_4_4
def point_4_4_matrix(camera_T_obj_dope_4_4, point_3_3, point):
    point_4_4 = rotation_4_4_to_transformation_4_4(point_3_3,point)
    camera_T_point = np.dot(camera_T_obj_dope_4_4,point_4_4)
    point_pos = [camera_T_point[0][3],camera_T_point[1][3],camera_T_point[2][3]]
    return point_pos
def main():
    """Main routine to run DOPE"""

    # Initialize ROS node
    rospy.init_node('draw_box')
    listener = tf.TransformListener()
    while True:
        try:
            (trans,rot) = listener.lookupTransform('/RealSense', '/cracker', rospy.Time(0))
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    camera_T_obj_dope_pos = list(trans)
    camera_T_obj_dope_ori = list(rot)
    
    ros_listen = Ros_listener()
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
# _camera_intrinsic_matrix = [[504.91994222,          0.0, 259.28746541],
#                             [         0.0, 503.71668498, 212.89422353],
#                             [         0.0,          0.0,          1.0]]
#_camera_intrinsic_matrix = [[908.8558959960938,               0.0, 626.7174072265625],
#                            [              0.0, 906.6900634765625, 383.2095947265625],
#                            [              0.0,               0.0,               1.0]]

_camera_intrinsic_matrix = [[605.903930664062,               0.0, 311.144958496094],
                            [              0.0, 604.460021972656, 255.473068237305],
                            [              0.0,               0.0,               1.0]]

_dist_coeffs = [[0.], [0.], [0.], [0.]]
pybullet_robot_pos = [0.0, 0.0, 0.026]
pybullet_robot_ori = [0,0,0,1]
if __name__ == "__main__":
    # main()
    t_begin = time.time()
    DOPE_flag = True
    PFPE_flag = True
    Opti_flag = False
    optitrack_working_flag = True
    object_soup_flag = False
    object_cheezit_flag = True
    task_flag = "1"
    rospy.init_node('draw_box')
    listener = tf.TransformListener()
    PFPE_listener = Ros_listener_PFPE()
    OPTI_listener = Ros_listener_OPTI()
    ros_listener = Ros_listener()
