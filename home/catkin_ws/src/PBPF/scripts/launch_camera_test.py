# #!/usr/bin/python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Mar 10 10:57:49 2021

# @author: 12106
# """
# from gazebo_msgs.msg import ModelStates
# #ROS
# from concurrent.futures.process import _threads_wakeups
# import itertools
# import os.path
# from pickle import TRUE
# from re import T
# from ssl import ALERT_DESCRIPTION_ILLEGAL_PARAMETER
# from tkinter.tix import Tree
# import rospy
# import threading
# import rospkg
# from std_msgs.msg import String
# from std_msgs.msg import Float32
# from std_msgs.msg import Int8
# from std_msgs.msg import ColorRGBA, Header
# from visualization_msgs.msg import Marker
# from sensor_msgs.msg import JointState
# from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped, Vector3
# from PBPF.msg import estimated_obj_pose, object_pose, particle_list, particle_pose
# import tf
# import tf.transformations as transformations
# from cv_bridge import CvBridge
# from cv_bridge import CvBridgeError
# import cv2
# #pybullet
# from pyquaternion import Quaternion
# import pybullet as p
# import time
# import pybullet_data
# from pybullet_utils import bullet_client as bc
# import numpy as np
# import math
# import random
# import copy
# import os
# import signal
# import sys
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imsave
# import pandas as pd
# import multiprocessing
# import yaml
# #from sksurgerycore.algorithms.averagequaternions import average_quaternions
# from quaternion_averaging import weightedAverageQuaternions
# #class in other files
# from Franka_robot import Franka_robot
# from Ros_Listener import Ros_Listener
# from Particle import Particle
# from InitialSimulationModel import InitialSimulationModel
# from Realworld import Realworld
# from Visualisation_World import Visualisation_World
# from Create_Scene import Create_Scene
# from Object_Pose import Object_Pose
# from Robot_Pose import Robot_Pose
# from Center_T_Point_for_Ray import Center_T_Point_for_Ray
# from launch_camera import LaunchCamera

# class ExpWorld():
#     def __init__(self):
#         self.pos_x_sigma = 0.05
#         self.pos_y_sigma = 0.05
#         self.pos_z_sigma = 0
#         self.ang_x_sigma = 0
#         self.ang_y_sigma = 0
#         self.ang_z_sigma = 0
#     def real_world_scene(self):
#         p_real_world = bc.BulletClient(connection_mode=p.GUI_SERVER) # DIRECT, GUI_SERVER
#         # physicsClient = p.connect(p.direct)
#         p_real_world.setAdditionalSearchPath(pybullet_data.getDataPath())
#         p_real_world.setGravity(0.2,0,-9.81)
#         p_real_world.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])#转变视角
#         plane_id = p_real_world.loadURDF("plane.urdf")
#         #load and set real robot
#         real_world_cracker_pos = [0, 0.2, 0.081]
#         real_world_cracker_ori = p_real_world.getQuaternionFromEuler([0, math.pi/2.0, 0])
#         real_world_cracker_id = p_real_world.loadURDF(os.path.expanduser("~/project/object/cracker/cracker_par_no_visual_hor.urdf"),
#                                                     real_world_cracker_pos,
#                                                     real_world_cracker_ori)
#         return p_real_world, real_world_cracker_pos, real_world_cracker_ori
# class LaunchCamera():
#     def __init__(self, width, height):
#         self.a = 0
#         self.pixelWidth = width
#         self.pixelHeight = height
#     def setCameraPicAndGetPic(self, p_world=0):
#         # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
        
#         pw_T_cam_tf = [[-0.01873203,  0.57812041, -0.81573642,  1.14657295],
#                        [ 0.99966816, -0.00360018, -0.0255072 ,  0.13681107],
#                        [-0.01768303, -0.81594352, -0.57786113,  0.41750256],
#                        [ 0.        ,  0.        ,  0.        ,  1.        ]]
#         pw_T_cam_tf_pos = [pw_T_cam_tf[0][3], pw_T_cam_tf[1][3], pw_T_cam_tf[2][3]]
#         pw_T_cam_tf_ori = transformations.quaternion_from_matrix(pw_T_cam_tf)
#         print("pw_T_cam_tf_ori")
#         print(pw_T_cam_tf_ori)
#         pw_T_cam_tf_ori = quaternion_correction(pw_T_cam_tf_ori) # x, y, z, w
#         # getEulerFromQuaternion returns a list of 3 floating point values, a vec3. 
#         # The rotation order is first roll around X, then pitch around Y and finally yaw around Z, as in the ROS URDF rpy convention.

#         pw_T_cam_tf_radian = p_world.getEulerFromQuaternion(pw_T_cam_tf_ori)
#         x_rad = pw_T_cam_tf_radian[0]
#         y_rad = pw_T_cam_tf_radian[1]
#         z_rad = pw_T_cam_tf_radian[2]
#         RADIAN_TO_ANGLE_VALUE = 180.0 / math.pi
#         x_ang = RADIAN_TO_ANGLE_VALUE * x_rad
#         y_ang = RADIAN_TO_ANGLE_VALUE * y_rad
#         z_ang = RADIAN_TO_ANGLE_VALUE * z_rad
#         tz_vec = [0.0, 0.0, 1]
#         cameraPos = [-0.4, -0.4, 0.4]
#         cameraPos = pw_T_cam_tf_pos
#         targetPos = [0.0, 0.0, 0.0]
#         viewMatrix = p_world.computeViewMatrix(
#             cameraEyePosition=cameraPos,
#             cameraTargetPosition=targetPos,
#             cameraUpVector=tz_vec
#         )
#         print("x:", x_ang)
#         print("y:", y_ang)
#         print("z:", z_ang)
#         # Facing y
#         camTargetPos = [0., 0., 0.3]
#         camDistance = 1
#         yaw = 45 # z
#         pitch = -90 # x
#         roll = 45 # y
#         yaw = z_ang # z  z: 91.07349685292553
#         pitch = y_ang # x  y: 1.0132157040776695
#         roll = -x_ang # y  x: -125.3065901872438
#         upAxisIndex = 1
#         viewMatrix = p_world.computeViewMatrixFromYawPitchRoll(camTargetPos, 
#                                                                camDistance, 
#                                                                yaw, pitch, roll,
#                                                                upAxisIndex)

#         width = self.pixelWidth
#         height = self.pixelHeight
#         projectionMatrix = p_world.computeProjectionMatrixFOV(
#             # fov=86.0,               # 摄像头的视线夹角
#             # aspect=16.0/9,          # width / height
#             # nearVal=0.3,            # 摄像头焦距下限
#             # farVal=3                # 摄像头能看上限
#             fov = 69.40,               # 摄像头的视线夹角
#             aspect = width/height,          # width / height
#             nearVal = 0.01,            # 摄像头焦距下限
#             farVal = 10.0                # 摄像头能看上限
#         )
#         width, height, rgbImg, depthImg, segImg = p_world.getCameraImage(
#             width=width, height=height,
#             viewMatrix=viewMatrix,
#             projectionMatrix=projectionMatrix
#             # flags=p.ER_NO_SEGMENTATION_MASK
#         )
#         return width, height, rgbImg, depthImg, segImg
#     def setCameraPicAndGetPic2(self, p_world=0):
        
#         # viewMatrix = [1.0, 0.0, -0.0, 0.0, -0.0, 0.1736481785774231, -0.9848078489303589, 0.0, 0.0, 0.9848078489303589, 0.1736481785774231, 0.0, -0.0, -5.960464477539063e-08, -4.0, 1.0]
#         camTargetPos = [-0.3, -0, 0.0]
#         upAxisIndex = 2
#         camDistance = 1
        
#         pixelWidth = self.pixelWidth
#         pixelHeight = self.pixelHeight
#         pixelWidth = 320
#         pixelHeight = 220
#         camTargetPos = [0, 0, 0]
#         camDistance = 4
#         yaw = 0.0
#         pitch = -10
#         roll = 0
#         upAxisIndex = 2
#         viewMatrix = p_world.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll,
#                                                         upAxisIndex)
#         projectionMatrix = [
#             1.0825318098068237, 0.0, 0.0, 0.0,
#             0.0, 1.732050895690918, 0.0, 0.0,
#             0.0, 0.0, -1.0002000331878662, -1.0,
#             0.0, 0.0, -0.020002000033855438, 0.0
#         ]
#         # projectionMatrix = [908.8558959960938, 0.0, 626.7174072265625, 0.0,
#         #                     0.0, 906.6900634765625, 383.2095947265625, 0.0,
#         #                     0.0, 0.0, 1.0, 0.0,
#         #                     0.0, 0.0, -0.020002000033855438, 0.0]
#         print(viewMatrix)
#         print(projectionMatrix)
#         width, height, rgbImg, depthImg, segImg = p_world.getCameraImage(pixelWidth,
#                                                                          pixelWidth,
#                                                                          viewMatrix=viewMatrix,
#                                                                          projectionMatrix=projectionMatrix,
#                                                                          shadow=1,
#                                                                          lightDirection=[1, 1, 1]
#                                                                          )
#         return width, height, rgbImg, depthImg, segImg

# def quaternion_correction(quaternion): # x,y,z,w
#     new_quat = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]) # w,x,y,z
#     cos_theta_over_2 = new_quat.w
#     sin_theta_over_2 = math.sqrt(new_quat.x ** 2 + new_quat.y ** 2 + new_quat.z ** 2)
#     theta_over_2 = math.atan2(sin_theta_over_2,cos_theta_over_2)
#     theta = theta_over_2 * 2.0
#     while theta >= math.pi:
#         theta = theta - 2.0*math.pi
#     while theta <= -math.pi:
#         theta = theta + 2.0*math.pi
#     new_quaternion = [math.sin(theta/2.0)*(new_quat.x/sin_theta_over_2), math.sin(theta/2.0)*(new_quat.y/sin_theta_over_2), math.sin(theta/2.0)*(new_quat.z/sin_theta_over_2), math.cos(theta/2.0)]
#     #if theta >= math.pi or theta <= -math.pi:
#     #    new_quaternion = [-quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
#     #    return new_quaternion
#     #return quaternion # x,y,z,w
#     return new_quaternion

# def depthImageRealTransfer(self, depth_image_real):
#     cv_image = bridge.imgmsg_to_cv2(depth_image_real,"16UC1")
#     if DEBUG_DEPTH_IMG_FLAG == True:
#         real_depth_img_name = str(_particle_update_time) + "_real_depth_img.png"
#         cv2.imwrite(os.path.expanduser("~/catkin_ws/src/PBPF/scripts/img_debug/")+real_depth_img_name, (cv_image).astype(np.uint16))


# if __name__ == '__main__':
#     run_camera_flag = True
#     loop_flag = 0
#     exp_world = ExpWorld()
#     p_real_world, real_world_cracker_pos, real_world_cracker_ori = exp_world.real_world_scene()
#     width = 1280
#     height = 720
#     launch_camera = LaunchCamera(width, height)
#     width, height, rgbImg, depthImg, segImg = launch_camera.setCameraPicAndGetPic(p_real_world)
#     fig, axs = plt.subplots(1,3)
#     axs[0].imshow(depthImg, cmap="gray")
#     axs[0].set_title('Depth image')
#     axs[1].imshow(rgbImg, cmap="gray")
#     axs[1].set_title('Rgb image')
#     axs[2].imshow(segImg, cmap="gray")
#     axs[2].set_title('Segmentation image')
#     # plt.imshow(depthImg, cmap="gray")
#     plt.plot(label='ax2')
#     # plt.legend()
#     plt.show()
#     while True:
#         width, height, rgbImg, depthImg, segImg = launch_camera.setCameraPicAndGetPic(p_real_world)
#         # width, height, rgbImg, depthImg, segImg = launch_camera.setCameraPicAndGetPic2(p_real_world)
#         p_real_world.stepSimulation()
#         # print(depthImg)
#         time.sleep(1/240)        


import os
import sys
import yaml
from rosbag.bag import Bag
import cv2

import roslib;   #roslib.load_manifest(PKG)
import rosbag
import rospy
import cv2
import numpy as np
import argparse
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

class ImageCreator():
    def __init__(self, bagfile, rgbpath, depthpath, rgbstamp, depthstamp):
        self.bridge = CvBridge()
        with rosbag.Bag(bagfile, 'r') as bag:
            for topic,msg,t in bag.read_messages():
                print(topic)
                if topic == "/camera/color/image_raw": #图像的topic；
                    # try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
                    # except CvBridgeError as e:
                    #     print(e)
                    timestr = "%.6f" %  msg.header.stamp.to_sec()
                    #%.6f表示小数点后带有6位，可根据精确度需要修改；
                    image_name = timestr+ ".png" #图像命名：时间戳.png
                    # cv2.imshow("color", cv_image)
                    cv2.waitKey(1);
                    cv2.imwrite(rgbpath + image_name, cv_image)  #保存；

                    # #写入时间戳
                    # with open(rgbstamp, 'a') as rgb_time_file:
                    #     rgb_time_file.write(timestr+" rgb/"+image_name+"\n")
                elif topic == "/camera/aligned_depth_to_color/image_raw": #图像的topic；
                    # try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg,"16UC1")
                    # except CvBridgeError as e:
                    #     print(e)
                    timestr = "%.6f" %  msg.header.stamp.to_sec()
                    #%.6f表示小数点后带有6位，可根据精确度需要修改；
                    image_name = timestr+ ".png" #图像命名：时间戳.png
                    
                    # cv2.imwrite(depthpath + image_name, cv_image)  #保存；
                    cv2.imwrite(depthpath + image_name, (cv_image).astype(np.uint16))

                    # #写入时间戳
                    # with open(depthstamp, 'a') as depth_time_file:
                    #     depth_time_file.write(timestr+" depth/"+image_name+"\n")

if __name__ == '__main__':
    ImageCreator('cheezit_new_2.bag', "/home/sc19zx/rgb/", "/home/sc19zx/depth/", 1, 1)