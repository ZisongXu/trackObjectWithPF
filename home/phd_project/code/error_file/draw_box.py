# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""
from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
from PIL import Image
from PIL import ImageDraw
from cv_bridge import CvBridge

import cv2
import message_filters
import numpy as np
import resource_retriever
import rospy
import tf.transformations
from PIL import Image
from PIL import ImageDraw
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image as ImageSensor_msg

class Ros_listener():
    def __init__(self):
#        self.joint_subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_values_callback,queue_size=1)
#        self.robot_pose = rospy.Subscriber('/mocap/rigid_bodies/pandaRobot/pose',PoseStamped, self.robot_pose_callback,queue_size=10)
#        self.object_pose = rospy.Subscriber('/mocap/rigid_bodies/cheezit/pose',PoseStamped, self.object_pose_callback,queue_size=10)
#        self.base_pose = rospy.Subscriber('/mocap/rigid_bodies/baseofcheezit/pose', PoseStamped, self.base_of_cheezit_callback,queue_size=10)
        self.cv_bridge = CvBridge()
        image_sub = message_filters.Subscriber('/camera/color/image_raw',ImageSensor_msg)
#        image_sub = message_filters.Subscriber(
#            rospy.get_param('~topic_camera'),
#            ImageSensor_msg
#        )
#        image_sub=np.uint8(image_sub)
#        
#        ts = message_filters.ApproximateTimeSynchronizer([image_sub], 10, 1, allow_headerless=True)
        ts = message_filters.TimeSynchronizer([image_sub], 1)
        ts.registerCallback(self.image_callback)
        rospy.spin()
    def image_callback(self, image_msg):
        
        img = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
        cv2.imshow("image_sub:", img)
        height, width, _ = img.shape
        img_copy = img.copy()
        
        im = Image.fromarray(img_copy)
        im = im.convert('RGB')
        draw = Draw(im)
        draw.draw_cube(points)
        rgb_points_img = CvBridge().cv2_to_imgmsg(np.array(im)[..., ::-1], "bgr8")
        cv2.imshow("rgb_points_img:", rgb_points_img)
        cv2.waitKey(3)
        
class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)

    def draw_line(self, point1, point2, line_color, line_width=2):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            print(point1)
            print(point2)
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
        self.draw_line(tuple(points[0]), tuple(points[1]), color)
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
points = [[0,0,0], [0,0,1], [1,0,1], [1,0,0], [0,1,0], [0,1,1], [1,1,1], [1,1,0]]
#draw = Draw()
#draw.draw_cube(points2d, self.draw_colors[m])


def main():
    """Main routine to run DOPE"""

    # Initialize ROS node
    rospy.init_node('draw_box')
    ros_listen = Ros_listener()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
