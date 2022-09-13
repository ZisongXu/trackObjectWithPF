# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:09:51 2022

@author: 12106
"""
from __future__ import print_function

import cv2
import message_filters
import numpy as np
import resource_retriever
import rospy
import tf.transformations
from PIL import Image
from PIL import ImageDraw
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image as ImageSensor_msg
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import tf
import tf.transformations as transformations


class Ros_listener():
    def __init__(self):
        # Start ROS publishers
#        self.pub_rgb_dope_points = \
#            rospy.Publisher(
#                rospy.get_param('~topic_publishing') + "/rgb_points",
#                ImageSensor_msg,
#                queue_size=10
#            )

        self.cv_bridge = CvBridge()
        image_sub = message_filters.Subscriber('/camera/color/image_raw', ImageSensor_msg)
        
        ts = message_filters.TimeSynchronizer([image_sub], 1)
        ts.registerCallback(self.image_callback)
        rospy.spin()
    
    def image_callback(self, image_msg):
        
        while True:
            try:
                (trans,rot) = listener.lookupTransform('/RealSense', '/cracker', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        camera_T_obj_dope_pos = list(trans)
        camera_T_obj_dope_ori = list(rot)
        camera_T_obj_dope_3_3 = transformations.quaternion_matrix(camera_T_obj_dope_ori)
        print("camera_T_obj_dope_3_3:",camera_T_obj_dope_3_3)
        while True:
            try:
                (trans,rot) = listener.lookupTransform('/panda_link0', '/cracker', rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        rob_T_obj_dope_pos = list(trans)
        results_cv2 = cv2.projectPoints(np.array(rob_T_obj_dope_pos),
                                        np.array([0,0,0]),
                                        np.array([0,0,0]),
                                        np.array(_camera_intrinsic_matrix),
                                        np.array(_dist_coeffs))
        print("results_cv2[0]:", results_cv2[0])
        img = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
        height, width, _ = img.shape
        img_copy = img.copy()
        
        im = Image.fromarray(img_copy)
        im = im.convert('RGB')
        draw = Draw(im)
        
        points = [(270.73113541500015, 171.23152323217352), (241.46136527990785, 86.12035538275252), (355.92305269234504, 104.18411471846555), (393.4423106422681, 179.7875187171499), (252.5544934396236, 198.2046834515927), (226.773595371207, 114.98036848759719), (337.3676219356163, 128.83336788753047), (371.01054493868116, 202.87435265364252), (308.9311599655143, 146.9263616063306)]

        draw.draw_cube(points, (255, 0, 0))

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

#draw = Draw()
#draw.draw_cube(points2d, self.draw_colors[m])

def rotation_4_4_to_transformation_4_4(rotation_4_4,pos):
    rotation_4_4[0][3] = pos[0]
    rotation_4_4[1][3] = pos[1]
    rotation_4_4[2][3] = pos[2]
    return rotation_4_4
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
_camera_intrinsic_matrix = [[504.91994222,          0.0, 259.28746541],
                            [         0.0, 503.71668498, 212.89422353],
                            [         0.0,          0.0,          1.0]]
_dist_coeffs = [[0.], [0.], [0.], [0.]]
if __name__ == "__main__":
    # main()
    rospy.init_node('draw_box')
    listener = tf.TransformListener()
    
    ros_listen = Ros_listener()
