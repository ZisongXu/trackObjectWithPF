#!/usr/bin/python3
# license removed for brevity
import rospy
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import numpy as np
import copy

import matplotlib  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def depth_image_callback(depth_image_data):
    global flag
    cv_image = bridge.imgmsg_to_cv2(depth_image_data,"16UC1")
    cv_image_y = copy.deepcopy(cv_image) / 1000.0
    print(type(cv_image))
    # print("len(cv_image):", len(cv_image))
    # print("len(cv_image[0]):", len(cv_image[0]))
    # print(cv_image)
    farVal = 10.0
    nearVal = 0.01
    # depth = farVal * nearVal / (farVal - (farVal - nearVal) * depth_image_render)
    cv_image_x = (farVal - (farVal * nearVal / cv_image_y)) / (farVal - nearVal)
    # print(cv_image_x)
    if flag == 0:
        flag = 1
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(cv_image_x, cmap="gray")
        axs[0].set_title('Depth image')
        axs[1].imshow(cv_image_x, cmap="gray")
        axs[1].set_title('Rgb image')
        axs[2].imshow(cv_image_x, cmap="gray")
        axs[2].set_title('Segmentation image')
        # plt.imshow(depthImg, cmap="gray")
        plt.plot(label='ax2')

    # print(cv_image[0])
    # print("Image Height:", depth_image_data.height)
    # print("Image Width:", depth_image_data.width)
    # print("Image Encoding:", depth_image_data.encoding)
    # print("Image Step:", depth_image_data.step)
    # print("Image Is_bigendian:", depth_image_data.is_bigendian)
    # print(cv_image)
    # print("Length of Image Data:", len(depth_image_data.data))
    return cv_image
    
def listener():
    global sub
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    
    sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_image_callback, queue_size=1)

    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()

if __name__ == '__main__':
    bridge = CvBridge()
    listener()
    flag = 0
    while not rospy.is_shutdown():
        pass
