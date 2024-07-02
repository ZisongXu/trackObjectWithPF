# import rospy
# from cv_bridge import CvBridge, CvBridgeError
# import sensor_msgs.msg  
# import numpy as np
# from PIL import Image
# i = 0
# root="/home/sc19zx/"

# def convert_depth_image(ros_image_depth):
#     bridge = CvBridge()
#     global i
#     depth_image = bridge.imgmsg_to_cv2(ros_image_depth, desired_encoding="passthrough")
#     depth_array = np.array(depth_image, dtype=np.uint16)
#     im = Image.fromarray(depth_array)
#     im = im.convert("L")
#     idx = str(i).zfill(4)
#     im.save(root+"/depth/frame{index}.png".format(index = idx))
#     i += 1
#     print("depth_idx: ", i)

# def convert_rgb_image(ros_image_rgb):
#     bridge = CvBridge()
#     global i
#     rgb_image = bridge.imgmsg_to_cv2(ros_image_rgb, desired_encoding="passthrough")
#     rgb_array = np.array(rgb_image, dtype=np.uint8)
#     im = Image.fromarray(rgb_array)
#     im = im.convert("L")
#     idx = str(i).zfill(4)
#     im.save(root+"/rgb/frame{index}.png".format(index = idx))
#     i += 1
#     print("rgb_idx: ", i)

# def pixel2depth():
#     rospy.init_node('pixel2depth',anonymous=True)
#     # rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", sensor_msgs.msg.Image,callback=convert_depth_image, queue_size=10)
#     rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image,callback=convert_rgb_image, queue_size=10)

#     rospy.spin()

# if __name__ == '__main__':
#     pixel2depth()





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
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

class ImageCreator():
    def __init__(self, bagfile, rgbpath, depthpath, rgbstamp, depthstamp):
        self.bridge = CvBridge()
        with rosbag.Bag(bagfile, 'r') as bag:
            count_rgb = 0
            count_depth = 0
            for topic,msg,t in bag.read_messages():
                if topic == "/camera/color/image_raw": #图像的topic；
                    count_rgb = count_rgb + 1
                    # try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
                    # except CvBridgeError as e:
                    #     print(e)
                    timestr = "%.6f" %  msg.header.stamp.to_sec()
                    #%.6f表示小数点后带有6位，可根据精确度需要修改；
                    # image_name = timestr+ ".png" #图像命名：时间戳.png
                    # image_name = str(count_rgb)+ ".png" #图像命名：时间戳.png
                    image_name = f"{count_rgb:04d}.png"
                    # cv2.imshow("color", cv_image)
                    cv2.waitKey(1);
                    cv2.imwrite(rgbpath + image_name, cv_image)  #保存；

                    # #写入时间戳
                    # with open(rgbstamp, 'a') as rgb_time_file:
                    #     rgb_time_file.write(timestr+" rgb/"+image_name+"\n")
                elif topic == "/camera/aligned_depth_to_color/image_raw": #图像的topic；
                    count_depth = count_depth + 1
                    # try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg,"16UC1")
                    # except CvBridgeError as e:
                    #     print(e)
                    timestr = "%.6f" %  msg.header.stamp.to_sec()
                    #%.6f表示小数点后带有6位，可根据精确度需要修改；
                    # image_name = timestr+ ".png" #图像命名：时间戳.png
                    # image_name = str(count_depth)+ ".png" #图像命名：时间戳.png
                    image_name = f"{count_depth:04d}.png"
                    cv2.imwrite(depthpath + image_name, cv_image)  #保存；
                    # cv2.imwrite(depthpath + image_name, (cv_image).astype(np.uint16))

                    # #写入时间戳
                    # with open(depthstamp, 'a') as depth_time_file:
                    #     depth_time_file.write(timestr+" depth/"+image_name+"\n")
                print("count_rgb:", count_rgb)
                print("count_depth:", count_depth)


rosbag_file_path = os.path.expanduser("~/pyvkdepth/rosbag/")
rosbag_file_name = "2_scene1_ParmesanKetchup1.bag"
save_file_path = os.path.expanduser("~/")
if __name__ == '__main__':
    ImageCreator(rosbag_file_path+rosbag_file_name, save_file_path+"rgb/", save_file_path+"depth/", 1, 1)









    # #rospy.init_node(PKG)
    # parser = argparse.ArgumentParser(description="Grab the rgb and depth images from a ros bag")
    # # parser.add_argument("--verbose", "-v", action='store_true', help='verbose_mode')
    # help = "The bag file"
    # parser.add_argument('bag', help=help)
    # help="The output folder"
    # print("parser:", parser)
    # parser.add_argument('output_folder', help=help)
    # print("parser:", parser)
    # args = parser.parse_args()
    # print("parser:", parser)
    # bagfile = args.bag
    # rgb_path = args.output_folder + '/rgb/'
    # depth_path = args.output_folder + '/depth/'

    # rgb_timestamp_txt =  args.output_folder + "/rgb.txt"
    # depth_timestamp_txt = args.output_folder + "/depth.txt"

    # if not os.path.exists(rgb_path):
    #     os.makedirs(rgb_path)
    # if not os.path.exists(depth_path):
    #     os.makedirs(depth_path)
    
    # try:
    #     image_creator = ImageCreator(bagfile, rgb_path, depth_path, rgb_timestamp_txt, depth_timestamp_txt)
    # except rospy.ROSInterruptException:
    #     pass
