

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
                    # image_name = f"{count_rgb:03d}.png"
                    # cv2.imshow("color", cv_image)
                    cv2.waitKey(1);
                    cv2.imwrite(rgbpath + image_name, cv_image)  #保存；
                    print("count_rgb:", count_rgb)
                    
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
                    # image_name = f"{count_depth:03d}.png"
                    cv2.imwrite(depthpath + image_name, cv_image)  #保存；
                    # cv2.imwrite(depthpath + image_name, (cv_image).astype(np.uint16))

                    # #写入时间戳
                    # with open(depthstamp, 'a') as depth_time_file:
                    #     depth_time_file.write(timestr+" depth/"+image_name+"\n")
                    print("count_depth:", count_depth)
                # if abs(count_depth - count_rgb) > 1:
                #     if count_depth > count_rgb:
                #         count_rgb = count_rgb + 1
                #         image_name = f"{count_rgb:04d}.png"
                #         cv2.imwrite(rgbpath + image_name, cv_image)  #保存；
                #     elif count_rgb > count_depth:
                #         count_depth = count_depth + 1
                #         image_name = f"{count_depth:04d}.png"
                #         cv2.imwrite(rgbpath + image_name, cv_image)  #保存；



rosbag_file_path = os.path.expanduser("~/pyvkdepth/rosbag/")
rosbag_file_name = "4_scene2_crackerParmesansoupMayo1_CP.bag"
save_file_path = os.path.expanduser("~/")
if __name__ == '__main__':
    ImageCreator(rosbag_file_path+rosbag_file_name, save_file_path+"rgb/", save_file_path+"depth/", 1, 1)




# import os
# import sys
# import yaml
# import cv2
# import rosbag
# import rospy
# import numpy as np
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
 
# class ImageCreator():
#     def __init__(self, bagfile, rgbpath, depthpath, rgbstamp, depthstamp):
#         self.bridge = CvBridge()
#         # 用于保存上一帧图像（用于复制填充缺失帧）
#         last_rgb_image = None
#         last_depth_image = None
 
#         # 初始化保存计数（图片的编号从1开始）
#         count_rgb = 0
#         count_depth = 0
 
#         with rosbag.Bag(bagfile, 'r') as bag:
#             for topic, msg, t in bag.read_messages():
#                 if topic == "/camera/color/image_raw":
#                     # 每收到一条 RGB 消息，则编号加1
#                     count_rgb += 1
#                     try:
#                         rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#                     except CvBridgeError as e:
#                         print(e)
#                         continue
#                     last_rgb_image = rgb_image  # 更新上一帧RGB
#                     image_name = f"{count_rgb:04d}.png"
#                     cv2.imwrite(os.path.join(rgbpath, image_name), rgb_image)
#                     print("Saved RGB image:", image_name)
 
#                     # 检查 RGB 与 Depth 的计数差是否超过1
#                     while (count_rgb - count_depth) > 1:
#                         # 表示缺失了一个 Depth 帧，使用上一帧 Depth 图像复制
#                         missing_depth_index = count_depth + 1
#                         if last_depth_image is not None:
#                             dup_name = f"{missing_depth_index:04d}.png"
#                             cv2.imwrite(os.path.join(depthpath, dup_name), last_depth_image)
#                             print("Filling missing Depth image:", dup_name)
#                         else:
#                             print("Warning: 无法复制 Depth 图像（上一帧不存在），缺失帧编号：", missing_depth_index)
#                         count_depth += 1
 
#                 elif topic == "/camera/aligned_depth_to_color/image_raw":
#                     # 每收到一条 Depth 消息，则编号加1
#                     count_depth += 1
#                     try:
#                         depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
#                     except CvBridgeError as e:
#                         print(e)
#                         continue
#                     last_depth_image = depth_image  # 更新上一帧Depth
#                     image_name = f"{count_depth:04d}.png"
#                     cv2.imwrite(os.path.join(depthpath, image_name), depth_image)
#                     print("Saved Depth image:", image_name)
 
#                     # 检查 Depth 与 RGB 的计数差是否超过1
#                     while (count_depth - count_rgb) > 1:
#                         missing_rgb_index = count_rgb + 1
#                         if last_rgb_image is not None:
#                             dup_name = f"{missing_rgb_index:04d}.png"
#                             cv2.imwrite(os.path.join(rgbpath, dup_name), last_rgb_image)
#                             print("Filling missing RGB image:", dup_name)
#                         else:
#                             print("Warning: 无法复制 RGB 图像（上一帧不存在），缺失帧编号：", missing_rgb_index)
#                         count_rgb += 1
 
# if __name__ == '__main__':
#     # 注意修改rosbag文件路径和保存路径
#     rosbag_file_path = os.path.expanduser("~/pyvkdepth/rosbag/")
#     rosbag_file_name = "4_scene2_crackerParmesansoupMayo1_CP.bag"
#     save_file_path = os.path.expanduser("~/")
#     ImageCreator(os.path.join(rosbag_file_path, rosbag_file_name),
#                  os.path.join(save_file_path, "rgb_t/"),
#                  os.path.join(save_file_path, "depth_t/"),
#                  1, 1)



# import os
# import cv2
# import rosbag
# import numpy as np
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
 
# class ImageCreator():
#     def __init__(self, bagfile, rgbpath, depthpath):
#         self.bridge = CvBridge()
#         expected_frame = 1  # 当前帧编号，从1开始
#         last_rgb = None     # 保存上一帧的 RGB，用于填充缺失
#         last_depth = None   # 保存上一帧的 Depth，用于填充缺失
#         current_depth = None  # 当前帧收到的 Depth 消息
#         current_rgb = None    # 当前帧收到的 RGB 消息
 
#         with rosbag.Bag(bagfile, 'r') as bag:
#             for topic, msg, t in bag.read_messages():
#                 # 如果先收到深度消息
#                 if topic == "/camera/aligned_depth_to_color/image_raw":
#                     try:
#                         depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
#                     except CvBridgeError as e:
#                         print("CvBridgeError for depth:", e)
#                         continue
 
#                     if current_depth is not None:
#                         # 说明上一帧已经有深度消息，但对应的 RGB 消息未收到
#                         # 此时认为上一帧缺失 RGB，用上一帧保存的 RGB（若存在）来填充
#                         if last_rgb is not None:
#                             missing_rgb = last_rgb.copy()
#                         else:
#                             # 若没有上一帧 RGB，则构造一张全黑图（尺寸参考当前深度图对应尺寸）
#                             missing_rgb = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
#                         rgb_filename = os.path.join(rgbpath, f"{expected_frame:04d}.png")
#                         depth_filename = os.path.join(depthpath, f"{expected_frame:04d}.png")
#                         cv2.imwrite(rgb_filename, missing_rgb)
#                         cv2.imwrite(depth_filename, current_depth)
#                         print(f"Frame {expected_frame:04d}: Missing RGB filled")
#                         # 更新上一帧备份，帧编号自增
#                         last_rgb = missing_rgb
#                         last_depth = current_depth
#                         expected_frame += 1
#                         # 将当前的深度消息作为新一帧的起始
#                         current_depth = depth_img
#                     else:
#                         # 如果当前帧还没有收到任何深度消息，则直接保存
#                         current_depth = depth_img
 
#                 # 当收到 RGB 消息时
#                 elif topic == "/camera/color/image_raw":
#                     try:
#                         rgb_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#                     except CvBridgeError as e:
#                         print("CvBridgeError for rgb:", e)
#                         continue
 
#                     if current_depth is None:
#                         # 如果当前帧没有收到对应的深度图，
#                         # 则使用上一帧的 depth 填充（如果存在），否则用全零图
#                         if last_depth is not None:
#                             current_depth = last_depth.copy()
#                             print(f"Frame {expected_frame:04d}: Missing Depth filled from last frame")
#                         else:
#                             current_depth = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint16)
#                             print(f"Frame {expected_frame:04d}: Missing Depth filled with zeros")
#                     current_rgb = rgb_img
 
#                     # 保存当前帧完整配对的数据
#                     rgb_filename = os.path.join(rgbpath, f"{expected_frame:04d}.png")
#                     depth_filename = os.path.join(depthpath, f"{expected_frame:04d}.png")
#                     cv2.imwrite(rgb_filename, current_rgb)
#                     cv2.imwrite(depth_filename, current_depth)
#                     print(f"Frame {expected_frame:04d}: Saved RGB and Depth")
#                     # 更新上一帧备份
#                     last_rgb = current_rgb
#                     last_depth = current_depth
#                     expected_frame += 1
#                     # 清空当前帧数据，等待新一帧
#                     current_depth = None
#                     current_rgb = None
 
#             # 遍历结束后，如果最后一帧不完整，则补全后保存
#             if current_depth is not None and current_rgb is None:
#                 # 最后收到的是深度，但缺少 rgb
#                 if last_rgb is not None:
#                     missing_rgb = last_rgb.copy()
#                 else:
#                     missing_rgb = np.zeros((current_depth.shape[0], current_depth.shape[1], 3), dtype=np.uint8)
#                 rgb_filename = os.path.join(rgbpath, f"{expected_frame:04d}.png")
#                 depth_filename = os.path.join(depthpath, f"{expected_frame:04d}.png")
#                 cv2.imwrite(rgb_filename, missing_rgb)
#                 cv2.imwrite(depth_filename, current_depth)
#                 print(f"Final Frame {expected_frame:04d}: Missing RGB filled")
#             elif current_rgb is not None and current_depth is None:
#                 # 最后收到的是 rgb，但缺少 depth
#                 if last_depth is not None:
#                     missing_depth = last_depth.copy()
#                 else:
#                     missing_depth = np.zeros((current_rgb.shape[0], current_rgb.shape[1]), dtype=np.uint16)
#                 rgb_filename = os.path.join(rgbpath, f"{expected_frame:04d}.png")
#                 depth_filename = os.path.join(depthpath, f"{expected_frame:04d}.png")
#                 cv2.imwrite(rgb_filename, current_rgb)
#                 cv2.imwrite(depth_filename, missing_depth)
#                 print(f"Final Frame {expected_frame:04d}: Missing Depth filled")
 
# if __name__ == '__main__':
#     # 注意根据实际情况修改路径
#     rosbag_file_path = os.path.expanduser("~/pyvkdepth/rosbag/")
#     rosbag_file_name = "2_soups.bag"
#     save_file_path = os.path.expanduser("~/")
#     ImageCreator(os.path.join(rosbag_file_path, rosbag_file_name),
#                  os.path.join(save_file_path, "rgb/"),
#                  os.path.join(save_file_path, "depth/"))