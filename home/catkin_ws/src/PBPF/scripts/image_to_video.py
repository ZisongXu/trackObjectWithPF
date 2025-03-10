import cv2
import numpy as np
import os
 
def images_to_video(image_folder, output_video, frame_rate):
    # 获取所有图像文件名
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # 排序文件名以确保正确顺序
 
    # 读取第一张图像以获取帧的尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
 
    # 定义视频编码和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    
    count = 0
    for image in images:
        count = count + 1
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)
        print("Conversion in progress. Processing to the "+str(count)+" image.")
 
    video.release()
 
# 示例使用
image_folder = 'track_vis'  # 替换为图像文件夹的路径
output_video = 'track_vis.mp4'      # 输出视频文件名
frame_rate = 30                        # 帧率
 
images_to_video(image_folder, output_video, frame_rate)
