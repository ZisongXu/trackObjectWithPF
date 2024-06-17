import pybullet 
import time
import pybullet_data
from pybullet_utils import bullet_client
import numpy as np
import os
import matplotlib  
from matplotlib import pyplot as plt
from simulated_camera import SimulatedCamera
import math
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
from jax import jit

# ===================================================================================
# # 定义一个函数来找到位置

# def find_positions(matrix, targets):
#     # 使用广播和比较来创建一个布尔数组，其中True代表匹配的元素
#     match_positions = matrix == targets[:, None, None]
#     # 获取匹配位置的索引
#     all_positions = jnp.argwhere(match_positions)
#     positions = all_positions[:, 1:]
    
#     # 返回每个目标的第一个匹配位置（如果有多个匹配）
#     # 注意：这里假设每个目标在matrix中至少出现一次
#     return positions
# seg_id = jnp.array([[1,0,0,2,2,2,0,0,0],
#                     [1,0,0,2,2,2,0,0,0],
#                     [1,1,0,0,2,2,2,0,0],
#                     [1,1,1,0,0,2,2,0,0]])
# object_id = jnp.array([2])
# positions = find_positions(seg_id, object_id)

# print("positions:")
# print(positions)

# # 获取横坐标的最大值、最小值
# x_min = jnp.min(positions[:, 0])
# x_max = jnp.max(positions[:, 0])
 
# # 获取纵坐标的最大值、最小值
# y_min = jnp.min(positions[:, 1])
# y_max = jnp.max(positions[:, 1])

# # print("x_min:", x_min, "; x_max:", x_max)
# # print("y_min:", y_min, "; y_max:", y_max)

# import jax.numpy as jnp
 
# # 定义三维数组，注意这里使用列表来处理不均匀的结构
# jax_1 = jnp.array([[0, 1], [0, 2]])
# jax_2 = jnp.array([[0, 1], [2, 1], [3, 2]])
# jax_3 = jnp.array([[0, 3], [2, 1], [0, 4], [0, 2]])
# arr = [1] * 3
# arr[0] = jax_1
# arr[1] = jax_2
# arr[2] = jax_3

# # 将所有坐标平铺成一个列表
# flat_list = [item for sublist in arr for item in sublist]
 
# # 转换成JAX数组
# flat_arr = jnp.array(flat_list)
 
# # 计算x坐标和y坐标的最大值和最小值
# x_min = flat_arr[:, 0].min()
# x_max = flat_arr[:, 0].max()
# y_min = flat_arr[:, 1].min()
# y_max = flat_arr[:, 1].max()
 
# print(f"x坐标的最小值: {x_min}, 最大值: {x_max}")
# print(f"y坐标的最小值: {y_min}, 最大值: {y_max}")



# # 示例二维数组和一维目标数组
# real_depth_image = jnp.array([[5,2,4,2,4,1,3,9,4],
#                               [3,4,0,2,8,8,0,9,5],
#                               [6,4,0,7,3,0,0,2,7],
#                               [2,7,1,6,9,1,6,2,4]])
 
# rendered_depth_image = jnp.array([[1,4,5,2,2,2,6,4,2],
#                                   [1,1,0,2,4,6,0,9,5],
#                                   [2,1,0,7,2,2,2,2,7],
#                                   [1,3,1,6,0,2,4,2,4]])


# rectangular_region_data_real = real_depth_image[y_min:y_max+1, x_min:x_max+1]
# rectangular_region_data_real = real_depth_image[x_min:x_max+1, y_min:y_max+1]
# rectangular_region_data_render = rendered_depth_image[x_min:x_max+1, y_min:y_max+1]
# print("rectangular_region_data_real:", rectangular_region_data_real)
# print("rectangular_region_data_render:", rectangular_region_data_render)


# import jax.numpy as jnp
 
# # 创建一个二维数组
# arr_2d = jnp.array([[1, 2, 3], [4, 5, 6]])
 
# # 使用ravel函数将其变成一维数组
# arr_1d = arr_2d.ravel()
 
# print(arr_1d)

# ===================================================================================


def replace_values(a, b):
    # 创建一个与a形状相同、全为0的矩阵
    result = jnp.zeros_like(a)
    # 提取b中所有坐标的x和y值
    x_coords, y_coords = b[:, 0], b[:, 1]
    # 使用JAX的高级索引批量设置值
    result = result.at[x_coords, y_coords].set(7)
    return result
 
# 示例矩阵a和b
a = jnp.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
 
b = jnp.array([[0, 1],  # (x=0, y=1)坐标
               [2, 2]]) # (x=2, y=2)坐标
 
# 调用函数并打印结果
result = replace_values(a, b)
print(result)


# @jit
# def extract_values(matrix, positions):     
#     # 提取对应位置的数值    
#     values = matrix[positions[:, 0], positions[:, 1]]     
#     return values 

# @jit
# def _threshold_array_optimized(arr):
#     modified_array = (arr > DEPTH_DIFF_VALUE_0_1_THRESHOLD).astype(jnp.int32)
#     num_zeros = jnp.sum(modified_array == 0)
#     return modified_array, num_zeros

# # 使用函数提取数值
# extracted_values_real = extract_values(real_depth_image, positions) 
# extracted_values_render = extract_values(rendered_depth_image, positions) 

# number_of_pixels = len(extracted_values_render)

# print("Extracted Real Depth Values:", extracted_values_real)
# print("Extracted Rendered Depth Values:", extracted_values_render)

# depth_value_diff_sub_abs_jax = jnp.abs(extracted_values_real - extracted_values_render)
# print("depth_value_diff_sub_abs_jax:", depth_value_diff_sub_abs_jax)
# DEPTH_DIFF_VALUE_0_1_THRESHOLD = 0.1
# DEPTH_DIFF_VALUE_0_1_ALPHA = 0.8
# depth_value_diff_sub_abs_0_1_jax, num_zeros = _threshold_array_optimized(depth_value_diff_sub_abs_jax)
# print("depth_value_diff_sub_abs_0_1_jax:", depth_value_diff_sub_abs_0_1_jax)
# print("num_zeros:", num_zeros)
# print("number_of_pixels:", number_of_pixels)
# e_VSD_o = jnp.sum(depth_value_diff_sub_abs_0_1_jax) / number_of_pixels
# print("e_VSD_o:", e_VSD_o)

# score = DEPTH_DIFF_VALUE_0_1_ALPHA * (1-e_VSD_o) + (1-DEPTH_DIFF_VALUE_0_1_ALPHA) * num_zeros/number_of_pixels
# depth_value_difference_jax = score
# print("score:", depth_value_difference_jax)
# depth_value_difference = float(depth_value_difference_jax.item())
# print("depth_value_difference:", type(depth_value_difference))
# if score != "nan":
#     print("True")

# test_weight = 0.5 * depth_value_difference
# print("test_weight:",test_weight)
# ======================================================================================================



# # 定义两个一维数组
# array1 = jnp.array([5, 6, 8])
# array2 = jnp.array([4, 5, 6])
 
# # 计算欧几里得距离
# distance = jnp.linalg.norm(array1 - array2)
 
# print("Euclidean Distance:", distance)

# a = math.sqrt((((5-4) ** 2)+((6-5) ** 2)+((8-6) ** 2)))
# print(a)
# print(a / math.sqrt(3))
# ======================================================================================================


# import jax.numpy as jnp
# from jax import jit
 
# # 定义从正交投影到透视投影的转换
# @jit
# def orthographic_to_perspective(depth_orthographic, f):
#     return f * (depth_orthographic / (f - depth_orthographic))
 
# # 定义从透视投影到正交投影的转换
# @jit
# def perspective_to_orthographic(depth_perspective, f):
#     return f * (depth_perspective / (f + depth_perspective))
 
# # 示例深度值
# depth_orthographic = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# depth_perspective = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
 
# # 假设焦距 f = 1000
# f = 1000
 
# # 正交到透视
# depth_perspective_converted = orthographic_to_perspective(depth_orthographic, f)
 
# # 透视到正交
# depth_orthographic_converted = perspective_to_orthographic(depth_perspective, f)
 
# print("从正交到透视转换的深度值:\n", depth_perspective_converted)
# print("从透视到正交转换的深度值:\n", depth_orthographic_converted)

# =================================================================================================

# import jax.numpy as jnp
# from jax import jit
 
# def perspective_to_orthographic(depth_perspective, fov, image_resolution):
#     """
#     从透视投影深度图转换到正交投影深度图。
#     """
#     h, w = image_resolution
#     f = 1 / jnp.tan(fov / 2)  # 焦距
#     aspect_ratio = w / h
#     # 构造透视矩阵
#     perspective_matrix = jnp.array([[f/aspect_ratio, 0, 0, 0],
#                                     [0, f, 0, 0],
#                                     [0, 0, 1, 1],
#                                     [0, 0, -1, 0]])
#     # 逆矩阵转换
#     orthographic_depth = depth_perspective * perspective_matrix[2, 2] + perspective_matrix[2, 3]
#     orthographic_depth = orthographic_depth / (depth_perspective + perspective_matrix[3, 2])
#     return orthographic_depth
 
# def orthographic_to_perspective(depth_orthographic, fov, image_resolution):
#     """
#     从正交投影深度图转换到透视投影深度图。
#     """
#     h, w = image_resolution
#     f = 1 / jnp.tan(fov / 2)  # 焦距
#     aspect_ratio = w / h
#     # 透视校正因子，基于像素位置
#     x, y = jnp.meshgrid(jnp.linspace(-1, 1, w), jnp.linspace(-1, 1, h))
#     z = depth_orthographic
#     perspective_depth = z / (f * jnp.sqrt(x**2 + y**2 / aspect_ratio**2 + 1))
#     return perspective_depth
 
# # 示例参数
# fov = 90.0 * (jnp.pi / 180)  # FOV，转换为弧度
# image_resolution = (480, 640)  # 图像分辨率
 
# # 假设的深度图
# depth_orthographic = jnp.ones(image_resolution)  # 正交投影深度图
# depth_perspective = jnp.ones(image_resolution)  # 透视投影深度图
 
# # 转换
# depth_perspective_converted = orthographic_to_perspective(depth_orthographic, fov, image_resolution)
# depth_orthographic_converted = perspective_to_orthographic(depth_perspective, fov, image_resolution)
# print("depth_orthographic:")
# print(depth_orthographic)
# print("depth_perspective_converted")
# print(depth_perspective_converted)

# ==========================================================================================================

# from jax import random
# import jax.numpy as jnp
# from jax import jit

# @jit
# def compute_projection_parameters(fov, resolution):
#     """计算透视投影所需的参数"""
#     h, w = resolution
#     # f = 0.5 * w / jnp.tan(fov * 0.5)  # 假设fov是水平的
#     f = 0.5 * h / jnp.tan(fov * 0.5)  # 假设fov是水平的
#     cx, cy = w * 0.5, h * 0.5
#     return f, cx, cy
 
# @jit
# def ortho_to_persp(depth_ortho, fov, resolution):
#     """正交投影深度图转换为透视投影深度图"""
#     f, cx, cy = compute_projection_parameters(fov, resolution)
#     y, x = jnp.indices(depth_ortho.shape)
#     z = depth_ortho
#     x_persp = (x - cx) * z / f
#     y_persp = (y - cy) * z / f
#     depth_persp = jnp.sqrt(x_persp**2 + y_persp**2 + z**2)
#     return depth_persp
 
# @jit
# def persp_to_ortho(depth_persp, fov, resolution):     
#     """透视投影深度图转换为正交投影深度图"""    
#     f, cx, cy = compute_projection_parameters(fov, resolution)     
#     y, x = jnp.indices(depth_persp.shape)     
#     # 逆向透视效果调整深度值    
#     z = depth_persp     
#     # 假设所有点在深度图中直接面向相机，计算透视图中的实际深度    
#     depth_ortho = z / jnp.sqrt(((x - cx) / f)**2 + ((y - cy) / f)**2 + 1)     
#     return depth_ortho
 
# # 创建随机键
# key = random.PRNGKey(0)
 
# # 示例参数
# fov = jnp.radians(58.0)  # 90度的视场
# resolution = (555, 555)  # 深度图的分辨率
# print(type(resolution))
# # 使用JAX的随机数生成函数来创建假设的深度图
# depth_ortho = random.uniform(key, resolution)
# depth_persp = random.uniform(key, resolution)
 
# # 执行转换
# depth_persp_converted = ortho_to_persp(depth_ortho, fov, resolution)
# depth_ortho_converted = persp_to_ortho(depth_persp, fov, resolution)

# print("depth_ortho[277][277]:")
# print(depth_ortho[277][277])
# print("depth_ortho:")
# print(depth_ortho)
# print("depth_persp_converted[277][277]:")
# print(depth_persp_converted[277][277])
# print("depth_persp_converted:")
# print(depth_persp_converted)
# print("depth_persp[277][277]:")
# print(depth_persp[277][277])
# print("depth_persp:")
# print(depth_persp)
# print("depth_ortho_converted[277][277]:")
# print(depth_ortho_converted[277][277])
# print("depth_ortho_converted:")
# print(depth_ortho_converted)

# ===============================================================================================================

# import jax.numpy as jnp

# @jit
# def threshold_array_optimized(arr):
#     # 直接将布尔数组转换为整数数组
#     return (arr < 0.5).astype(jnp.int32)
 
# # 创建一个示例一维数组
# arr = jnp.array([0.2, 0.6, 0.4, 0.8, 0.1])
 
# # 应用函数并打印结果
# result = threshold_array_optimized(arr)
# print(result)


# ===============================================================================================================

