import pybullet
import numpy as np
from matplotlib.pyplot import imsave
import os
class SimulatedCamera:
    def __init__(self, client_id, camera_in_robot_transform, pixel_width, pixel_height, near_val, far_val):
        self.client_id = client_id
        self.camera_in_robot_transform = camera_in_robot_transform
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.near_val = near_val
        self.far_val = far_val

    def get_camera_pose(self, robot_id):
        robot_pos, robot_orn = pybullet.getBasePositionAndOrientation(robot_id, physicsClientId=self.client_id)
        
        # Convert robot orientation (quaternion) to transformation matrix
        robot_transform = np.array(pybullet.getMatrixFromQuaternion(robot_orn)).reshape(3, 3)
        robot_transform = np.c_[robot_transform, robot_pos]  # Add position to create 3x4 matrix
        robot_transform = np.r_[robot_transform, [[0, 0, 0, 1]]]  # Convert to 4x4 homogeneous matrix
        
        # Calculate camera's world transformation matrix
        camera_transform = robot_transform @ self.camera_in_robot_transform
        
        return camera_transform

    def generate_depth_image(self, robot_id):
        camera_transform = self.get_camera_pose(robot_id)
        camera_position = camera_transform[:3, 3]
        camera_orientation = camera_transform[:3, :3]
        # Calculate camera target position based on its orientation
        camera_direction = camera_orientation @ np.array([0, 0, 1])  # Assuming Z-axis is the forward direction
        camera_target_position = camera_position + camera_direction
        
        # Calculate up vector for the camera (assuming Y-axis is down)
        camera_up_vector = camera_orientation @ np.array([0, -1, 0])
        print(camera_position)
        print(camera_target_position)
        print(camera_up_vector)
        view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=camera_target_position,
            cameraUpVector=camera_up_vector,
            physicsClientId=self.client_id
        )
        
        projection_matrix = pybullet.computeProjectionMatrixFOV(
            # fov=57.86,  # Field of view in degrees
            fov=50,  # Field of view in degrees
            aspect=float(self.pixel_width) / float(self.pixel_height),
            nearVal=self.near_val,
            farVal=self.far_val,
            physicsClientId=self.client_id
        )
        
        # Get depth image from PyBullet
        _, _, _, depth_img, _ = pybullet.getCameraImage(
            width=self.pixel_width,
            height=self.pixel_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            physicsClientId=self.client_id
        )
        imsave(os.path.expanduser("~/catkin_ws/src/PBPF/test.png"), depth_img, cmap='gray')
        # Convert depth image from PyBullet format to actual depth values
        depth_buffer = np.array(depth_img)
        depth = self.near_val + (self.far_val - self.near_val) * depth_buffer
        return depth
        
#[1.15092245 0.12411099 0.99269871]
#[0.27885813 0.04274432 0.51011889]
#[-0.48345446 -0.00989689  0.87531356]

#[1.1524622998573197, 0.12385074094019527, 0.47269871]
#[ 0.30413611  0.03528415 -0.0493153 ]
#[-0.52322255 -0.01082788  0.85212729]

#[[-0.07594714  0.48361591 -0.87197918  1.15092785]
# [ 0.99664881  0.01017067 -0.0811647   0.1241224 ]
# [-0.03038392 -0.87522124 -0.48276765  0.396666  ]
# [ 0.          0.          0.          1.        ]]

#[[-0.08193518  0.52384573 -0.84786336  1.15240988]
# [ 0.99600298  0.01268279 -0.08841502  0.12418143]
# [-0.03556256 -0.85171873 -0.52279107  0.39739042]
# [ 0.          0.          0.          1.        ]] 
