import tf.transformations as transformations
import numpy as np
from pyquaternion import Quaternion #w,x,y,z
pos = [1.3, 0.89, -2.985]
q = [0.06146124,0.234234,-0.1341234,0.99810947]
#q = [0.99810947,0.06146124,0.234234,-0.1341234]
#q = [0,0,0,1]
#q = [0,0,1,0] #x,y,z,w
print("q:")

def rotation_4_4_to_transformation_4_4(rotation_4_4,pos):
    rotation_4_4[0][3] = pos[0]
    rotation_4_4[1][3] = pos[1]
    rotation_4_4[2][3] = pos[2]
    return rotation_4_4
def transformation_4_4_to_rotation_4_4(rotation_4_4):
    rotation_4_4[0][3] = 0.0
    rotation_4_4[1][3] = 0.0
    rotation_4_4[2][3] = 0.0
    return rotation_4_4
r = transformations.quaternion_matrix(q) #[x,y,z,w]
#r = transformations.quaternion_matrix([0,0,0,1])
print(r)
T = rotation_4_4_to_transformation_4_4(r,pos)
print(T)
T1= transformation_4_4_to_rotation_4_4(T)
print(T1)
print(q[3],q[0],q[1],q[2]) #w,x,y,z







transformation = r
q8d_t = Quaternion(matrix=transformation)
print("q8d_t:")
print(q8d_t)



#R = transformations.rotation_matrix(0.123, (1, 2, 3))
#print(type(R))
#print(R)
q = transformations.quaternion_from_matrix(T)
print(q)
q1 = transformations.quaternion_from_matrix(T1)
print(q1)


'''
init_robot_pos = [1,2,3]
init_robot_ori = [0,0,1,0] #x,y,z,w
init_object_pos = [2,3,4]
init_object_ori = [0,0,0,1]

robot_transformation_matrix = transformations.quaternion_matrix(init_robot_ori)
ow_T_robot = rotation_4_4_to_transformation_4_4(robot_transformation_matrix,init_robot_pos)
print("ow_T_robot:",ow_T_robot)
object_transformation_matrix = transformations.quaternion_matrix(init_object_ori)
ow_T_object = rotation_4_4_to_transformation_4_4(object_transformation_matrix,init_object_pos)
print("ow_T_object:",ow_T_object)
robot_T_ow = np.linalg.inv(ow_T_robot)
print("robot_T_ow:",robot_T_ow)
robot_T_object = np.dot(robot_T_ow,ow_T_object)
print("robot_T_object:",robot_T_object)
robot_T_object_pos = [robot_T_object[0][3],
                      robot_T_object[1][3],
                      robot_T_object[2][3]]
print(robot_T_object_pos)
'''









'''
rotation = np.eye(3)
#rotation = np.linalg.inv(rotation)
print("rotation:")
print(rotation)
q8d_r = Quaternion(matrix=rotation)
print("q8d_r:")
print(q8d_r)



def quaternion_rotation_matrix(Q):
    # [w,x,y,z]
    q0=Q[0]
    q1=Q[1]
    q2=Q[2]
    q3=Q[3]
    
    r00 = 2 * (q0*q0+q1*q1) - 1
    r01 = 2 * (q1*q2-q0*q3)
    r02 = 2 * (q1*q3+q0*q2)
    
    r10 = 2 * (q1*q2+q0*q3)
    r11 = 2 * (q0*q0+q2*q2) - 1
    r12 = 2 * (q2*q3-q0*q1)
    
    r20 = 2 * (q1*q3-q0*q2)
    r21 = 2 * (q2*q3+q0*q1)
    r22 = 2 * (q0*q0+q3*q3) - 1
    
    rot_matrix = np.array([[r00, r01, r02],
                          [r10, r11, r12],
                          [r20, r21, r22]])
    return rot_matrix

Q = [0.99810947,0.06146124,0.234234,-0.1341234]
#Q = [0.234234,-0.1341234,0.99810947,0.06146124]
#Q = [1,0,0,0]
R = quaternion_rotation_matrix(Q)
print(R)
'''
