#!/usr/bin/python3
#Class of particle's structure
class Robot_Pose(object):
    def __init__(self, obj_name=0, obj_id=0, pos=[0.0, 0.0, 0.0], ori=[0.0, 0.0, 0.0, 1.0], joints=0, trans_matrix=0, index=0):
        self.obj_name = obj_name
        self.obj_id = obj_id
        self.pos = pos
        self.ori = ori # x, y, z, w
        self.joints = joints
        self.trans_matrix = trans_matrix
        self.index = index
        
    def as_pose(self):
        return True
