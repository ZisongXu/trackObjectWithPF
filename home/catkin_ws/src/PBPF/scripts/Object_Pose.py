#!/usr/bin/python3
#Class of particle's structure
class Object_Pose(object):
    def __init__(self, obj_name=0, obj_id=0, pos=[0.0, 0.0, 0.0], ori=[0.0, 0.0, 0.0, 1.0], index=0):
        self.obj_name = obj_name
        self.obj_id = obj_id
        self.pos = pos
        self.ori = ori # x, y, z, w
        self.index = index
        
    def as_pose(self):
        return True
