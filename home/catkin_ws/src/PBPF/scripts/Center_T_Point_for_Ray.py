#!/usr/bin/python3
#Class of particle's structure
class Center_T_Point_for_Ray(object):
    def __init__(self, parC_T_p_pos=[0.0, 0.0, 0.0], parC_T_p_ori=[0.0, 0.0, 0.0, 1.0], parC_T_p_4_4=0, index=0):
        self.parC_T_p_pos = parC_T_p_pos
        self.parC_T_p_ori = parC_T_p_ori # x, y, z, w
        self.parC_T_p_4_4 = parC_T_p_4_4
        self.index = index
        
    def as_pose(self):
        return True
