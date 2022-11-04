#Class of particle's structure
class OptitrackPose(object):
    def __init__(self, opti_obj_name=0, opti_obj_id=0, pos=[0.0, 0.0, 0.0], ori=[0.0, 0.0, 0.0, 1.0], index=0):
        self.opti_obj_name = opti_obj_name
        self.opti_obj_id = opti_obj_id
        self.pos = pos
        self.ori = ori # x, y, z, w
        self.index = index
        
    def as_pose(self):
        return True
