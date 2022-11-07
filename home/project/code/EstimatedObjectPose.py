#Class of particle's structure
class EstimatedObjectPose(object):
    def __init__(self, esti_obj_name=0, esti_obj_id=0, pos=[0.0, 0.0, 0.0], ori=[0.0, 0.0, 0.0, 1.0], index=0):

        self.esti_obj_name = esti_obj_name
        self.esti_obj_id = esti_obj_id
        self.pos = pos
        self.ori = ori # x, y, z, w
        self.index = index
        
    def as_pose(self):
        return True
