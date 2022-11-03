#Class of particle's structure
class ObservationPose(object):
    def __init__(self, obs_obj_id=0, pos=[0.0, 0.0, 0.0], ori=[0.0, 0.0, 0.0, 1.0], index=0):
        self.obs_obj_id = obs_obj_id
        self.pos = pos
        self.ori = ori # x, y, z, w
        self.index = index

    def as_pose(self):
        return True
