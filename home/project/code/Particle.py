#Class of particle's structure
class Particle(object):
    def __init__(self, par_name=0, par_id=0, pos = [0.0, 0.0, 0.0], ori = [0.0, 0.0, 0.0, 1.0], w=1.0, index=0, linearVelocity=[0,0,0], angularVelocity=[0,0,0]):
        self.par_name = par_name
        self.par_id = par_id
        self.pos = pos
        self.ori = ori # x, y, z, w
        self.w = w
        self.index = index
        self.linearVelocity = linearVelocity
        self.angularVelocity = angularVelocity
        
    def as_pose(self):
        return True
