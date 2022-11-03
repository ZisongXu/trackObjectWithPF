#Class of particle's structure
class Particle(object):
    def __init__(self, pos = [0.0, 0.0, 0.0], ori = [0.0, 0.0, 0.0, 1.0], w=1.0, index=0, linearVelocity=[0,0,0], angularVelocity=[0,0,0]):
        self.pos = pos
        self.ori = ori # x, y, z, w
        
        self.w = w
        self.index = index
        self.linearVelocity = linearVelocity
        self.angularVelocity = angularVelocity
        
    def as_pose(self):
        return True
