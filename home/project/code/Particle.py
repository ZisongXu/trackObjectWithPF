#Class of particle's structure
class Particle(object):
    def __init__(self,x=0.0,y=0.0,z=0.0,x_angle=0.0,y_angle=0.0,z_angle=0.0,w=1.0,index=0,linearVelocity=[0,0,0],angularVelocity=[0,0,0]):
        self.x = x
        self.y = y
        self.z = z
        self.x_angle = x_angle
        self.y_angle = y_angle
        self.z_angle = z_angle
        self.w = w
        self.index = index
        self.linearVelocity = linearVelocity
        self.angularVelocity = angularVelocity
        
    def as_pose(self):
        return True
