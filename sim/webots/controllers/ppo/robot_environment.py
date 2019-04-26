
import numpy as np



#timestep should be 2X simulation time
TIMESTEP = 32

#TODO turn into class
class Robot_Environment() extends Supervisor:
    
    #initial position of robot
    INIT_TRANS = [0,0,0]
    INIT_ROT = [0,1,0,np.pi/2]

    
    #direction to drive
    direction = 3*np.pi/2
    
   
    
    #
    def distance_travelled(self, pos0, pos1,direction):
        #travlled euclidean distance
        l = np.sqrt((pos1[0]-pos0[0])**2+(pos1[2]-pos0[2])**2)
        
        #travelled angle (based on unit circle)
        x = pos1[0]-pos0[0]
        y = pos1[2]-pos0[2]
        alpha = np.arctan2(y,x)
        
        #compute distance of normal to position by cos(angle) times distance
        theta = direction - alpha
        d = (np.cos(theta)*l)
        return d
      
 
    
    def step(self,action):
        
        for r in robots:
            r.step(action)
        
        #track position for distance measure
        pos0 = self.trans_field.getSFVec3f()
        
        #robot step -> exit condition TODO
        if self.sv.step(TIMESTEP) is -1:
            pass
        
        #actuate the motors
        self.control_motors(action)
        #print(action)
        
        pos1 = self.trans_field.getSFVec3f()
        
        #compute the reward
        rew = self.distance_travelled(pos0,pos1, self.direction)
        
        #get observations from sensors
        obs = self.readSensorData()
        
        #check if robot simulation should be reset
        done = self.check_collision()
        
        
        
        #print(rew,obs, done)
        
        return rew, obs, done 
    
    #reset robot
    def reset(self):
        self.trans_field.setSFVec3f(INIT_TRANS)
        self.rot_field.setSFRotation(INIT_ROT)
    
        for m in self.motors:
            m.setPosition(float('inf'))
            m.setVelocity(0)
            m.setAcceleration(-1)
         
        self.sv.simulationResetPhysics()
        
        #return reward
        return self.readSensorData()