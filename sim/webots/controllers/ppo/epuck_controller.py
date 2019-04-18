from controller import Robot
from controller import Supervisor
import numpy as np

#initial position of robot
INIT_TRANS = [0,0,0]
INIT_ROT = [0,1,0,np.pi/2]

#timestep should be 2X simulation time
TIMESTEP = 32

#TODO turn into class
class Robot_Environment():
    #sensors
    sensor_names = ["ps0", "ps1", "ps2", "ps3", "ps4", 
                    "ps5", "ps6", "ps7"]
    
    #motors
    motor_names = ["left wheel motor","right wheel motor"] 
    
    #sampling period
    sp = TIMESTEP * 4  
    
    #direction to drive
    direction = np.pi
    
    def __init__(self):
        self.sv = Supervisor() 
        self.robot_node = self.sv.getFromDef('epuck')
        self.init_motors()
        self.init_sensors()
    
    def init_motors(self):
        #motors
        self.motors = []
        for n in self.motor_names:
            self.motors.append(self.sv.getMotor(n))
        
        self.maxVel = self.motors[0].getMaxVelocity()

        #translation field for distance measure
        self.trans_field = self.robot_node.getField('translation')
        #rotational field for reset purpose
        self.rot_field = self.robot_node.getField('rotation')
    
    def init_sensors(self):
        self.sensors = []
        for n in self.sensor_names:
            self.sensors.append(self.sv.getDistanceSensor(n)) 
            
         #enable sensors
        for s in self.sensors:
            s.enable(self.sp)           
    
    #velocity control
    def control_motors(self, vel):
        for m in range(len(self.motors)):
            self.motors[m].setVelocity(np.double(vel[m]))
     
         
    def readSensorData(self):
        data = []
        for s in self.sensors:
            data.append(s.getValue())
        return np.array(data)    
    
    
    
    
    def distance_travelled(self, pos0, pos1,direction):
        #travlled euclidean distance
        l = np.sqrt((pos1[0]-pos0[0])**2+(pos1[2]-pos0[2])**2)
        #travelled angle (based on unit circle)
        x = pos1[0]-pos0[0]
        y = pos1[2]-pos0[2]
        alpha = np.arctan2(y,x)
        
        #compute distance of normal to position by cos(theta)*l
        theta = direction - alpha
        d = (np.cos(theta)*l)
        return d
    
    def check_collision(self):
        return False
    
    def step(self,action):
        #track position for distance measure
        pos0 = self.trans_field.getSFVec3f()
        
        #actuate the motors
        self.control_motors(action)
        
        pos1 = self.trans_field.getSFVec3f()
        
        #compute the reward
        rew = self.distance_travelled(pos0,pos1, self.direction)
        
        #get observations from sensors
        obs = self.readSensorData()
        
        #check if robot simulation should be reset
        done = self.check_collision()
        
        return rew, obs, done 
    
    #reset simulation
    def reset(self):
        self.trans_field.setSFVec3f(INIT_TRANS)
        self.rot_field.setSFVec3f(INIT_ROT)
    
        for m in self.motors:
            m.setPosition(0)
            m.setVelocity(0)
            m.setAcceleration(0)
     
        self.sv.simulationResetPhysics()
        
        #return reward
        return self.readSensorData()