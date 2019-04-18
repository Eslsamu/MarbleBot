from controller import Robot
from controller import Supervisor
from controller import Motor
import numpy as np

#initial position of robot
INIT_TRANS = [0,0,0]
INIT_ROT = [0,1,0,np.pi/2]

#timestep should be 2X simulation time
TIMESTEP = 32

#TODO turn into class
class Robot_Environment():
    #sensors
    pos_sensor = ["HeadPitchS",
                "HeadYawS",
                "LAnklePitchS",
                "LAnkleRollS",
                "LElbowRollS",
                "LElbowYawS",
                "LHipPitchS",
                "LHipRollS",
                "LHipYawPitchS",
                "LKneePitchS",
                "LPhalanx1S",
                "LPhalanx2S",
                "LPhalanx3S",
                "LPhalanx4S",
                "LPhalanx5S",
                "LPhalanx6S",
                "LPhalanx7S",
                "LPhalanx8S",
                "LShoulderPitchS",
                "LShoulderRollS",
                "LWristYawS",
                "RAnklePitchS",
                "RAnkleRollS",
                "RElbowRollS",
                "RElbowYawS",
                "RHipPitchS",
                "RHipRollS",
                "RHipYawPitchS",
                "RKneePitchS",
                "RPhalanx1S",
                "RPhalanx2S",
                "RPhalanx3S",
                "RPhalanx4S",
                "RPhalanx5S",
                "RPhalanx6S",
                "RPhalanx7S",
                "RPhalanx8S",
                "RShoulderPitchS",
                "RShoulderRollS",
                "RWristYawS"]
    
    inertia_sensor = ["inertial unit"]
    
    touch_sensor = ["LFoot/Bumper/Left",
                    "LFoot/Bumper/Right",
                    "LFsr",
                    "RFoot/Bumper/Left",
                    "RFoot/Bumper/Right",
                    "RFsr"]
               
    receiver = ["receiver"]
     
    distance_sensor = ["Sonar/Left",
                        "Sonar/Right"]
    
    #motors
    motor_names = ["HeadPitch"] 
    
    #sampling period
    sp = TIMESTEP * 4  
    
    #direction to drive
    direction = np.pi
    
    def __init__(self):
        self.sv = Supervisor() 
        self.robot_node = self.sv.getFromDef("nao")
        if self.robot_node is None:
            print("nao not found")
            return
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
        for n in sensor_names:
            self.sensors.append(sv.getDistanceSensor(n)) 
            
         #enable sensors
        for s in self.sensors:
            s.enable(sp)           
    
    #velocity control
    def control_motors(self, vel):
        for m in range(len(motors)):
            motors[m].setVelocity(vel[m])
     
         
    def readSensorData(self):
        data = []
        for s in self.sensors:
            data.append(s.getValue())
        
        return data    
    
    
    
    
    def distance_travelled(pos0, pos1,direction):
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
    
    def check_collision():
        return False
    
    def step(action):
        #track position for distance measure
        pos0 = trans_field.getSFVec3f()
        
        #actuate the motors
        control_motors(action)
        
        pos1 = trans_field.getSFVec3f()
        
        #compute the reward
        rew = distance_travelled(pos0,pos1, direction)
        
        #get observations from sensors
        obs = readSensorData()
        
        #check if robot simulation should be reset
        done = check_collision
        
        return rew, obs, done 
    
    #reset simulation
    def reset():
        trans_field.setSFVec3f(INIT_TRANS)
        rot_field.setSFVec3f(INIT_ROT)
    
        for m in motors:
            m.setPosition(0)
            m.setVelocity(0)
            m.setAcceleration(0)
     
        sv.simulationResetPhysics()
        
        #return reward
        return readSensorData()
