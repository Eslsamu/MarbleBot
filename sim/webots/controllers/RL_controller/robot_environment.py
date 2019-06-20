import numpy as np
import json
import logging
logging.basicConfig(filename='env.log', format='%(asctime)s %(message)s')

TIMESTEP = 32
DIRECTION = 0

DEVICES_FILE = "devices.json"

class Robot_Environment():

    def __init__(self, supervisor, file = DEVICES_FILE):
        self.sv = supervisor
        with open(file) as f:
            devices = json.load(f)
            sensor_names = devices["force_sensors"]
            motor_names = devices["motors"]
            collision_detector_names = devices["collision_detectors"]
            IMU_names = devices["IMUs"]

        self.init_sensors(sensor_names)
        self.init_motors(motor_names)
        self.init_collision_detectors(collision_detector_names)
        self.trans_field = self.sv.getFromDef("robot").getField("translation")

    def init_sensors(self, sensor_names):
        self.sensors = []
        for n in force_sensor_names:
            #TODO IMU
            s = self.sv.getTouchSensor(n)
            s.enable(2*TIMESTEP) # 2 * TIMESTEP  is twice as good.
            self.sensors.append(s)
        for n in IMU_names:
            y =  self.sv.getInertialUnit(n)
            y.enable(2*TIMESTEP)
            y.sensors.append(y)

    def init_motors(self, motor_names):
        # motors
        self.motors = []
        for n in motor_names:
            m = self.sv.getMotor(n)
            m.setPosition(float(np.inf))
            self.motors.append(m)

    def init_collision_detectors(self, collision_detector_names):
        self.collision_detectors = []
        for n in collision_detector_names:
           t = self.sv.getToucSensor(n)
           t.enable(TIMESTEP)
           self.collision_detectors.append(t)


        #TODO different maxvel for linear and rotational motor
        self.maxVel = self.motors[0].getMaxVelocity()

    def distance_travelled(self, pos0, pos1, direction=DIRECTION):
        # travlled euclidean distance
        l = np.sqrt((pos1[0] - pos0[0]) ** 2 + (pos1[2] - pos0[2]) ** 2)

        # travelled angle (based on unit circle)
        x = pos1[0] - pos0[0]
        y = pos1[2] - pos0[2]
        alpha = np.arctan2(y, x)

        # compute distance of normal to position by cos(angle) times distance
        theta = direction - alpha
        d = (np.cos(theta) * l)
        return d


    def get_sensor_data(self):
        data = []
        for s in self.sensors:
            data.append(s.getValues())
        data = np.array(data)
        return data


    def actuate_motors(self, vel):
        for m in range(len(self.motors)):
            self.motors[m].setVelocity(float(vel[m]))
    
    def enableBattery(self):
        batSample = 500 # battery sampling period
        self.sv.batterySensorEnable(batSample)
        #startingPower = robot.batterySensorGetValue() # this one won't work because the first proper battery reading will only appear after the first sampling period amount of time has passed

    #TODO for every sim. step. So compare change in power from step to step, not from start and current step
    def calculate_energy(self, lastPower, nowPower):
        return (lastPower - nowPower)

    def calculate_reward(self, pos0, pos1, lastPower, nowPower):
        # energy has high value 
        rew = self.distance_travelled(pos0, pos1) - self.calculate_energy(lastPower, nowPower) 
        return rew

    def check_termination(self):
        # if body rotation is upside down then terminated
        done = False
        if (self.collision_detectors[0].getValue() == 1): # If robot's touch sensor is making contact with any solid
            done = True
        return done


    def step(self,action, t=TIMESTEP):
        self.actuate_motors(action)
        pos0 = self.trans_field.getSFVec3f()
        power0 = self.sv.batterySensorGetValue()
        self.sv.step(t)
        power1 = self.sv.batterySensorGetValue()
        pos1 = self.trans_field.getSFVec3f()
        
        rew = self.calculate_reward(pos0, pos1, power0, power1)
        obs = self.get_sensor_data()
        done = self.check_termination()

        return obs,rew, done
