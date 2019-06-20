import numpy as np
import json
import logging


TIMESTEP = 32
DIRECTION = 0

DEVICES_FILE = "devices.json"

class Robot_Environment():

    def __init__(self, supervisor, file = DEVICES_FILE):
        self.sv = supervisor
        with open(file) as f:
            devices = json.load(f)
            sensor_names = devices["sensors"]
            lin_motor_names = devices["lin_motors"]
            rot_motor_names = devices["rot_motors"]

        self.init_sensors(sensor_names)
        self.init_motors(lin_motor_names, rot_motor_names)
        self.trans_field = self.sv.getFromDef("robot").getField("translation")

        # small float for handling motor velocity limit
        self.e = 1e-5

    def init_sensors(self, sensor_names):
        self.sensors = []
        for n in sensor_names:
            #TODO IMU
            s = self.sv.getTouchSensor(n)
            s.enable(TIMESTEP*4)
            self.sensors.append(s)

    def init_motors(self, lin_motor_names, rot_motor_names):
        #linear motors
        self.lin_motors = []
        for n in lin_motor_names:
            m = self.sv.getMotor(n)
            m.setPosition(float(np.inf))
            self.lin_motors.append(m)

        #rotational motors
        self.rot_motors = []
        for n in rot_motor_names:
            m = self.sv.getMotor(n)
            m.setPosition(float(np.inf))
            self.rot_motors.append(m)

        #TODO different maxvel for linear and rotational motor
        self.maxLinVel = self.lin_motors[0].getMaxVelocity()
        self.maxRotVel = self.rot_motors[0].getMaxVelocity()

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
        n_lin = len(self.lin_motors)
        n_rot = len(self.rot_motors)

        #set linear motor velocity
        for m in range(n_lin):
            #if np.abs(vel[m]) >= self.maxLinVel:
                #logging.info("action is clipped")

            #TODO better solution than clipping
            v = np.clip(vel[m], -self.maxLinVel+self.e, self.maxLinVel-self.e)
            self.lin_motors[m].setVelocity(float(v))

        #set rotational motor velocity
        for m in range(n_lin, n_rot):
            #if np.abs(vel[m]) >= self.maxRotVel:
             #   logging.info("action is clipped")
            v = np.clip(vel[m], -self.maxRotVel, self.maxRotVel)
            self.rot_motors[m].setVelocity(float(v))

    #TODO
    def calculate_energy(self):
        return 0

    def calculate_reward(self, pos0, pos1):
        rew = self.distance_travelled(pos0, pos1) #- energy
        return rew


    def check_termination(self):
        done = False
        return done


    def step(self,action, t=TIMESTEP):
        self.actuate_motors(action)
        pos0 = self.trans_field.getSFVec3f()
        self.sv.step(t)
        pos1 = self.trans_field.getSFVec3f()

        rew = self.calculate_reward(pos0, pos1)
        obs = self.get_sensor_data()
        done = self.check_termination()

        return obs,rew, done