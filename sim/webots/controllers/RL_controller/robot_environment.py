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
            sensor_names = devices["sensors"]
            motor_names = devices["motors"]

        self.init_sensors(sensor_names)
        self.init_motors(motor_names)

    def init_sensors(self, sensor_names):
        self.sensors = []
        for n in sensor_names:
            #TODO IMU
            logging.warning(n + " "+ str(type(n)))
            self.sensors.append(self.sv.getTouchSensor(n))

    def init_motors(self, motor_names):
        # motors
        self.motors = []
        for n in motor_names:
            self.motors.append(self.sv.getMotor(n))

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
            self.sv.motors[m].setVelocity(vel[m])

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
        pos0 = self.sv.trans_field.getSFVec3f()
        self.step(t)
        pos1 = self.sv.self.trans_field.getSFVec3f()

        rew = self.calculate_reward(pos0, pos1)
        obs = self.get_sensor_data()
        done = self.check_termination()

        return rew, obs, done