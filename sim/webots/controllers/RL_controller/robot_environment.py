import numpy as np
from controller import Supervisor

TIMESTEP = 32
DIRECTION = 0

# TODO sensor names
SENSORS = []
MOTORS = []

class Robot_Environment(Supervisor):

    def __init__(self):
        self.init_sensors()
        self.init_motors()

    def init_sensors(self, sensor_names = SENSORS):
        self.sensors = []
        for n in sensor_names:
            self.sensors.append(self.getDistanceSensor(n))

    def init_motors(self, motor_names = MOTORS):
        # motors
        self.motors = []
        for n in self.motor_names:
            self.motors.append(self.getMotor(n))

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


    def get_sensor_data(self,sensors=SENSORS):
        data = []
        for s in sensors:
            data.append(s.getValue())
        return data


    def actuate_motors(self, vel):
        for m in range(len(self.motors)):
            self.motors[m].setVelocity(vel[m])

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
        sv.step(t)
        pos1 = self.self.trans_field.getSFVec3f()

        rew = self.calculate_reward(pos0, pos1)
        obs = self.get_sensor_data()
        done = self.check_termination()

        return rew, obs, done