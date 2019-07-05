import numpy as np
import json


TIMESTEP = 32

"""
North: pi
West: 3pi/2
South: 0
East: pi/2
"""

DIRECTION = np.pi/2

DEVICES_FILE = "devices_gd.json"

class Robot_Environment():

    def __init__(self, supervisor, file = DEVICES_FILE):
        self.sv = supervisor
        with open(file) as f:
            devices = json.load(f)
            force_sensor_names = devices["force_sensors"]
            rot_motor_names = devices["rot_motors"]
            collision_detector_name = devices["collision_detector"][0]

        self.sv.batterySensorEnable(TIMESTEP)
        self.init_motors(rot_motor_names)
        self.init_sensors(force_sensor_names)#IMU_names, gyro_name, acc_name)

        self.init_collision_detection(collision_detector_name)
        self.trans_field = self.sv.getFromDef("robot").getField("translation")

        # small float for handling motor velocity limit
        self.e = 1e-5

        # time in s
        self.t = 0

        #sensor init step
        self.sv.step(TIMESTEP)
        self.t += TIMESTEP/1000

        #reference gait parameters
        self.frq = 1.7
        self.ampl = 1.0

        assert TIMESTEP == self.sv.getBasicTimeStep()


    def init_collision_detection(self, collision_detector_name):
        s = self.sv.getTouchSensor(collision_detector_name)
        s.enable(TIMESTEP*2)
        self.collision_detector = s


    def init_sensors(self, force_sensor_names):#, acc_name,IMU_names ):

        self.force_sensors = []
        for n in force_sensor_names:
            s = self.sv.getTouchSensor(n)
            s.enable(TIMESTEP*2)
            self.force_sensors.append(s)

        #must be executed after motors are init
        self.pos_sensors = []
        for n in self.rot_motors:
            s = n.getPositionSensor()
            s.enable(TIMESTEP * 2)
            self.pos_sensors.append(s)


    """
    position control
    """
    def init_motors(self, rot_motor_names):
        #rotational motors
        self.rot_motors = []
        for n in rot_motor_names:
            m = self.sv.getMotor(n)
            self.rot_motors.append(m)



    """
    distance measure based on checkpoint or relative to robot itself
    """
    def distance_travelled(self, pos0, pos1, direction=None):
        if not direction:
            dx = (pos1[0] - pos0[0])/TIMESTEP
            return dx
        else:
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
        for s in self.force_sensors:
            val = s.getValue()
            #in the beginning of the sampling period the sensor data is Nan
            if np.isnan(val):
                val = 0
            data.append(val)


        for i in self.pos_sensors:
            val = i.getValue()
            # in the beginning of the sampling period the sensor data is Nan
            if np.isnan(val):
                val = 0
            data.append(val)

        return np.array(data)


    """
    velocity motor control for abduction, rotation and contraction of legs
    
    this version of controlling returns the sum of the clipped velocity amount
    """
    def actuate_motors_vel(self, vel):
        n_rot = len(self.rot_motors)

        total_clipped = 0

        #set rotational motor velocity
        for m in range(n_rot):
            dif = np.abs(vel[m]) - (self.maxRotVel - self.e)
            if dif > 0:
                total_clipped += dif
                if vel[m] > 0:
                    v = float(vel[m] - dif)
                else:
                    v = float(vel[m] + dif)
            else:
                v = float(vel[m])

            self.rot_motors[m].setVelocity(v)


        avg_clipped = total_clipped / (n_rot)
        return avg_clipped

    def actuate_motors_pos(self, vel):
        n_rot = len(self.rot_motors)
        for m in range(n_rot):
            v = float(vel[m])
            self.rot_motors[m].setPosition(v)



    def energy_consumed(self, power0, power1):
        return power0-power1

    def calculate_reward(self, pos0, pos1, power0, power1, c_rew=150, c_ene = 0.0003, survival = 0.01):
        d = c_rew * self.distance_travelled(pos0, pos1)
        e = c_ene * self.energy_consumed(power0, power1)

        #do not count timesteps as survival where three or more feet touch the ground to avoid local minima
        contact = 0
        for f in self.force_sensors:
            if f.getValue():
                contact += 1
        if contact > 2:
            s = 0
        else:
            s = survival

        rew = d - e  + s

        return rew, {"distance":d,"energy":e,"survival":s}

    def check_termination(self, counter = 5):
        #if counter:
            #if val

        val = self.collision_detector.getValue()
        #sensor value is Nan at beginning of simulation
        if np.isnan(val):
            return False
        elif val:
            # If robot's torso is making contact with any solid
            #if counter:
             #   self.c
            return True
        return False

    """
    simple oscillatory galopp reference gait (just for hip motors)
    """
    def reference(self, action):

        m_activations = len(action)
        assert m_activations % 8 == 0

        a = action.reshape(int(m_activations/8),8)

        phase = self.t * 2 * np.pi * self.frq

        pos = self.ampl * a[0] * np.sin(phase + a[1]) + a[2]
        return pos

    def step(self,action, confidence=1,dt=TIMESTEP):
        control = self.reference(action) #* confidence

        self.actuate_motors_pos(control)

        pos0 = self.trans_field.getSFVec3f()
        power0 = self.sv.batterySensorGetValue()

        self.sv.step(dt)

        #timer for reference gait
        self.t += dt/1000

        pos1 = self.trans_field.getSFVec3f()
        power1 = self.sv.batterySensorGetValue()

        rew, r_info = self.calculate_reward(pos0, pos1, power0, power1)
        obs = self.get_sensor_data()
        done = self.check_termination()

        return obs, rew, done, r_info
