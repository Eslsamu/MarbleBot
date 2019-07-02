import numpy as np
import json


TIMESTEP = 32
DIRECTION = np.pi

DEVICES_FILE = "devices.json"

class Robot_Environment():

    def __init__(self, supervisor, file = DEVICES_FILE):
        self.sv = supervisor
        with open(file) as f:
            devices = json.load(f)
            force_sensor_names = devices["force_sensors"]
            IMU_names = devices["IMUs"]
            lin_motor_names = devices["lin_motors"]
            rot_motor_names = devices["rot_motors"]
            collision_detector_name = devices["collision_detector"][0]

        self.sv.batterySensorEnable(TIMESTEP)
        self.init_sensors(force_sensor_names, IMU_names)
        self.init_motors(lin_motor_names, rot_motor_names)
        self.init_collision_detection(collision_detector_name)
        self.trans_field = self.sv.getFromDef("robot").getField("translation")

        # small float for handling motor velocity limit
        self.e = 1e-5

    def init_collision_detection(self, collision_detector_name):
        s = self.sv.getTouchSensor(collision_detector_name)
        s.enable(TIMESTEP*2)
        self.collision_detector = s


    def init_sensors(self, force_sensor_names, IMU_names):
        self.force_sensors = []
        for n in force_sensor_names:
            s = self.sv.getTouchSensor(n)
            s.enable(TIMESTEP*2)
            self.force_sensors.append(s)
        self.IMUs = []
        for n in IMU_names:
            s = self.sv.getInertialUnit(n)
            s.enable(TIMESTEP*2)
            self.IMUs.append(s)

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
        for s in self.force_sensors:
            vals = s.getValues()
            #in the beginning of the sampling period the sensor data is Nan
            if np.isnan(vals).any():
                vals = np.zeros(len(vals))
            data.append(vals)

        for i in self.IMUs:
            vals = i.getRollPitchYaw()
            # in the beginning of the sampling period the sensor data is Nan
            if np.isnan(vals).any():
                vals = np.zeros(len(vals))
            data.append(vals)

        data = np.array(data)
        return data

    """
    velocity motor control for abduction, rotation and contraction of legs
    
    this version of controlling returns the sum of the clipped velocity amount
    """
    def actuate_motors(self, vel):
        n_lin = len(self.lin_motors)
        n_rot = len(self.rot_motors)

        total_clipped = 0

        #set linear motor velocity
        for m in range(n_lin):
            dif = np.abs(vel[m]) - (self.maxLinVel - self.e)
            if dif > 0:
                total_clipped += dif
                if vel[m] > 0:
                    v = float(vel[m] - dif)
                else:
                    v = float(vel[m] + dif)
            else:
                v = float(vel[m])
            self.lin_motors[m].setVelocity(v)

        #set rotational motor velocity
        for m in range(n_rot):
            dif = np.abs(vel[m+n_lin]) - (self.maxRotVel - self.e)
            if dif > 0:
                total_clipped += dif
                if vel[m+n_lin] > 0:
                    v = float(vel[m+n_lin] - dif)
                else:
                    v = float(vel[m+n_lin] + dif)
            else:
                v = float(vel[m+n_lin])

            self.rot_motors[m].setVelocity(v)

        return total_clipped


    def energy_consumed(self, power0, power1):
        return power0-power1

    def calculate_reward(self, pos0, pos1, power0, power1, clipped,c_rew=1, c_ene = 0.000001, c_clip=0.01):
        d = c_rew * self.distance_travelled(pos0, pos1)
        e = c_ene * self.energy_consumed(power0, power1)
        c = c_clip * clipped

        rew = d - e - c
        return rew, {"distance":d,"energy":e,"clipped":c}

    def check_termination(self):
        val = self.collision_detector.getValue()
        #sensor value is Nan at beginning of simulation
        if np.isnan(val):
            return False
        elif val:
            # If robot's torso is making contact with any solid
            return True
        return False


    def step(self,action, t=TIMESTEP):
        clipped = self.actuate_motors(action)

        pos0 = self.trans_field.getSFVec3f()
        power0 = self.sv.batterySensorGetValue()

        self.sv.step(t)

        pos1 = self.trans_field.getSFVec3f()
        power1 = self.sv.batterySensorGetValue()

        rew, r_info = self.calculate_reward(pos0, pos1, power0, power1, clipped)
        obs = self.get_sensor_data()
        done = self.check_termination()

        return obs, rew, done, r_info