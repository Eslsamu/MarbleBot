import numpy as np
import json


TIMESTEP = 32
DIRECTION = np.pi

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

        #sensor init step
        self.sv.step(TIMESTEP)

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



    def init_motors(self, rot_motor_names):
        #rotational motors
        self.rot_motors = []
        for n in rot_motor_names:
            m = self.sv.getMotor(n)
            m.setPosition(float(np.inf))
            self.rot_motors.append(m)

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
            for v in vals:
                data.append(v)

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
    def actuate_motors(self, vel):
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

    def energy_consumed(self, power0, power1):
        return power0-power1

    def calculate_reward(self, pos0, pos1, power0, power1, clipped,c_rew=10, c_ene = 0.00001, c_clip=0.01, survival = 0.01):
        d = c_rew * self.distance_travelled(pos0, pos1)
        e = c_ene * self.energy_consumed(power0, power1)
        c = c_clip * clipped

        rew = d - e - c + survival

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