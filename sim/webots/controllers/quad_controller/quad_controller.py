from controller import Robot

class Quad_Controller() extends Robot:
     
     
     shoulder_motor_names = ["BR_Motor_Shoulder_Abduction", 
                             "BR_Motor_Shoulder_Rotation",
                             "BL_Motor_Shoulder_Abduction",  
                             "FR_Motor_Shoulder_Abduction", 
                             "FR_Motor_Shoulder_Rotation",                    
                             "FL_Motor_Shoulder_Abduction", 
                             "FL_Motor_Shoulder_Rotation"
                             ]
     knee_motor_names = ["FL_Motor_Knee","BR_Motor_Knee",
                         "BL_Motor_Knee","FR_Motor_Knee",]
      
     imu_name = "IMU_Torso"
     force_sensors_names = ["BR_Force",
                      "BL_Force",
                      "FR_Force",
                      "FL_Force"]
     
     def __init__():
         self.init_motors()
         self.init_sensors()
         
         #TODO
         #translation field for distance measure
         self.trans_field = getField('translation')
         #rotational field for reset purpose
         self.rot_field = getField('rotation')
     
     def init_motors(self):
        #motors 
        self.motors = []
        for n in self.shoulder_motor_names:
            self.motors.append(getMotor(n))
        for n in self.knee_motor_names:
            self.motors.append(getMotor(n)
            
  
    def init_sensors(self):
        self.sensors = {}
        #imu
        imu = getInertialUnit(imu_name)
        imu.enable()
        self.sensors["imu"] = imu
        #force sensors
        self.sensors["force"] = []
        for n in self.force_sensors_name:
            force_sensor = getTouchSensor(n)
            force_sensor.enable()
            self.sensors["force"].append(force_sensor)
         
            
   #sets target position of all motors
    def control_motors(self, pos):
        for m in range(len(self.motors)):
            self.motors[m].setPosition(np.double(pos[m]))
   
   #returns imu and force sensor data
    def readSensorData(self):
        data = []
        #imu data
        data =  data + self.sensors["imu"].getRollPitchYaw()
        #input force vector as three seperate sensor values 
        for s in self.sensors["force"]:
            data = data + s.getValues()      
        return np.array(data)      
    
    
    def check_collision(self):
        return False
        
    