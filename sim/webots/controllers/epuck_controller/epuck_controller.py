from controller import Robot
class Epuck_Controller():

    #sensors
    sensor_names = ["ps0", "ps1", "ps2", "ps3", "ps4", 
                    "ps5", "ps6", "ps7"]
    
    #motors
    motor_names = ["left wheel motor","right wheel motor"] 
    
    #sampling period
    sp = TIMESTEP * 4  