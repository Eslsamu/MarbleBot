"""rouqhQuadPyCtrl controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, LED, DistanceSensor
#from controller import Robot
from controller import *

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  led = robot.getLED('ledname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)

maxSpeed = 6.28

# Hip motors
motorHipFL1 = Motor("rotational motor1") # FL = Front left, etc. 
motorHipFL2 = Motor("rotational motor2")
motorHipFR1 = Motor("rotational motor3")
motorHipFR2 = Motor("rotational motor4")
motorHipRL1 = Motor("rotational motor5")
motorHipRL2 = Motor("rotational motor6")
motorHipRR1 = Motor("rotational motor7")
motorHipRR2 = Motor("rotational motor8")

# Gyroscope
#gyro1 = Gyro("gyro")
#gyro1.enable(500)
#print(gyro1.getValues())

# GPS
gps1 = GPS("gps")
gps1.enable(500)

#motorHipFL2.setAvailableForce(30)
#motorHipFL2.setForce(30)

# Slider motors
motorKneeFL = Motor("linear motor1")
#motorKneeFL.setForce(0)

# Commands for motors
motorHipFL1.setPosition(float('0.0')) # +inf
#motorHipFL2.setPosition(float('+inf'))

#motorHipFL1.setVelocity(0.5 * maxSpeed)
#motorHipFL2.setVelocity(0.5 * maxSpeed)

#motorKneeFL.setPosition(0.128)

#motorKneeFR.setVelocity(0.5 * maxSpeed)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    #print(gyro1.getValues())
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  led.set(1)
    pass

# Enter here exit cleanup code.
