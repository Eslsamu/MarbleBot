"""This controller measures robot energy consumption."""

from controller import *
import time

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())


maxSpeed = 6.28

#-------------------------Motor Stuff-----------------------------------
# Hip motors
motorHipFL1 = Motor("rotational motor1") # FL = Front left, etc. 
motorHipFL2 = Motor("rotational motor2")
motorHipFR1 = Motor("rotational motor3")
motorHipFR2 = Motor("rotational motor4")
motorHipRL1 = Motor("rotational motor5")
motorHipRL2 = Motor("rotational motor6")
motorHipRR1 = Motor("rotational motor7") # RR = Rear right, etc.
motorHipRR2 = Motor("rotational motor8")

#motorHipFL2.setAvailableForce(30)
#motorHipFL2.setForce(30)

# Slider motors
motorKneeFL = Motor("linear motor1")
#motorKneeFL.setForce(0)

# Commands for motors
#motorHipFL1.setPosition(float('+inf')) # +inf
#motorHipFL2.setPosition(float('-1'))

#motorHipFL1.setVelocity(0.5 * maxSpeed)
#motorHipFL2.setVelocity(0.5 * maxSpeed)

#motorKneeFL.setPosition(0.128)

#motorKneeFR.setVelocity(0.5 * maxSpeed)

motSample = 500 # sampling period for getting torque force feedback measurements. 
motorHipFL2.enableTorqueFeedback(motSample)
#-----------------------------------------------------------------------

#--------------------------Battery Stuff--------------------------------
batSample = 500 # battery sampling period

robot.batterySensorEnable(batSample) # enable battery sensor

currentPower = 0
firstMeasurement = 0
firstMeasurementRecorded = False
#-----------------------------------------------------------------------
#-------------------------Timing stuff in loop---------------------------
timeDifference = 0 # used in the sim. loop below.
desiredPrintInterval = 0.5 # In seconds. Time interval in which the desired sim. details are printed.
start = time.time() # for the interval info printing
anotherStart = time.time() # for when the final energy consumption is to be calculated.
endTime = 15 # Time after which total energy consumption is printed.
#-----------------------------------------------------------------------

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    nowTime = time.time()
    timeDifference = nowTime - start
        
    if (timeDifference > desiredPrintInterval): # this timing check ensures that info's printed within the desired time interval.
        start = time.time()
        # Get battery measurements
        currentPower = robot.batterySensorGetValue()
        if ((firstMeasurementRecorded == False) and (currentPower > 0)):
            firstMeasurementRecorded = True
            firstMeasurement = currentPower
        
        print("Remaining power: " + str(currentPower))
        
        # Print motor torque feedback.
        print("Torque feedback: " + str(motorHipFL2.getTorqueFeedback()))
      
    if ((time.time() - anotherStart) > endTime):
        print("Energy consumed: " + str(firstMeasurement - currentPower))
    
    pass

# Enter here exit cleanup code.
