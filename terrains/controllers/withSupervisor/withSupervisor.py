"""rouqhQuadPyCtrl controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, LED, DistanceSensor
from controller import *
import random
#from  controller import Supervisor # Not

reloadTime = 1500000 # Time after which world will reload

def changeDyn():
    print('ya man!')
    """
    file1 = open("../../worlds/tex.txt", "w")
    file1.write('sumo wrestlers.')
    file1.close()
    """
    
    return


# create the Robot instance.
#robot = Robot() # Maybe not necessary because of the Supervisor?
superMan = Supervisor()

"""
# ------------------ Dynanimcs randomization related -------------------------
# Get node handle
node1 = superMan.getFromDef("boxer1")
field1 = node1.getField("size")
list1 = [5, 0.2, 0.1]
field1.setSFVec3f(list1)
# -----------------------------------------------------------------------------
"""

# ----------------- elevationGrid randomization -----------------------------------
def randElevGrid():
    # Get fields to randomize 
    elevNode = superMan.getFromDef("elevgridGeom")
    elevHeightField = elevNode.getField("height")
    elevXField = elevNode.getField("xDimension")
    elevZField = elevNode.getField("zDimension")
    xSpacingField = elevNode.getField("xSpacing") 
    zSpacingField = elevNode.getField("zSpacing")
    # Get field for position of elevGrid so you can change it right before the other changes are made.
    elevNode2 = superMan.getFromDef("Heightfield1")
    transField = elevNode2.getField("translation")
    updatePos = ([float(1.32), float(0.35), float(0.9)])
    transField.setSFVec3f(updatePos)
    
    # Ranges for random values for the x and z dimensions
    dimRandRangeMin = 1
    dimRandRangeMax = 5
    
    # Range for random heightfield heights
    # Values might be weakend a bit to decrease height extremities.
    elevMin = 9 # Originally 1 # Consider changing this value from 1 to something else at some point. 
    elevMax = 54 # Originally 100 # The actual heights won't go up to 100, as later the random value will be divided by 100 to get a decimal number
    
    # Random values for the x and z dimensions
    randX = random.randint(dimRandRangeMin, dimRandRangeMax)
    randZ = random.randint(dimRandRangeMin, dimRandRangeMax)
    
    # Random values for the x- and zSpacing
    # These spaceRange values have been made a little less extreme so the elevGrid isn't too extreme with resepct to spacing.
    spaceRandRangeMax = 85 # Originally 100
    spaceRandRangeMin = 50 # Originally 4
    randSpaceX = (random.randint(spaceRandRangeMin, spaceRandRangeMax) / 100) * 2 # Multiply by 2 so the range should be around 0-2?
    randSpaceZ = (random.randint(spaceRandRangeMin, spaceRandRangeMax) / 100) * 2
    print("randSpaceX: " + str(randSpaceX))
    print("randSpaceZ: " + str(randSpaceZ))
    
    # calculate length of elevHeightField
    #elevHeightFieldLength = randX * randZ # Comment this out, if size of elevGrid is to remain the same
    elevHeightFieldLength = elevXField.getSFInt32() * elevZField.getSFInt32() # Comment this out and use the one above if size of elevgrid is to be modified.
    
    # Generate elevHeightFieldLength amount of random height values
    heightStore = []
    for x in range(elevHeightFieldLength):
        randVal = random.randint(elevMin, elevMax) / 100 # / 100 so that the values are nice decimal numbers.
        heightStore.append(float(randVal)) # Cast to float because mehtod below requires floats.
    
    print(heightStore)  
     # Set the elevationGrid values to the new randommized values.
    #elevXField.setSFInt32(randX) # Comment out if size of elevgrid is to be unchanged.
    #elevZField.setSFInt32(randZ) # Comment out if size of elevgrid is to be unchanged.
    xSpacingField.setSFFloat(randSpaceX) # Comment out if spacing is to be unchanged.
    zSpacingField.setSFFloat(randSpaceZ) # Comment out if spacing is to be unchanged.
    for x in range(elevHeightFieldLength):
        elevHeightField.setMFFloat(x, heightStore[x])
     
    return
    
# -------------------------------------------------------------------------

# get the time step of the current world.
timestep = int(superMan.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  led = robot.getLED('ledname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)

"""
# Writes into filea one directory up.
file1 = open("../filea.txt", "w")
file1.write("fileWriter waz here!")
file1.close()
"""

maxSpeed = 6.28

# ------------------ Motor control related stuff -------------------------
"""
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
gyro1 = Gyro("gyro")
gyro1.enable(500)
#print(gyro1.getValues())

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
"""
# -------------------------------------------------------------------------

# Method calls and other stuff
randElevGrid() # Comment this to stop random elevGrid generation.

allowMethod = False

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while superMan.step(timestep) != -1:
    #print(gyro1.getValues())
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  led.set(1)
    
    if (allowMethod and superMan.getTime() > 500):
        allowMethod = False
        changeDyn()
    
    
    # Testing out worldReload()
    if (superMan.getTime() > reloadTime):
        print("Reload")
        #superMan.worldReload()
    
    pass # used when a statement is required syntactically, but you don't want any command or code to execute.

# Enter here exit cleanup code.
