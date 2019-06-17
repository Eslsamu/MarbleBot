"""rouqhQuadPyCtrl controller."""

"""--------------readme-----------------------------------------
-this supervisor file generates random terrains on which the 
robot is to be trained on. 
-terrain generation parameters can be changed in the function
calls section
---------------------------------------------------------------"""


from controller import *
import random

superMan = Supervisor()

reloadTime = 1500000 # Time after which world will reload

globalEndPos = ([float(0), float(0), float(0)])
globalElevGridPos = ([float(-2.5), float(1.5), float(-2)])

# create the Robot or supervisor instance.
#robot = Robot() # Maybe not necessary because of the Supervisor?

# ----------------- elevationGrid randomization -----------------------------------
def randElevGrid(elevNode):
    # Get fields to randomize 
    """
    #elevNode = superMan.getFromDef("elevgridGeom")
    elevHeightField = elevNode.getField("height")
    print("elevheightField: " + str(elevHeightField))
    elevXField = elevNode.getField("xDimension")
    print("elevXField: " + str(elevXField))
    elevZField = elevNode.getField("zDimension")
    xSpacingField = elevNode.getField("xSpacing") 
    print("xSpacingField: " + str(xSpacingField))
    zSpacingField = elevNode.getField("zSpacing")
    """
    
    # same as above but using DEF to get fields to randomize
    elevgridGeomNode = superMan.getFromDef("elevgridGeom")
    elevHeightField = elevgridGeomNode.getField("height")
    elevXField = elevgridGeomNode.getField("xDimension")
    elevZField = elevgridGeomNode.getField("zDimension")
    xSpacingField = elevgridGeomNode.getField("xSpacing")
    zSpacingField = elevgridGeomNode.getField("zSpacing")
    
    # Get field for position of elevGrid so you can change it right before the other changes are made.
    elevTransField = elevNode.getField("translation")
    print("current trans pos of elevGrid: " + str(elevTransField.getSFVec3f))
    #updatePos = ([float(-2.32), float(1.5), float(-0.9)])
    updatePos = globalElevGridPos
    elevTransField.setSFVec3f(updatePos)
    
    # Ranges for random values for the x and z dimensions
    dimRandRangeMin = 1
    dimRandRangeMax = 5
    
    # Range for random heightfield heights
    # Values might be weakend a bit to decrease height extremities.
    elevMin = 9 # Originally 1 # Consider changing this value from 1 to something else at some point. 
    elevMax = 54 # Originally 100 # The actual heights won't go up to 100, as later the random value will be divided by 100 to get a decimal number
    
    # Random values for the x and z dimensions
    randX = random.randint(dimRandRangeMin, dimRandRangeMax)
    global globalXDim
    globalXDim = randX
    randZ = random.randint(dimRandRangeMin, dimRandRangeMax)
    global globalZDim
    globalZDim = randZ
    print("randX: " + str(randX))
    print("randZ: " + str(randZ))
    
    # Random values for the x- and zSpacing
    # These spaceRange values have been made a little less extreme so the elevGrid isn't too extreme with resepct to spacing.
    spaceRandRangeMax = 85 # Originally 100
    spaceRandRangeMin = 50 # Originally 4
    randSpaceX = (random.randint(spaceRandRangeMin, spaceRandRangeMax) / 100) * 2 # Multiply by 2 so the range should be around 0-2?
    global globalXSpace
    globalXSpace = randSpaceX
    randSpaceZ = (random.randint(spaceRandRangeMin, spaceRandRangeMax) / 100) * 2
    global globalZSpace
    globalZSpace = randSpaceZ
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
    
# --------------------------------------------------------------------------

#----------------------------Terrain Randomization--------------------------
def getStartEnd(platformNode, sizeX, sizeZ, elev): # gets start and end locations of terrain platform
    desiredRobotSpace = 0.75 # length of space dedicated to the start and end positions of the terrain, in this case the length along the z-axis.
    
    floorTransField = platformNode.getField("translation")
    #elevPos = ([float(-2.5), float(1.5), float(-2)])
    floorX = floorTransField.getSFVec3f()[0]
    print("floorX: " + str(floorX))
    floorY = floorTransField.getSFVec3f()[1]
    floorZ = floorTransField.getSFVec3f()[2]
    
    
    startX = floorX
    startY = 0
    if (elev): # if elevgrid spawned, then make sure robot spawns a little bit higher so that it doesnr't instantly collide with the elevgrid
        startY = floorY + 1.6 # not sure what value is best suited to add here.
    else:
        startY = floorY + 0.6  
    startZ = floorZ + (sizeZ / 2) - (desiredRobotSpace / 2)
    
    endX = floorX 
    endY = floorY + 0.6 # kinda arbitrary, as long as it's around the correct height.
    endZ = floorZ - (sizeZ / 2) + (desiredRobotSpace / 2)
    
    startPos = [float(startX), float(startY), float(startZ)]
    endPos = [float(endX), float(endY), float(endZ)]
    
    globalEndPos = endPos
    
    posList = [startPos, endPos]
    
    return posList
    
# TODO: different physical parameters for the obstacles
def terRand(terX, terZ, obstCount, elev, elevRand, slantTerrainYeNo, slantParamz):
    """
    Input parameters
    -terX and terZ: terrain floor x and z dimensions.
    -obstCount: for now the number of obstacle cubes to spawn.
    -elev: if true then spawn elevGrid, else don't spawn elevGrid.
        """
    
    rangeMin = -20 # random position generator limits.
    rangeMax = 20
    childrenIndex = 0 # index used for importing objects (e.g. importMFNode(childrenIndex, path)).
    
    relRobotPath = "../../protos/secondVersionWithMotorsCOMChanged.wbo" # robot relative path
    relCubePath = "../../protos/obstCube.wbo" # small cube relative path
    transNode = superMan.getFromDef("randTerrainHolder") # get a handle to the translation node
    childField = transNode.getField("children") # get access to children field of transNode
    
    rootKids = superMan.getRoot().getField('children') # get handle to root's children
    rootKidsIndex = 0 # same as childrenIndex, but for the rootKids.
    
    # set size of terrain floor. (here we're dealing with the geometry node of the floor node, not the actual floor itself)
    floorGeomNode = superMan.getFromDef("boxGeom1")
    floorGeomSizeField = floorGeomNode.getField("size")
    terY = 1 # originally 0.1
    newSize = ([float(terX), float(terY), float(terZ)]) 
    floorGeomSizeField.setSFVec3f(newSize)
    
    # Get coords of where to put elevGrid relative to the terrianFloor
    floorNode = superMan.getFromDef("terrainFloor")
    transField = floorNode.getField("translation")
    xCoord = transField.getSFVec3f()[0] - (terX / 3) 
    yCoord = transField.getSFVec3f()[1] + 0.2
    zCoord = transField.getSFVec3f()[2] - (terZ / 3) 
    globalElevGridPos = ([float(xCoord), float(yCoord), float(zCoord)])
    
    # Calculate start & end locations of the platform
    floorNode = superMan.getFromDef("terrainFloor")
    startEnd = getStartEnd(floorNode, terX, terZ, elev)
    startPos1 = startEnd[0]
    endPos1 = startEnd[1] 
    print("startPos1: " + str(startPos1))
     
    # If desired, change terrainFloor angle accordingly
    if (slantTerrainYeNo):
        rotField = floorNode.getField("rotation")
        rotField.setSFRotation(slantParamz) 
     
    
    indexUpdated = False # if false then the childrenIndex is updated in the upcoming loop.
    
    # loop to spawn obstCount number of obstacles in random places.
    if (obstCount > 0):
        for x in range(obstCount):
           if (indexUpdated == False): # update children index if necessary
               childrenIndex += obstCount
               indexUpdated = True
           # import the node/object
           childField.importMFNode(x, relCubePath) # 1st parameter was 0 originally, not x.
           # get a handle to the new node/object
           latestAddedNode = childField.getMFNode(-1)
           # get a handle to the translation field of the new object
           objectTransField = latestAddedNode.getField("translation")
           # change translation field of newly added object randomly.
           randX = random.randint(rangeMin, rangeMax)
           randZ = random.randint(rangeMin, rangeMax)
           randPos1 = ([float(randX / 10), float(1.07), float(randZ / 10)])
           print("randPos: " + str(randPos1))
           objectTransField.setSFVec3f(randPos1) 
           #print("cubePos: " + objectTransField.getSFVec3f)
     
    print("childrenIndex: " + str(childrenIndex))
    
    # if requested, create elevGrid
    if (elev): 
        elevPath = "../../protos/elevGrid.wbo"
        childField.importMFNode(childrenIndex, elevPath)
        childrenIndex += 1 # increment childrenIndex cause something was spawned
        
        # set elevGrid position
        elevNode1 = childField.getMFNode(-1)
        elevTransField = elevNode1.getField("translation")
        elevPos = globalElevGridPos 
        elevTransField.setSFVec3f(elevPos)
        
        # If desired, randomize the elevGrid.
        if (elevRand):
            randElevGrid(elevNode1)
        
        """
        # fit elevgrid to the terrainfloor size
        elevgridGeomNode = superMan.getFromDef("elevgridGeom")
        #xDimField = elevgridGeomNode.getField("xDimension")
        #xDim = xDimField.getSFInt32()
        xDim = globalXDim
        #zDimField = elevgridGeomNode.getField("zDimension")
        #zDim = zDimField.getSFInt32()
        zDim = globalZDim
        #xSpaceField = elevgridGeomNode.getField("xSpacing")
        #xSpace = xSpaceField.getSFFloat()
        xSpace = globalXSpace
        #zSpaceField = elevgridGeomNode.getField("zSpacing")
        #zSpace = zSpaceField.getSFFloat()
        zSpace = globalZSpace
        print("xSpace: " + str(xSpace))
        print("zSpace: " + str(zSpace))
        # calculate current elevgrid size & then change size as required to be as big as the terrainfloor
        xLength = xDim * xSpace
        zLength = zDim * zSpace
        print("xLength: " + str(xLength))
        print("zLength: " + str(zLength))
        
        #terZ terX are terrainFloor lengths
        scaleX = terX / xLength # terX & terZ = 15
        scaleZ = terZ / zLength 
        print("scaleX: " + str(scaleX))
        print("scaleZ: " + str(scaleZ))
        # get handle to Heightfield1 solid
        #hFieldNode = superMan.getFromDef("Heightfield1")
        #scaleField = hFieldNode.getField("scale")
        scaleUno = 0
        if (scaleX > scaleZ):
            scaleUno = scaleX
        else:
            scaleUno = scaleZ
        """     
        scaleUno = 4
        scaleField = elevNode1.getField("scale")
        scales = [scaleUno, scaleUno, scaleUno] # all 3 scale values have to be the same in this case, hence the y-value is set to the x-value here.
        scaleField.setSFVec3f(scales)
        elevTransField.setSFVec3f(elevPos)
        
        
        
    
    # Set robot position to start of terrain
    robotNode = superMan.getFromDef("secondVersionWithMotors")
    transFieldUno = robotNode.getField("translation")
    transFieldUno.setSFVec3f(startPos1)
    
    
    # Cannot spawn robot at run time for now because of the centre of mass issue.
    """ 
    # spawn robot. I would spawn it into the transform's children field, however, it is prohibited to spawn robot nodes into 'children' field of a transfrom node.
    rootKids.importMFNode(rootKidsIndex, relRobotPath)
    rootKidsIndex += 1 # increment index after spawning object
    #robotNode = rootKids.getMFNode(-1)
    robotNode = rootKids.getMFNode(0) # normally -1 for most recent, but spawning the robot into root field makes the robot appear in the 1st position in the node tree, rather than the last.
    
    robotTransField = robotNode.getField("translation")
    
    robotTransField.setSFVec3f(startPos1) 
    print("position set!")
    """
    
    """
    EXAMPLE OF CHANGE DEF UPON IMPORT
    wb_supervisor_field_import_mf_node_from_string(root_children_field, 4, "DEF MY_ROBOT Robot { controller "my_controller" }");
    probably won't use this, instead we'll use +-fieldgetmfnode(-1), which gets the last imported node.
    """
    
    return
#----------------------------------------------------------------------------

# get the time step of the current world.
timestep = int(superMan.getBasicTimeStep())

maxSpeed = 6.28

#---------------------------------Function Calls & such------------------------------
# Function calls and other stuff, comment out a function call to stop the function (obviously)
# terRand() related
obstCount = 0 # default 10
#obstSizes
terrainX = 15 # default 2, three times each default value for the main platform size is cool, maybe make the shorter side even shorter though?
terrainZ = 15 # defualt 5
elevGridYesNo = True # if True, spawns elevgrid in the terRand function
elevRandYesNo = True # If True, randomize the elevGrid in the terRand function
slantTerrain = False # If true, then the terraingrid will be slanted by the angle and axes given below
slantParams = [1, 0, 0, 0.1] # Parameters to slant terrain with. The first 3 values in this list must be normalized!!!
#elevGridSlope

terRand(terrainX, terrainZ, obstCount, elevGridYesNo, elevRandYesNo, slantTerrain, slantParams)

#-----------------------------------------------------------------------------


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while superMan.step(timestep) != -1:
    
    # world reload if desired
    if (superMan.getTime() > reloadTime):
        print("Reload")
        #superMan.worldReload()
    
    pass # used when a statement is required  syntactically, but you don't want any command or code to execute.

# Enter here exit cleanup code.
