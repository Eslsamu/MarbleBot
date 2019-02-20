#!/usr/bin/env python3
import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
#p.setJointMotorControlArray #Added by you, used because using MJCF file.
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1] 
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadMJCF("models/quad_world_slip.xml")
arr1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
p.setJointMotorControlArray(boxId[2], arr1, 0)
#for i in range (3): 
    #print(p.getNumJoints(boxId[i])) # Get number of joints. Currently 20.
    #p.setJointMotorControlArray(boxId[i], arr1, 1)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
    #time.sleep(1./10.) slow-mo
#cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
#print(cubePos,cubeOrn)
p.disconnect()

