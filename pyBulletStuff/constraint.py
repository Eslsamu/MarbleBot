import pybullet as p
import time
import math

p.connect(p.GUI)

p.loadURDF("models/plane.urdf")
cubeId = p.loadURDF("models/cube_small.urdf",0,0,1)
p.setGravity(0,0,-10)
p.setRealTimeSimulation(1)
cid = p.createConstraint(cubeId, -1, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,1])
p.changeConstraint(cid, maxForce=20)
print (cid)
print (p.getConstraintUniqueId(0))
forceId = p.addUserDebugParameter("force",0,11,10)

prev=[0,0,1]
a=-math.pi
while 1:
        force = p.readUserDebugParameter(forceId)
        p.changeConstraint(cid, maxForce=force)
        s = p.getConstraintState(cid)
        print(s)    

p.removeConstraint(cid)
