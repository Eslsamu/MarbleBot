import xml.etree.ElementTree as ET
tree = ET.parse(
    '/Users/cdalenbrook/MarbleBot/sim/morphology/quad_world_slip.xml')
root = tree.getroot()

doc_path = '/Users/cdalenbrook/MarbleBot/sim/morphology/quad_world_slip.xml'

#partToRemove = input("What DOF would you like removed? (Options: hip, side, slide) ")
# remove_DOF(partToRemove)


def write(path):
    tree.write(path)


def remove_DOF(toRemove):
    partToRemove = toRemove
    # find joint in file and remove it
    for worldbody in root.findall('worldbody'):
        for body in worldbody.findall('body'):
            for body2 in body.findall('body'):
                # hip and side joints
                for joint in body2.findall('joint'):
                    if(partToRemove in joint.get('name')):
                        body2.remove(joint)
                        print('Removed: ', joint.get('name'))
                        print(joint)
                # slide joints
                for body3 in body2.findall('body'):
                    for joint in body3.findall('joint'):
                        if(partToRemove in joint.get('name')):
                            body3.remove(joint)
                            print('Removed: ', joint.get('name'))

    # find motor with joint name and remove that too
    for actuator in root.findall('actuator'):
        for motor in actuator.findall('motor'):
            if(motor.get('joint') != None):
                if(partToRemove in motor.get('joint')):
                    actuator.remove(motor)
                    print('Removed motor: ', motor.get('joint'))
    write(doc_path)


# array of hip joints
hip_1 = ET.fromstring(
    '<joint axis="0 1 0" name="hip_1" pos="0.0 0.0 0.0" range="-90 90" type="hinge" />')
hip_2 = ET.fromstring(
    '<joint axis="0 1 0" name="hip_2" pos="0.0 0.0 0.0" range="-90 90" type="hinge" />')
hip_3 = ET.fromstring(
    '<joint axis="0 1 0" name="hip_3" pos="0.0 0.0 0.0" range="-90 90" type="hinge" />')
hip_4 = ET.fromstring(
    '<joint axis="0 1 0" name="hip_4" pos="0.0 0.0 0.0" range="-90 90" type="hinge" />')
hips = [hip_1, hip_2, hip_3, hip_4]

# array of hip motors
motor_1 = ET.fromstring(
    '<motor ctrllimited="true" ctrlrange="-5.0 5.0" joint="hip_1" />')
motor_2 = ET.fromstring(
    '<motor ctrllimited="true" ctrlrange="-5.0 5.0" joint="hip_2" />')
motor_3 = ET.fromstring(
    '<motor ctrllimited="true" ctrlrange="-5.0 5.0" joint="hip_3" />')
motor_4 = ET.fromstring(
    '<motor ctrllimited="true" ctrlrange="-5.0 5.0" joint="hip_4" />')
hip_motors = [motor_1, motor_2, motor_3, motor_4]

# array of side joints
side_1 = ET.fromstring(
    '<joint axis="1 0 0" name="side_1" pos="0.0 0.0 0.0" range="-10 10" type="hinge" />')
side_2 = ET.fromstring(
    '<joint axis="1 0 0" name="side_2" pos="0.0 0.0 0.0" range="-10 10" type="hinge" />')
side_3 = ET.fromstring(
    '<joint axis="1 0 0" name="side_3" pos="0.0 0.0 0.0" range="-10 10" type="hinge" />')
side_4 = ET.fromstring(
    '<joint axis="1 0 0" name="side_4" pos="0.0 0.0 0.0" range="-10 10" type="hinge" />')
sides = (side_1, side_2, side_3, side_4)

# array of side motors
motor_side_1 = ET.fromstring(
    '<motor ctrllimited="true" ctrlrange="-5.0 5.0" joint="side_1" />')
motor_side_2 = ET.fromstring(
    '<motor ctrllimited="true" ctrlrange="-5.0 5.0" joint="side_2" />')
motor_side_3 = ET.fromstring(
    '<motor ctrllimited="true" ctrlrange="-5.0 5.0" joint="side_3" />')
motor_side_4 = ET.fromstring(
    '<motor ctrllimited="true" ctrlrange="-5.0 5.0" joint="side_4" />')
side_motors = (motor_side_1, motor_side_2, motor_side_3, motor_side_4)

# array of slide joints
slide_1 = ET.fromstring(
    '<joint axis="0 0 1" limited="false" name="slide1" type="slide" />')
slide_2 = ET.fromstring(
    '<joint axis="0 0 1" limited="false" name="slide2" type="slide" />')
slide_3 = ET.fromstring(
    '<joint axis="0 0 1" limited="false" name="slide3" type="slide" />')
slide_4 = ET.fromstring(
    '<joint axis="0 0 1" limited="false" name="slide4" type="slide" />')
slides = (slide_1, slide_2, slide_3, slide_4)


def add_DOF(toAdd):
    if(toAdd == 'hip'):
        # add only if they're not already there
        for worldbody in root.findall('worldbody'):
            for body in worldbody.findall('body'):
                x = 0
                for body2 in body.findall('body'):
                    for joint in body2.findall('joint'):
                        if toAdd in joint.get('name'):
                            print(joint.get('name'),
                                  'DOF is already currently added.')
                            break
                        else:
                            # add hip joints
                            if('upper' in body2.get('name')):
                                body2.append(hips[x])
                                print('Added: Hip joint', str(x+1))
                            # add hip motors
                            for actuator in root.findall('actuator'):
                                actuator.append(hip_motors[x])
                                print('Added: Hip motor', str(x+1))
                    x += 1

    elif(toAdd == 'side'):
        for worldbody in root.findall('worldbody'):
            for body in worldbody.findall('body'):
                y = 0
                for body2 in body.findall('body'):
                    for joint in body2.findall('joint'):
                        if(toAdd in joint.get('name')):
                            print(joint.get('name'),
                                  'DOF is already currently added.')
                            break
                        else:
                            # add side joints
                            if('upper' in body2.get('name')):
                                body2.append(sides[y])
                                print('Added: Side joint', str(y+1))
                            # add side motors
                            for actuator in root.findall('actuator'):
                                actuator.append(side_motors[y])
                                print('Added: Side motor', str(y+1))
                    y += 1

    elif(toAdd == 'slide'):
        for worldbody in root.findall('worldbody'):
            for body in worldbody.findall('body'):
                z = 0
                for body2 in body.findall('body'):
                    for body3 in body2.findall('body'):
                        # add joint
                        body3.append(slides[z])
                        z += 1
                        print('Added: Slide joint', str(z))
                        # if there is 2 joints now remove one and alert user that DOF is already added
                        joint_count = 0
                        for joint in body3.findall('joint'):
                            joint_count += 1
                        if joint_count > 1:
                            print(toAdd + 'DOF is currently already added')
                            body3.remove(joint)

    else:
        print('Please enter a valid DOF to add.')

    write(doc_path)

# length should be the total length of the leg


def change_leg_length(length):
    start1 = '0.0 0.0 0.0 0.0 0.0 '
    newFromTo = start1 + str(-length/2)
    start2 = '0 0 '
    newPos = start2 + str(-length/2)
    for worldbody in root.findall('worldbody'):
        for body in worldbody.findall('body'):
            for body2 in body.findall('body'):
                for geom in body2.findall('geom'):
                    geom.set('fromto', newFromTo)
                for site in body2.findall('site'):
                    site.set('pos', newPos)
                for body3 in body2.findall('body'):
                    body3.set('pos', newPos)
                    for geom in body3.findall('geom'):
                        geom.set('fromto', newFromTo)
                    for site in body3.findall('site'):
                        if('touchSite' in site.get('name')):
                            site.set('pos', newPos)
    print('Length of legs changed to: ', str(length))
    write(doc_path)
