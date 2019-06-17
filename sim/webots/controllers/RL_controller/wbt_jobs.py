import subprocess
import pickle
from os import listdir
import time
import logging
import numpy as np
logging.basicConfig(format='%(asctime)s %(message)s',filename='server.log',level=logging.DEBUG)

INIT_FILE = "../../worlds/speed_test.wbt"
INSTANCE_DEST = "../../instances/world"
COUNTER_DEST = "counter/c"
EPOCH_DATA_FOLDER = "epoch_data"

"""
copies the source world file and writes the controller arguments
"""
def create_worldfile(id, args, src = INIT_FILE, dest = INSTANCE_DEST):
    #turns all arguments into strings with quotation and concats them
    args = ' '.join([str(a) for a in args])

    dest = dest + str(id) + '.wbt'

    with open(dest, 'w') as outfile, open(src, 'r') as infile:
        for line in infile:
            if "controllerArgs" in line:
                line = line.replace("None",args)
            outfile.write(line)

    return dest

"""
loads the data of all timesteps that an episode gathered in an epoch
"""
def load_proc_data(id, folder = EPOCH_DATA_FOLDER):
    files = [f for f in listdir(folder) if "proc"+str(id)+"-" in f]
    proc_data = []
    for f in files:
        epi_data = pickle.load(open(folder+"/"+f, "rb"))
        proc_data.append(epi_data)
    return proc_data

"""
creates a specified amount of webots instances running for a given amount of total iterations
@param build_files: true if new controller arguments have to be specified for instances 
or new copies of world files have to be made
"""
def run_job(n_proc, total_steps, max_ep_steps, model_file, build_files = False, data_dir = EPOCH_DATA_FOLDER, count_file = COUNTER_DEST, instdir = INSTANCE_DEST):
    steps_per_process = int(n_proc/total_steps)
    extra_steps = n_proc % total_steps

    if build_files:
        worldfiles = []
        for p in range(n_proc):
            args = [max_ep_steps, data_dir+"/proc"+str(p)+"-", count_file + str(p) + '.p', model_file]
            f = create_worldfile(p, args)
            worldfiles.append(f)
    else:
        worldfiles = [instdir + str(p) for p in range(n_proc)]

    children = []
    for p in range(n_proc):
        with open(count_file + str(p) + '.p', 'wb') as file:
            #modulo steps for first process
            if p == 0 :
                pickle.dump(steps_per_process+extra_steps, file)
            else:
                pickle.dump(steps_per_process, file)
            children.append(subprocess.Popen(["webots --mode=fast --minimize --batch " + worldfiles[p]], shell=True))

    try:
        #constantly check process to terminate and retrieve simulation data if it did
        #stop when all processes are done
        epoch_data = []
        fin = []
        while len(epoch_data) < n_proc:
            for i, c in enumerate(children):
                if i not in fin:
                    done = c.poll()
                    if done:
                        print(i, "done")
                        proc_data = (load_proc_data(id=i), i)
                        epoch_data.append(proc_data)
                        fin.append(i)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    finally:
        print("data", len(epoch_data), np.array(epoch_data).shape) #,epoch_data)
        for c in children:
            try:
                c.kill()
            except Exception:
                pass

    return epoch_data


t = time.time()
run_job(n_proc = 10, n_it = 10, n_steps= 500, build_files=True)
print("time:", time.time()-t)