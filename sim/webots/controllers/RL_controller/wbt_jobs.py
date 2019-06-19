import subprocess
import pickle
from os import listdir
import logging
import numpy as np
from shutil import rmtree
import os.path as osp
from os import makedirs
logging.basicConfig(format='%(asctime)s %(message)s',filename='server.log',level=logging.DEBUG)

INIT_FILE = "../../worlds/RL_world.wbt"
INSTANCE_DEST = "../../instances/world"
COUNTER_DEST = "counter"
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
def run_job(n_proc, total_steps, max_ep_steps, model_path, build_files = False, data_dir = EPOCH_DATA_FOLDER, count_dir = COUNTER_DEST, instdir = INSTANCE_DEST):
    steps_per_process = int(total_steps/n_proc)
    extra_steps = total_steps % n_proc

    # delete epoch data and counter directory if exists
    if osp.isdir(data_dir):
        rmtree(data_dir)
    if osp.isdir(count_dir):
        rmtree(count_dir)

    #creates these directories
    makedirs(data_dir)
    makedirs(count_dir)

    if build_files:
        worldfiles = []
        for p in range(n_proc):
            args = [max_ep_steps, data_dir+ "/proc" + str(p) + "-", count_dir  + "/c" + str(p) + '.p', model_path]
            f = create_worldfile(p, args)
            worldfiles.append(f)
    else:
        worldfiles = [instdir + str(p) for p in range(n_proc)]

    children = []
    for p in range(n_proc):
        with open(count_dir + "/c" + str(p) + '.p', 'wb') as file:
            #modulo steps for first process
            if p == 0 :
                pickle.dump(steps_per_process+extra_steps, file)
            else:
                pickle.dump(steps_per_process, file)
            children.append(subprocess.Popen(["webots --stdout --stderr --mode=fast --minimize --batch " + worldfiles[p]], shell=True))

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
                        proc_data = load_proc_data(id=i)
                        epoch_data.append(proc_data)
                        fin.append(i)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    finally:
        for c in children:
            try:
                c.kill()
            except Exception:
                pass

    return epoch_data

