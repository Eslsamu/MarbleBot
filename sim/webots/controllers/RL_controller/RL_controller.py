from controller import Supervisor
import sys
import pickle

TIMESTEP = 32

def run_simulation(steps = 100, t = TIMESTEP):
    sim_data = []
    for s in range(steps):
        #obs = robot...
        sim_data.append(s)
        sv.step(t)
    return sim_data
        
        
        
sv = Supervisor()

n_steps = int(sys.argv[1])
data_dir = sys.argv[2]
count_file = sys.argv[3]
sim_data = run_simulation(n_steps)
#TODO append or number
pickle.dump(sim_data, open(data_dir, "wb"))

it = pickle.load(open(count_file, "rb"))

if it == 0:
    sv.simulationQuit(1)
else:
    it -= 1
    print("it", it)
    pickle.dump(it,open(count_file, "wb"))
    sv.simulationReset()

