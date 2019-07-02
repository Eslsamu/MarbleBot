from controller import Supervisor
from tensorflow import Session, Graph, saved_model
import sys
from robot_environment import Robot_Environment
import pickle
import joblib
import os.path as osp
import logging
import numpy as np
logging.basicConfig(filename='env.log', format='%(asctime)s %(message)s', level=logging.INFO)

"""
run one episode until it terminates, the epoch is over or it reached the max time steps
"""
def run_episode(env, sess, inp, out, total_steps, max_ep_steps):
    #episode data
    ep_obs = []
    ep_a = []
    ep_vals = []
    ep_rews = []
    ep_logp =[]
    ep_ene = 0
    ep_clipped = 0
    ep_dist = 0
    ep_abs_dist = 0
    obs, rew, done, ep_ret, ep_len = env.get_sensor_data(), 0, False, 0, 0

    #cut of episode if total process steps are reached
    if total_steps < max_ep_steps:
        n_steps = total_steps
        logging.info("episode will cut off after " + str(total_steps) + " steps")
    else:
        n_steps = max_ep_steps

    logging.info("has to do steps " + str(n_steps))

    #episode loop
    for s in range(n_steps):
        #flatten sensor data
        obs = obs.reshape(1, -1)

        # network inferences
        action, val_t, logp_t = sess.run(out, feed_dict={inp: obs})
        action, val_t, logp_t = action[0], val_t[0], logp_t[0]  # tf output is array

        # store timestep data
        ep_ret += rew
        ep_len += 1
        ep_vals.append(val_t)
        ep_obs.append(obs)
        ep_a.append(action)
        ep_rews.append(rew)
        ep_logp.append(logp_t)

        #agent-environment interaction
        obs, rew, done, r_info = env.step(action)

        ep_ene += r_info["energy"]
        ep_dist += r_info["distance"]
        ep_abs_dist += np.abs(r_info["distance"])
        ep_clipped += r_info["clipped"]

        #terminate
        if done:
            logging.warning("terminated after steps: " + str(s))
            break

    last_val = rew if done else sess.run(out[1], feed_dict={inp: obs.reshape(1, -1)})

    return {"obs": ep_obs, "a": ep_a,"val":ep_vals, "rew": ep_rews, "logp": ep_logp,
            "ep_ret": ep_ret,"ep_len": ep_len, "last_val":last_val, "ep_ene":ep_ene,
            "ep_dist": ep_dist,"ep_abs_dist": ep_abs_dist,"ep_clipped":ep_clipped}

#inits webots api
sv = Supervisor()

#read controller arguments
max_ep_steps = int(sys.argv[1])
data_dir = sys.argv[2]
model_dir = sys.argv[3]
try:
    count_file = sys.argv[4]
except IndexError:
    count_file = None

#create new session
graph = Graph()
sess = Session(graph=graph)

#restore model
saved_model.loader.load(sess, [saved_model.tag_constants.SERVING], model_dir)

#get restored model ops and tensors
model_info = joblib.load(osp.join(model_dir, 'model_info.pkl'))
model = dict()
model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})

#inputs
inp = model["obs"]

#outputs
pi = model["pi"]
val = model["val"]
logp_pi = model["logp_pi"]
out = [pi, val, logp_pi]

if count_file:
    #get steps to go for this epoch
    steps_to_go = pickle.load(open(count_file, "rb"))
else:
    steps_to_go = max_ep_steps

#create robot environment
env = Robot_Environment(supervisor = sv)

#run episode and collect episode data
ep_data = run_episode(env, sess, inp, out,steps_to_go, max_ep_steps)

#calculate steps for next episode
steps_to_go = steps_to_go - ep_data["ep_len"]

#save episode data
pickle.dump(ep_data, open(data_dir+str(steps_to_go), "wb"))

#reset simulation or quit if epoch is over
if steps_to_go <= 0:
    sv.simulationQuit(1)
else:
    #log episode info
    logging.info("obs: len " + str(len(ep_data["obs"])) + " index0 " + str(ep_data["obs"][0]) + "\n" +
                    "a: len " + str(len(ep_data["a"])) + " index0 " + str(ep_data["a"][0]) + "\n" +
                    "val: len " + str(len(ep_data["val"])) + " index0 " + str(ep_data["val"][0]) + "\n" +
                    "rew: len " + str(len(ep_data["rew"])) + " index0 " + str(ep_data["rew"][0]) + "\n" +
                    "logp: len " + str(len(ep_data["logp"])) + " index0 " + str(ep_data["logp"][0]) + "\n" +
                    "ep ret " + str(ep_data["ep_ret"]) + " ep len " + str(ep_data["ep_len"]) + " last val " +
                    str(ep_data["last_val"]) + "\n" +
                    "stepstogo " + str(steps_to_go) + " countfile " + str(count_file) + " ep file " + str(data_dir+str(steps_to_go))
                    )

    ep_len = ep_data.get("len")
    pickle.dump(steps_to_go,open(count_file, "wb"))
    sv.simulationReset()

