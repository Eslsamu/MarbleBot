from tensorflow import Session, Graph, saved_model
import sys
from webots.controllers.robot_environment import Robot_Environment
from controller import Supervisor
import pickle
import logging
logging.basicConfig(filename='ppo.log',level=logging.DEBUG)



"""
run one episode until it terminates, the epoch is over or it reached the max time steps
"""
def run_episode(env, sess, total_steps, max_ep_steps):
    #initialize robot
    env.init()

    #episode data
    ep_obs = []
    ep_a = []
    ep_vals = []
    ep_rews = []
    ep_logp =[]
    obs, rew, done, ep_ret, ep_len = env.get_obs(), 0, False, 0, 0

    #cut of episode if total process steps are reached
    if total_steps < max_ep_steps:
        n_steps = total_steps
        #TODO log cutting episode off here
    else:
        n_steps = max_ep_steps

    #episode loop
    for s in range(n_steps):
        action, val, logp = sess.run(["pi", "val"], feed_dict={"obs": obs.reshape(1, -1)})
        obs, rew, done = env.step(action)

        #store timestep data
        ep_ret += rew
        ep_len += 1
        ep_vals.append(val)
        ep_obs.append(obs)
        ep_a.append(action)
        ep_rews.append(rew)
        ep_logp.append(logp)

        #terminate
        if done:
            break

    last_val = rew if done else sess.run("val", feed_dict={"obs": obs.reshape(1, -1)})

    return {"obs": ep_obs, "a": ep_a,"val":ep_vals, "rew": ep_rews, "logp": ep_logp,
            "ret": ep_ret,"len": ep_len, "last_val":last_val}


#read controller arguments
max_ep_steps = int(sys.argv[1])
data_dir = sys.argv[2]
count_file = sys.argv[3]
model_dir = sys.argv[4]

#load the saved tensorflow model as session
with Session(graph=Graph()) as sess:
  saved_model.loader.load(sess, [saved_model.tag_constants.TRAINING], model_dir)

#get steps to go for this epoch
steps_to_go = pickle.load(open(count_file, "rb"))

#create robot environment
env = Robot_Environment()

#run episode and collect episode data
ep_data = run_episode(env, sess, steps_to_go, max_ep_steps)

#save episode data
pickle.dump(ep_data, open(data_dir+str(steps_to_go), "wb"))

#reset simulation or quit if epoch is over
sv = Supervisor()
if steps_to_go == 0:
    sv.simulationQuit(1)
else:
    ep_len = ep_data.get("len")
    steps_to_go = steps_to_go - ep_len
    pickle.dump(steps_to_go,open(count_file, "wb"))
    sv.simulationReset()

