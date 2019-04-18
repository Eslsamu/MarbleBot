import tensorflow as tf
from rl_policy import actor_critic 
import epuck_controller as controller
from advantage_estimation import Buffer
import numpy as np
import time


def ppo(epochs=2,epoch_steps = 4000 , max_ep_len=1000 ,pi_lr = 3e-4, vf_lr=1e-3,gamma=0.99,lam=0.97,pi_iters = 80,target_kl = 0.01,val_iters=80, clip_ratio=0.2):
    print("init robot..")
    print(1, time.time())
    robot = controller.Robot_Environment()
    if robot is not None:
        print("robot initialized: ", robot)
    
    act_dim = len(robot.motors)
    obs_dim = len(robot.sensors)
    print("action_space: ", act_dim) 
    print("obs_space: ", obs_dim) 
    
    
    #tensorflow graph inputs
    obs_ph = tf.placeholder(dtype = tf.float32, shape=(None,obs_dim))
    act_ph = tf.placeholder(dtype = tf.float32, shape=(None,act_dim))
    adv_ph = tf.placeholder(dtype = tf.float32, shape=(None,))
    ret_ph = tf.placeholder(dtype = tf.float32, shape=(None,))
    logp_old_ph = tf.placeholder(dtype = tf.float32, shape=(None,))
    graph_inputs = [obs_ph, act_ph, adv_ph, ret_ph, logp_old_ph]


    #tensorflow graph outputs
    pi, logp, logp_pi, val = actor_critic(obs_ph, act_ph)

    #experience buffer
    buf = Buffer(obs_dim, act_dim, epoch_steps, gamma, lam)

    #loss functions
    ratio = tf.exp(logp - logp_old_ph)      #pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - val)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    #optimizer
    opt_pi = tf.train.AdamOptimizer(learning_rate = pi_lr).minimize(pi_loss)
    opt_val = tf.train.AdamOptimizer(learning_rate = vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #gradient update
    def update():
        #create input dictionary from trajector and graph inputs
        inputs =  {g:t for g,t in zip(graph_inputs,buf.get())}

        pi_l_old, val_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        #Training
        for i in range(pi_iters):
            _, kl = sess.run([opt_pi, approx_kl], feed_dict=inputs)
            kl = np.mean(kl)
            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break

        #policy gradient step
        sess.run(opt_pi, feed_dict=inputs)

        #value function training
        #t = time.time()
        for i in range(val_iters):
            sess.run(opt_val, feed_dict=inputs)
       # print("value time: ", time.time()-t)

        pi_l_new, val_l_new = sess.run([pi_loss, v_loss], feed_dict=inputs)
        """
        print("policy loss:", pi_l_new)
        print("value func loss:", val_l_new)
        print("delta policy loss: ",pi_l_new-pi_l_old)
        print("delta value func loss: ",val_l_new-val_l_old)
        """


    obs, done, rew, ep_ret, ep_len = robot.reset() , False , 0 , 0, 0

    #visualize
    render = False

    #experience and training loop
    for epoch in range(epochs):
        traj_start = 0
        epoch_ret = []
        first_episode = True
        
        e = time.time()
        for t in range(epoch_steps):
            obs = obs.reshape(1,-1)
            a, v_t, logp_t = sess.run([pi, val, logp_pi], feed_dict={obs_ph: obs})

            #save traj info
            buf.store(obs,a,rew,v_t,logp_t)
            rew ,obs , done, = robot.step(a[0])
            ep_ret += rew
            ep_len += 1

            terminated = done or (ep_len == max_ep_len)
            if terminated or (t==epoch_steps-1):
                if not terminated:
                    print("traj cut off")
                else:
                    epoch_ret.append(ep_ret)
                    if first_episode:
                        print("done after steps: ", ep_len)
                        print("episode return: ", ep_ret)
                        # only render first episode
                        first_episode = False



                last_val = rew if done else sess.run(val, feed_dict={obs_ph: obs.reshape(1,-1)})
                buf.finish_path(last_val)

                obs, done, rew, ep_ret, ep_len = robot.reset(),False, 0, 0 , 0
        print("--------------")
        print("epoch: ", epoch)
        print("average return: ", np.mean(epoch_ret) )
        print("max return: ", max(epoch_ret) )
        print("min return: ", min(epoch_ret) )
        print("episodes: ",len(epoch_ret))
        print("experience time: ",time.time()-e)

        #gradient update
        t = time.time()
        update()
        print("update time:", time.time()-t)

ppo()