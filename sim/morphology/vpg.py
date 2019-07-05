#this VPG implementation is heaviliy influenced from the spinningup course of OpenAi

import environment
import numpy as np
import tensorflow as tf
import scipy.signal
import time

env = environment.RoboEnv('model/quad_world_knee2.xml')

#noise constant 
EPS = 1e-8

#cumulative sum
def disc_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

#log likelihood of x given normal distribution(mean,log_std)
def log_likelihood(x, mean, log_std):
   prob = -0.5 * (( (x-mean)/(tf.exp(log_std)+EPS))**2 + 2 * log_std + np.log(2*np.pi))
   return tf.reduce_sum(prob, axis=1)

def nn(inp, hidden_sizes=(8,), activation=tf.tanh):
    for h in hidden_sizes[:-1]:
        inp = tf.layers.dense(inp, units=h, activation=activation)
    output = tf.layers.dense(inp,units=hidden_sizes[-1], activation = None)
    return output

def policy(state, action, hidden_sizes):
    act_dim = action.shape.as_list()[-1]
    mean = nn(state, list(hidden_sizes)+[act_dim])

    #take log standard deviation because it can take any values in (-inf, inf) -> easier to train without constraints 
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)

    #samples actions from policy given a state
    pi = mean + tf.random_normal(tf.shape(mean))*std

    #gives log probability of taking 'actions' according to the policy in states
    logp = log_likelihood(action, mean, log_std)

    #gives log probability according to the policy of the actions sampled by pi
    logp_pi = log_likelihood(pi, mean, log_std)
    return pi, logp, logp_pi

def actor_critic(state, action, hidden_sizes=(64,64)):

    #policy 
    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(state, action, hidden_sizes)

    #state value estimation network
    with tf.variable_scope('v'):
            val = tf.squeeze(nn(state, list(hidden_sizes)+[1]), axis = 1)

    return pi, logp, logp_pi, val

class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = disc_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = disc_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr
        
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_var = np.var(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_var**0.5


        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]



def vpg(epochs=500, epoch_steps = 4000, max_ep_len=1000 ,pi_lr = 3e-4, vf_lr=1e-3,gamma=0.99,lam=0.97,pi_iters = 80,target_kl = 0.01,val_iters=80, clip_ratio=0.2):
    act_dim = env.act_shape[0]
    obs_dim = env.state_shape[0]

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
    buf = VPGBuffer(obs_dim, act_dim, epoch_steps, gamma, lam)

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
                #print('Early stopping at step %d due to reaching max kl.'%i)
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


    obs, done, rew, ep_ret, ep_len = env.reset(),False , 0 , 0, 0


    #visualize
    render = False

    #experience and training loop
    for epoch in range(epochs):
        traj_start = 0
        epoch_ret = []

        first_episode = True

        for t in range(epoch_steps):
            obs = obs.reshape(1,-1)
            a, v_t, logp_t = sess.run([pi, val, logp_pi], feed_dict={obs_ph: obs})

            #save traj info
            buf.store(obs,a,rew,v_t,logp_t)

            if render and first_episode:
                env.render()

            rew ,obs , done, = env.step(a[0])
            ep_ret += rew
            ep_len += 1

            terminated = done or (ep_len == max_ep_len)
            if terminated or (t==epoch_steps-1):
                if not terminated:
                    #print("traj cut off")
                    continue
                else:
                    epoch_ret.append(ep_ret)

                    if first_episode:
                        print("done after steps: ", ep_len)
                        print("episode return: ", ep_ret)
                        # only render first episode
                        first_episode = False

                last_val = rew if done else sess.run(val, feed_dict={obs_ph: obs.reshape(1,-1)})
                buf.finish_path(last_val)

                obs, done, rew, ep_ret, ep_len = env.reset(),False, 0, 0 , 0
       
        # OUTPUT INFO IN COMMAND LINE
        # print("--------------")
        print("epoch: ", epoch)
        # print("average return: ", np.mean(epoch_ret) )
        # print("max return: ", max(epoch_ret) )
        # print("min return: ", min(epoch_ret) )

        #Print info of last epoch to text file
        if epoch == len(range(epochs))-1:
            l1 = '\n--------------'
            l2 = '\nepoch: ' + str(epoch)
            l3 = '\naverage return: ' + str(np.mean(epoch_ret))
            l4 = '\nmax return: ' + str(max(epoch_ret))
            l5 = '\nmin return: ' + str(min(epoch_ret))
            file = open("output.txt","a")
            file.write(l1)
            file.write(l2)
            file.write(l3)
            file.write(l4)
            file.write(l5)
            file.close()


        #gradient update
        update()


    # test speed sim result: is similar to spinningup implementation
    """
    s = []
    for i in range(100):
        t = time.time()
        obs = env.reset()
        for _ in range(10):
            obs = obs.reshape(1, -1)
            a = sess.run(pi, feed_dict={obs_ph: obs})
            rew, obs, done, = env.step(a[0])
        s.append(t - time.time())
    print("speed: ", np.mean(s))
    """

    #demonstrate learned policy
    # obs = env.reset()
    # ep_ret = 0
    # while True:
    #     for i in range(max_ep_len):
    #         obs = obs.reshape(1, -1)
    #         a = sess.run(pi, feed_dict={obs_ph: obs})
    #         env.render()
    #         rew, obs, done, = env.step(a[0])
    #         ep_ret += rew
    #         if done:
    #             ep_ret = 0
    #             obs=env.reset()
    #             break

vpg()


