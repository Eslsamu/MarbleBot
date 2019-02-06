#this VPG implementation is heaviliy influenced from the spinningup course of OpenAi

import environment
import numpy as np
import tensorflow as tf

env = environment.robo_env('model/quad_world.xml')

#noise constant 
EPS = 1e-8

    #log likelihood of x given normal distribution(mean,log_std)
def log_likelihood(x, mean, log_std):
    prob = -0.5 * (( (x-mean)/(tf.exp(log_std)+EPS))**2 + 2 * log_std + np.log(2*np.pi))

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
            val = nn(state, list(hidden_sizes)+[1])
    #maybe need tf.squeeze

    return pi, logp, logp_pi, val


def vpg(epoch_steps = 5000):
    act_dim = env.act_shape[0]
    state_dim = env.state_shape[0]

    #tensorflow graph inputs
    state_ph = tf.placeholder(dtype = tf.float32, shape=(None,state_dim))
    act_ph = tf.placeholder(dtype = tf.float32, shape=(None,act_dim))
    adv_ph = tf.placeholder(dtype = tf.float32, shape=(None,))
    ret_ph = tf.placeholder(dtype = tf.float32, shape=(None,))
    logp_old_ph = tf.placeholder(dtype = tf.float32, shape=(None,))
    graph_inputs = [state_ph, act_ph, adv_ph, ret_ph, logp_old_ph]


    #tensorflow graph outputs
    pol, logp, logp_pi, val = actor_critic(state_ph, act_ph)

    #experience store
    states_store = np.zeros((epoch_steps,state_dim), dtype = np.float32)
    act_store = np.zeros((epoch_steps,state_dim), dtype = np.float32)
    adv_store = np.zeros(epoch_steps, dtype=np.float32)
    rew_store = np.zeros(epoch_steps, dtype=np.float32)
    ret_store = np.zeros(epoch_steps, dtype=np.float32)
    val_store = np.zeros(epoch_steps, dtype=np.float32)
    logp_store = np.zeros(epoch_steps, dtype=np.float32)
    exp_store = [states_store,act_store,adv_store,rew_store,ret_store,val_store,logp_store]

    #loss functions
    pi_loss = -tf.reduce_mean(logp*adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    #optimizer
    opt_pi = tf.train.AdamOptimizer(learning_rate = pi_lr).minimize(pi_loss)
    opt_val = tf.train.AdamOptimizer(learning_rate = vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #gradient update
    def update():
        #normalize/standardize advantage
        adv_mean, adv_var = tf.nn.moments(adv_store, 0)
        adv_store = (adv_store-adv_mean)/adv_std

        #create input dictionary from trajector and graph inputs
        inputs =  {g:t for g,t in zip(graph_inputs,exp_store)}

        #policy gradient step
        sess.run(opt_pi, feed_dict=inputs)

        #value function training
        for i in range(val_iterations):
            sess.run(opt_val, feed_dict=inputs)

    #cumulative sum
    def disc_cumsum(x, discount):
        """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    state, done, rew, ep_ret, ep_len = env.reset(), 0, 0 , 0
    #experience and training loop
    for epoch in range(epochs):
        traj_start = 0
        for t in range(epoch_steps):
            a, v_t, logp_t = sess.run([pi, val, logp_pi], feed_dict={state_ph: state.reshape(1,-1)})

            #save traj info
            states_store[t] = state
            act_store[t] = a
            rew_store[t] = rew
            val_store[t] = v_t
            logp_store[t] = logp_t

            state, rew, done, = env.step(a[0])
            ep_ret += rew
            ep_len += 1

            terminated = done or (ep_len == max_ep_len)
            if terminated or (t==epoch_steps-1):
                if not terminated:
                    print("traj cut off")
                last_val = rew if done else sess.run(val, feed_dict={state_ph: state.reshape(1,-1)})

                #get rewards and values of trajectory
                traj_slice = slice(traj_start, t)
                rews = np.append(rew_store[traj_slice], last_val)
                vals = np.append(val_store[traj_slice], last_val)

                #GAE-lambda advantage
                deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
                adv_store[traj_slice] = disc_cumsum(deltas, gamma * lam)
                #rewards to go (minimize difference of these and value function)
                ret_store[traj_slice] = disc_cumsum(rews, gamma)[:-1]

                #traj start
                traj_start = t

                state, done, rew, ep_ret, ep_len = env.reset(), 0, 0 , 0

        #gradient update
        update()

vpg()
