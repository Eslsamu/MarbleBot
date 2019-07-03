import numpy as np
import tensorflow as tf

#noise constant
EPS = 1e-8

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

def apply_squashing_func(action, pi, logp, logp_pi):
    #with tf.variable_scope('pi'):
    with tf.name_scope('squashing'):
        pi = tf.tanh(pi)

        #TODO check correctness before using
        #logp -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - action**2, l=0, u=1) + 1e-6), axis=1)

        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        #with tf.variable_scope('logp_pi'):
        logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return pi, logp, logp_pi

#log likelihood of x given normal distribution(mean,log_std)
def log_likelihood(x, mean, log_std):
   prob = -0.5 * (( (x-mean)/(tf.exp(log_std)+EPS))**2 + 2 * log_std + np.log(2*np.pi))
   return tf.reduce_sum(prob, axis=1)

def nn(inp, hidden_sizes=(8,), activation=tf.tanh, pi=True):
    for idx, h in enumerate(hidden_sizes[:-1]):
        if pi==True:
            with tf.variable_scope('pi_layer_'+str(idx)):
                inp = tf.layers.dense(inp, units=h, activation=activation)
        else:
            with tf.variable_scope('val_layer_'+str(idx)):
                inp = tf.layers.dense(inp, units=h, activation=activation)
    if pi==True:
        with tf.variable_scope('pi_output'):
            output = tf.layers.dense(inp,units=hidden_sizes[-1], activation=tf.tanh)
    else:
        with tf.variable_scope('val_output'):
            output = tf.layers.dense(inp,units=hidden_sizes[-1], activation=tf.tanh)

    return output


LOG_STD_MAX = 2
LOG_STD_MIN = -20

def policy(state, action, hidden_sizes):
    act_dim = action.shape.as_list()[-1]

    net = nn(state, list(hidden_sizes))
    with tf.name_scope('mean'):
        mean = tf.layers.dense(net, act_dim, activation=None)

    with tf.variable_scope('log_std'):
        log_std = tf.get_variable(name='log_std', initializer=np.array([-2.5,-2.5,-2.5,-2.5,1,1,1,1,1,1,1,1], dtype=np.float32))
        #log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
        #log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    #take log standard deviation because it can take any values in (-inf, inf) -> easier to train without constraints 
    std = tf.exp(log_std)

    #samples actions from policy given a state
    with tf.variable_scope('pi'):
        pi = mean + tf.random_normal(tf.shape(mean))*std

    #gives log probability of taking 'actions' according to the policy in states
    with tf.variable_scope('logp'):
        logp = log_likelihood(action, mean, log_std)

    #gives log probability according to the policy of the actions sampled by pi
    with tf.variable_scope('logp_pi'):
        logp_pi = log_likelihood(pi, mean, log_std)
    return pi, logp, logp_pi

def actor_critic(state, action, action_scale, hidden_sizes=(400,300)):

    #policy 
    with tf.variable_scope('pi_nn'):
        pi, logp, logp_pi = policy(state, action, hidden_sizes)

    #state value estimation network
    with tf.variable_scope('val_nn'):
        val = tf.squeeze(nn(state, list(hidden_sizes)+[1],pi=False), axis = 1)

    return pi, logp, logp_pi, val