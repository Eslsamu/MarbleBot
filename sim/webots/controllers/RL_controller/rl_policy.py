import numpy as np
import tensorflow as tf

#noise constant
EPS = 1e-8

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
    pi = tf.add(mean,tf.random_normal(tf.shape(mean))*std)

    #gives log probability of taking 'actions' according to the policy in states
    logp = log_likelihood(action, mean, log_std)

    #gives log probability according to the policy of the actions sampled by pi
    logp_pi = log_likelihood(pi, mean, log_std)
    return pi, logp, logp_pi

def actor_critic(state, action, hidden_sizes=(64,64)):

    #policy 
    with tf.name_scope('pi'):
        pi, logp, logp_pi = policy(state, action, hidden_sizes)

    #state value estimation network
    with tf.variable_scope('val'):
            val = tf.squeeze(nn(state, list(hidden_sizes)+[1]), axis = 1)

    return pi, logp, logp_pi, val