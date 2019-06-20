import time
import logging
logging.basicConfig(filename='ppo.log',format='%(asctime)s %(message)s', level=logging.DEBUG)
import numpy as np
import tensorflow as tf
from rl_policy import actor_critic
from buffer import Buffer
from wbt_jobs import run_job
import os.path as osp
import shutil
import joblib

SAVE_MODEL_PATH = "saved_model"

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

class ModelSaver():


    def __init__(self, sess, inputs, outputs, export_dir):
        self.sess = sess
        self.inputs = inputs
        self.outputs = outputs
        self.export_dir = export_dir
        self.tf_saver_info = {'inputs': {k:v.name for k,v in inputs.items()},
                          'outputs': {k:v.name for k,v in outputs.items()}}

    def save_model(self, itr= None):
        #needs to find a nonexistant directory path
        self.fpath = 'simple_save' + ('%d' % itr if itr is not None else '')
        self.fpath = osp.join(self.export_dir, self.fpath)

        if osp.exists(self.fpath):
        # simple_save refuses to be useful if fpath already exists,
        # so just delete fpath if it's there.
            shutil.rmtree(self.fpath)

        tf.saved_model.simple_save(
            session=self.sess,
            export_dir=self.fpath,
            inputs=self.inputs,
            outputs=self.outputs
        )
        joblib.dump(self.tf_saver_info, osp.join(self.fpath, 'model_info.pkl'))

def run_ppo(epochs=30,epoch_steps = 4000 , max_ep_len=500 ,pi_lr = 3e-4, vf_lr=1e-3,
            gamma=0.99,lam=0.97,pi_iters = 80,target_kl = 0.01,val_iters=80, clip_ratio=0.2,
            hidden_sizes = (64,64), act_dim=None, obs_dim= None, n_proc = 2, model_path = SAVE_MODEL_PATH):

    if not(act_dim and obs_dim):
            logging.info("Missing action or obs dimension")

    #tensorflow graph inputs
    obs_ph = tf.placeholder(dtype = tf.float32, shape=(None,obs_dim))
    act_ph = tf.placeholder(dtype = tf.float32, shape=(None,act_dim))
    adv_ph = tf.placeholder(dtype = tf.float32, shape=(None,))
    ret_ph = tf.placeholder(dtype = tf.float32, shape=(None,))
    logp_old_ph = tf.placeholder(dtype = tf.float32, shape=(None,))
    graph_inputs = [obs_ph, act_ph, adv_ph, ret_ph, logp_old_ph]


    #tensorflow graph outputs
    pi, logp, logp_pi, val = actor_critic(obs_ph, act_ph, hidden_sizes)

    #experience buffer
    buf = Buffer(obs_dim, act_dim, epoch_steps, gamma, lam)


    #count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
    logging.info('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

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
                logging.info('Early stopping at step %d due to reaching max kl.'%i)
                break

        #value function training
        for i in range(val_iters):
            sess.run(opt_val, feed_dict=inputs)

        pi_l_new, val_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logging.info(
            "policy loss: " + str(pi_l_old) + "\n" +
            "value func loss:" + str(val_l_old) + "\n" +
            "delta policy loss: " + str(pi_l_new-pi_l_old) + "\n" +
            "delta value func loss: " + str(val_l_new - val_l_old) + "\n" +
            "kl: " + str(kl) + "\n" +
            "entropy: " + str(ent) + "\n"
        )

    start_time = time.time()

    #initialize model saving
    saver = ModelSaver(sess, inputs = {"obs":obs_ph},outputs={"pi": pi, "val": val, "logp_pi": logp_pi}, export_dir= model_path)

    #experience and training loop
    for epoch in range(epochs):
        print("epoch", epoch)
        # save model
        saver.save_model()

        #build world files only for first epoch
        if epoch == 0:
            first = True
        epoch_data = run_job(n_proc=n_proc, total_steps=epoch_steps, max_ep_steps= max_ep_len,model_path =saver.fpath,build_files=first)

        #save epoch data
        buf.store_epoch(epoch_data)


        #TODO log episode return
        #update policy
        update()


import json
file = "devices.json"
with open(file) as f:
    devices = json.load(f)
    sensor_names = devices["sensors"]
    motor_names = devices["lin_motors"] + devices["rot_motors"]

obs_dim = len(sensor_names) * 3 #TODO better solution (now just multiplies force sensor by 3 for each dim)
act_dim = len(motor_names)
run_ppo(act_dim = act_dim, obs_dim = obs_dim, n_proc=1)