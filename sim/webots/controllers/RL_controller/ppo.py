import os
import time
import logging
logging.basicConfig(filename='ppo.log',format='%(asctime)s %(message)s', level=logging.DEBUG)
import numpy as np
import tensorflow as tf
from rl_policy import actor_critic
from buffer import Buffer
from wbt_jobs import run_job, visualize_policy
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
            hidden_sizes = (64,64), act_dim=None, obs_dim= None, action_scale=None,n_proc = 2, model_path = SAVE_MODEL_PATH):

    if not(act_dim and obs_dim):
            logging.info("Missing action or obs dimension")

    if action_scale is None:
        action_scale = np.ones(act_dim)

    #tensorflow graph inputs
    with tf.name_scope('graph_inputs'):
        obs_ph = tf.placeholder(dtype = tf.float32, shape=(None,obs_dim), name='observations')
        act_ph = tf.placeholder(dtype = tf.float32, shape=(None,act_dim), name='actions')
        adv_ph = tf.placeholder(dtype = tf.float32, shape=(None,), name='advantage')
        ret_ph = tf.placeholder(dtype = tf.float32, shape=(None,), name='return')
        logp_old_ph = tf.placeholder(dtype = tf.float32, shape=(None,), name='logp_old')

    graph_inputs = [obs_ph, act_ph, adv_ph, ret_ph, logp_old_ph]
    
    #input_summaries = tf.summary.merge([obs_summary,act_summary,adv_summary,ret_summary,logp_old_summary])


    #tensorflow graph outputs
    pi, logp, logp_pi, val = actor_critic(obs_ph, act_ph, action_scale, hidden_sizes)

    #output_summaries = tf.summary.merge([pi_summary,logp_summary,logp_pi_summary,val_summary])

    #experience buffer
    buf = Buffer(obs_dim, act_dim, epoch_steps, gamma, lam)


    #count variables
    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
    logging.info('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    #loss functions
    with tf.name_scope('pi_loss'):
        ratio = tf.exp(logp - logp_old_ph)      #pi(a|s) / pi_old(a|s)
        min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))

    with tf.name_scope('val_loss'):
        v_loss = tf.reduce_mean((ret_ph - val)**2)

    avg_ret = tf.reduce_mean(ret_ph)
    avg_ret_ph = tf.placeholder(tf.float32, shape=None,name='return_summary')
    ret_summary = tf.summary.scalar('ret_summ', avg_ret_ph)
    # Info (useful to watch during learning)
    with tf.name_scope('kl_divergence'):
        approx_kl = tf.reduce_mean(logp_old_ph - logp)  # a sample estimate for KL-divergence, easy to compute
        clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
        clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        kl_ph = tf.placeholder(tf.float32,shape=None,name='kl_summary')
        kl_summary = tf.summary.scalar('kl_summ', kl_ph)
    
    
    with tf.name_scope('entropy'):
        approx_ent = tf.reduce_mean(-logp)  # a sample estimate for entropy, also easy to compute
        
        ent_ph = tf.placeholder(tf.float32,shape=None,name='entropy')
        ent_summary = tf.summary.scalar('ent_summ', ent_ph)


    with tf.name_scope('loss'):
        pi_loss_ph = tf.placeholder(tf.float32,shape=None,name='piloss_summary')
        v_loss_ph = tf.placeholder(tf.float32,shape=None,name='vloss_summary')
        
        pi_loss_summary = tf.summary.scalar('pi_loss_summ', pi_loss_ph)
        v_loss_summary = tf.summary.scalar('val_loss_summ', v_loss_ph)
        
    
    performance_summaries = tf.summary.merge([pi_loss_summary,v_loss_summary,kl_summary, ent_summary,ret_summary])

    #optimizer
    with tf.name_scope('Train_pi'):
        opt_pi = tf.train.AdamOptimizer(learning_rate = pi_lr).minimize(pi_loss)
    
    with tf.name_scope('Train_val'):
        opt_val = tf.train.AdamOptimizer(learning_rate = vf_lr).minimize(v_loss)

    sess = tf.Session()

    if not os.path.exists('summaries'):
        os.mkdir('summaries')
    if not os.path.exists(os.path.join('summaries','first')):
        os.mkdir(os.path.join('summaries','first'))
    
    summ_writer = tf.summary.FileWriter(os.path.join('summaries','first'), sess.graph)

    sess.run(tf.global_variables_initializer())

    #gradient update
    def update():
        #create input dictionary from trajector and graph inputs
        inputs =  {g:t for g,t in zip(graph_inputs,buf.get())}

        pi_l_old, val_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        #Training
        for i in range(pi_iters):
            _, kl, rat, min, l , l_old = sess.run([opt_pi, approx_kl, ratio, min_adv, logp, logp_old_ph], feed_dict=inputs)
            print(kl)
            kl = np.mean(kl)
            print("kl",kl)
            print("ratio", rat)
            print("min_adv", min)
            print(l, l_old)
            if kl > 1.5 * target_kl:
                logging.info('Early stopping at step %d due to reaching max kl.'+str(kl))
                print("kl too big", kl)
                break

        #value function training
        for i in range(val_iters):
            sess.run(opt_val, feed_dict=inputs)

        pi_l_new, val_l_new, kl, cf, ret = sess.run([pi_loss, v_loss, approx_kl, clipfrac, avg_ret], feed_dict=inputs)
        update_info = "policy loss: " + str(pi_l_old) + \
                      "\n" + "value func loss:" + str(val_l_old) + "\n" +\
                      "delta policy loss: " + str(pi_l_new-pi_l_old) + "\n" +\
                      "delta value func loss: " + str(val_l_new - val_l_old) + "\n" +\
                      "kl: " + str(kl) + "\n" +"entropy: " + str(ent) + "\n"

        print(update_info)
        logging.info(update_info)
        summ = sess.run(performance_summaries, feed_dict={pi_loss_ph:pi_l_new, v_loss_ph:val_l_new, kl_ph:kl, ent_ph:ent, avg_ret_ph:ret})
        summ_writer.add_summary(summ, epoch)
    start_time = time.time()

    #initialize model saving
    saver = ModelSaver(sess, inputs = {"obs":obs_ph},outputs={"pi": pi, "val": val, "logp_pi": logp_pi}, export_dir= model_path)

    #experience and training loop
    for epoch in range(epochs):
        # save model
        saver.save_model()

        #build world files only for first epoch
        if epoch == 0:
            first = True
        epoch_data = run_job(n_proc=n_proc, total_steps=epoch_steps, max_ep_steps= max_ep_len,model_path =saver.fpath,build_files=first)

        #save epoch data
        max_ret, min_ret, avg_return, max_len, min_len, avg_len, avg_ene, avg_clipped, avg_dist, avg_abs_dist = buf.store_epoch(epoch_data)

        ep_info = "============Epoch " + str(epoch) + " max/min/avg return " + str(max_ret)\
                  +" " + str(min_ret) + " " + str(avg_return) + " max/min/avg length "\
                  + " " + str(max_len) + " " + str(min_len) +" "+ str(avg_len) \
                  + " avg energy/action_clip/distance/abs_distance" \
                  + " " + str(avg_ene) + " " + str(avg_clipped) +" "+ str(avg_dist)+" " \
                  + str(avg_abs_dist) \
                  + "============"
        print(ep_info)
        logging.info(ep_info)

        #update policy
        update()

        #if epoch > 1:
           # visualize_policy(max_ep_len, "saved_model/simple_save")


import json
file = "devices.json"
with open(file) as f:
    devices = json.load(f)
    sensor_names = devices["force_sensors"] + devices["IMUs"]
    motor_names = devices["lin_motors"] + devices["rot_motors"]

obs_dim = 33 #TODO better solution (now just multiplies force sensor by 3 for each dim)
act_dim = len(motor_names)
action_scale = np.array([0.2, 0.2, 0.2, 0.2, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
run_ppo(epochs=100, epoch_steps=100, act_dim = act_dim, obs_dim = obs_dim, action_scale=action_scale, n_proc=1)