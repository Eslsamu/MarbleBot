import scipy.signal
import numpy as np

#cumulative sum
def disc_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    
class Buffer():
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

    """
    stores epoch data and returns:
     - max/min/average return per episode
     - max/min/average episode length
    """
    def store_epoch(self, epoch_data):
        ep_lens = []
        ep_rets = []

        #additional info
        ep_enes = []
        ep_survs = []
        ep_dists = []
        ep_abs_dists = []

        for proc in epoch_data:
            for epi in proc:
                obs = epi["obs"]
                a = epi["a"]
                val = epi["val"]
                rew = epi["rew"]
                logp = epi["logp"]
                ep_len = epi["ep_len"]
                ep_ret = epi["ep_ret"]
                last_val = epi["last_val"]
                ep_ene = epi["ep_ene"]
                ep_surv = epi["ep_surv"]
                ep_dist = epi["ep_dist"]
                ep_abs_dist = epi["ep_abs_dist"]

                for t in range(ep_len):
                    self.store(obs[t], a[t], rew[t], val[t], logp[t])
                self.finish_path(last_val)

                ep_lens += [ep_len]
                ep_rets += [ep_ret]
                ep_enes += [ep_ene]
                ep_survs += [ep_surv]
                ep_dists += [ep_dist]
                ep_abs_dists += [ep_abs_dist]


        return np.max(ep_rets), np.min(ep_rets), np.mean(ep_rets), \
               np.max(ep_lens), np.min(ep_lens), np.mean(ep_lens), \
               np.mean(ep_enes), np.mean(ep_survs), np.mean(ep_dists),\
               np.mean(ep_abs_dists)

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
        try:
            assert self.ptr == self.max_size    # buffer has to be full before you can get
        except AssertionError:
            print("====================buffer filled with ", self.ptr,"==================")
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_var = np.var(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_var**0.5


        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]