import mujoco_py as mjc
import numpy as np

class RoboEnv():

    # energy parame
    e = 0.0001

    #collision punishment
    p = 0

    #survival param
    s = 0

    def __init__(self, xml_path):
        self.model = mjc.load_model_from_path(xml_path)
        self.sim = mjc.MjSim(self.model)
        self.act_shape = self.sim.data.ctrl.shape
        self.state_shape = np.concatenate([
               self.sim.data.qpos.flat,
               self.sim.data.qvel.flat]).shape
        self.viewer = None


    def step(self, action):
        sim = self.sim
        sim.data.ctrl[:] = action

        #x position before
        pos_before = sim.data.get_geom_xpos('torso_geom')[0]
        #simulation step
        sim.step()
        #x position after step
        pos_after = sim.data.get_geom_xpos('torso_geom')[0]

        #reward for movement in x direction
        move_rew = (pos_after-pos_before)/self.dt

        #energy cost
        en_cost = np.sum(np.abs(action))

        reward = move_rew + self.s - self.e*en_cost

        #check if episode is over
        done = self.check_collision()
        if done:
            reward = reward - self.p

        state = self._get_state()

        return reward, state, done

    def _get_state(self):
        state = np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat])
        return state 

    def check_collision(self):
        sim = self.sim
        torso = sim.model.geom_name2id('torso_geom')
        floor = sim.model.geom_name2id('floor')
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            if contact.geom1 == torso or contact.geom1 == floor and contact.geom2 == torso or contact.geom2 == floor:
                return True
        return False

    def reset(self):
        self.sim.reset()
        return self._get_state()

    @property
    def dt(self):
        return self.model.opt.timestep * 5 #this value is from the spinup library /mujoco_env.py

    def render(self, on=True):
        if on:
            if self.viewer is None:
                self.viewer = mjc.MjViewer(self.sim)
            self.viewer.render()
        if not on:
            if self.viewer:
                self.viewer = None


class PendelumEnv():
    def __init__(self, xml_path):
        self.model = mjc.load_model_from_path(xml_path)
        self.sim = mjc.MjSim(self.model)
        self.act_shape = self.sim.data.ctrl.shape
        self.state_shape = np.concatenate([
            self.sim.data.qpos,
            self.sim.data.qvel]).ravel().shape
        self.viewer = None
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.frame_skip = 2

    def step(self, a):
        reward = 1.0
        self.sim.data.ctrl[:] = a
        for _ in range(self.frame_skip):
            self.sim.step()
        obs = self._get_obs()
        done = not (np.isfinite(obs).all() and (np.abs(obs[1]) <= .2))
        return reward, obs, done

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def reset(self):
        qpos = self.init_qpos + np.random.uniform(size = self.sim.model.nq, low = 0.01, high = 0.01)
        qvel = self.init_qvel + np.random.uniform(size = self.sim.model.nv, low = 0.01, high = 0.01)

        old_state = self.sim.get_state()
        new_state = mjc.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)

        self.sim.set_state(new_state)
        self.sim.forward()
        return self._get_obs()

    def render(self, on=True):
        if on:
            if self.viewer is None:
                self.viewer = mjc.MjViewer(self.sim)
            self.viewer.render()
        if not on:
            if self.viewer:
                self.viewer = None