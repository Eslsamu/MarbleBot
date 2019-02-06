import mujoco_py as mjc
import numpy as np

class robo_env():
 
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
        en_cost = np.sum(action)

        reward = move_rew - en_cost

        #check if episode is over
        done = self.check_collision()

        state = self._get_state()

        return reward, state, done

    def _get_state(self):
        state = np.concatenate([
                sim.data.qpos.flat,
                sim.data.qvel.flat])
        return state 

    def check_collision(self):
        sim=self.sim
        torso = sim.model.geom_name2id('torso_geom')
        floor = sim.model.geom_name2id('floor')
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            if contact.geom1 == torso or contact.geom1 == floor and contact.geom2 == torso or contact.geom2 == floor:
                return True
        return False

    def reset(self):
        self.sim.reset()
        done = check_collision()
        return self._get_state(), done

    @property
    def dt(self):
        return self.model.opt.timestep * 5 #this value is from the spinup library /mujoco_env.py

    def render(self):
        if self.viewer is None:
            self.viewer = mjc.MjViewer(self.sim)
        self.viewer.render()
