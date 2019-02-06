import mujoco_py as mjc
import os
import random
import numpy as np

mj_path, _ = mjc.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'quad_world.xml')
model = mjc.load_model_from_path("model/quad_world.xml")
sim = mjc.MjSim(model)

class cpg():

    def __init__(self,inp_size, hid_size, out_size,dna=None):
        if dna is not None: #seperates and reshapes the concatenated and flattened dna list 
            self.w1 = np.reshape(dna[0:inp_size*hid_size],(inp_size,hid_size))
            self.w2 = np.reshape(dna[inp_size*hid_size:hid_size*inp_size+hid_size*out_size],(hid_size,out_size))
            self.b1 = dna[-(hid_size+out_size):-out_size]
            self.b2 = dna[-out_size:]
        else:
            self.w1 = 2*np.random.rand(inp_size,hid_size)-1
            self.b1 = 2*np.random.rand(hid_size,)-1
            self.w2 = 2*np.random.rand(hid_size,out_size)-1
            self.b2 = 2*np.random.rand(out_size,)-1


    def sigm(self,x):
        return 1/(1+np.exp(-x))

    def ff(self,x):
        inp = self.sigm(x)
        a1 = self.sigm(inp.dot(self.w1)+self.b1)
        a2 = np.tanh(a1.dot(self.w2)+self.b2)
        return a2

    def out(self,a):
        return a

    def dna(self):
        dna = np.concatenate((self.w1.flatten(),self.w2.flatten(),self.b1,self.b2))
        return dna

def simulate(cpg,time=5, render = True):
    if render:
        viewer = mjc.MjViewer(sim)
    
    energy = 0
    lam = 0

    #must render quickly after opening viewer or window crashes
    while sim.data.time < time:

        control = cpg.ff(sim.data.body_xpos.flatten())
        energy = energy + np.sum(np.absolute(control))
        sim.data.ctrl[:] = control

        sim.step()
        if render:
            viewer.render()
            print(control)
    target_x = sim.data.get_geom_xpos("target")
    robot_x = sim.data.get_geom_xpos("torso_geom")

    distance = sum(target_x - robot_x)
    return 1/distance - lam*energy  #-energy-torso stability 


#CPG parameters
inp_size = sim.data.body_xpos.size
hid_size = 3
out_size = 8

#population 
pop = [cpg(inp_size,hid_size,out_size) for i in range(50)]

#GA parameters
ef = 0.4 #elite fraction
mr = 0.5 #mutation rate
md = 0.1 #mutation stdeviation

def crossover(dna1,dna2,f1,f2):
    #normalize probability
    total = f1+f2
    f1 = f1/total
    f2 = f2/total

    c_dna = np.zeros(dna1.shape)
    for i in range(dna1.size):
        #choose chromosome 
        if np.random.rand()<f1:
            c_dna[i] = dna1[i]
        else:
            c_dna[i] = dna2[i]
        #mutation
        if np.random.rand()<mr:
            c_dna[i] = c_dna[i]+np.random.normal(0,md) 
    return c_dna

'''print("start")
a = cpg(5,2,5)
b = cpg(5,2,5)
print("a",a.dna())
print("b",b.dna())
o = crossover(a.dna(),b.dna(),0.5,0.5)

print(o)
print("f")
'''
#epochs
epochs = 1
best = None
for ep in range(epochs):
    print(ep)
    #eval fitness
    fit_pop = []
    for i in pop:
        fit = simulate(i,10,False)
        sim.reset()
        fit_pop.append((i,fit))
    fit_pop.sort(key= lambda tup: tup[1])
    print(np.mean([tup[1] for tup in fit_pop]))

    #select elites
    et = int(len(pop)*ef) #total number of elites
    elites = fit_pop[-et:]
    best = elites[-1]
    print("best:", best[1])
    print(best[0].w1)

    #crossover
    n_offspring = len(pop)-et
    offspring = []
    for os in range(n_offspring):
        p1 = random.choice(elites)
        p2 = random.choice(elites)
        dna1 = p1[0].dna()
        dna2 = p2[0].dna()
        os_dna = crossover(dna1,dna2,p1[1],p2[1])
        offspring.append(cpg(inp_size,hid_size,out_size,os_dna))


    #repopulate
    elites  =[e[0] for e in elites]
    pop = np.concatenate((elites,offspring))


simulate(best[0],5,True)



