from gym import error
from gym.spaces import Box, MultiDiscrete, Discrete
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import block


try:
    import floris.tools as wfct
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install floris dependencies with 'pip install gym[floris].)'".format(e))



class FarmEnv(gym.Env):

        #action_space is an array of changes to the angles
        #locations, ws, wd power?
        #state is the yaw angles

        def __init__(self, model, layout, num_turb, wd_up = 360, wd_low = 0, ws = 9, train = 1, render_graph = 0):
            super(FarmEnv, self).__init__()
            
            #varibles
            self.num_turb = num_turb
            self.ws = ws
            self.wd_up = wd_up
            self.wd_low = wd_low
            self.train = train
            self.state = [0]*self.num_turb
            self.model = model
            self.layout = layout
            self.render_graph = render_graph

            if train:
                self.wd = random.randint(self.wd_low,self.wd_up)
            else:
                self.wd = self.wd_low
            
            self.rew = 0
            self.power_prev = 1

            self.fi = wfct.floris_interface.FlorisInterface(model)
            self.fi.reinitialize_flow_field(wind_speed=self.ws,layout_array=self.layout)

            self.no_yaw = []
            for i in range(self.wd_low, self.wd_up+1):
                self.fi.reinitialize_flow_field(wind_direction=i)
                self.no_yaw.append(self.fi.get_farm_power_for_yaw_angle(yaw_angles=[0]*self.num_turb)) 
            


            self.fi.reinitialize_flow_field(wind_direction=self.wd)


            # Actions we can take - angle for each turbine
            self.action_space = MultiDiscrete([5]*self.num_turb) #Box(np.array([-5]*self.num_turb), np.array([5]*self.num_turb)) # # # #yaw angle actions
                                    
            # Temperature array
            self.observation_space = Box(low=np.array(([-90]*self.num_turb)+[0,0]), high=np.array(([90]*self.num_turb)+[360,100]), dtype=np.float32) #yaw 1 - n, wd, power/no yaw power
                                        
            # Set sim length
            self.sim_length = 50

            self.pow_list = np.ones(self.wd_up+1)

            if self.render_graph:
                plt.ion()  # enable interactivity
                self.fig = plt.figure()  # make a figure
                self.ax = self.fig.add_subplot(111)
                self.ax.set_ylim([0.9, 1.1])
                self.line1, = self.ax.plot(np.arange(self.wd_low,self.wd_up+1), self.pow_list, 'b-')

        def step(self, action):

            # Check if done
            if self.sim_length <= 0: 
                done = True
            else:
                done = False
      

            for i in range(0,len(self.state)):
                self.state[i] += 2 - action[i]
                if self.state[i] >= 50:
                  self.state[i] = 49
                if self.state[i] <= -50:
                  self.state[i] = -49


            # Reduce sim length by 1 second
            self.sim_length -= 1 
            
            # Calculate power
            power = self.fi.get_farm_power_for_yaw_angle(yaw_angles=self.state) / self.no_yaw[self.wd]

            if power < 1:
                reward = -1
                self.state = [0] * self.num_turb
                power = 1
            elif power > self.power_prev:
                reward = power
            elif power < self.power_prev:
                reward = -power
            else:
                reward = 0
                        
            self.rew += reward
            
            self.power_prev = power
    
            # Set placeholder for info
            info = {}

            # Return step information
            return self._get_obs(), reward, done, info

        def render(self):

    
            self.pow_list[self.wd] = self.power_prev  # or any arbitrary update to your figure's data
            self.ax.set_ylim([min(self.pow_list)-0.1, max(self.pow_list)+0.1])
    
            self.line1.set_ydata(self.pow_list)
            self.fig.canvas.draw()


        def _get_obs(self):
            return  np.array([*self.state, *[self.wd, self.power_prev]]).astype(np.float32)
        
        def reset(self):
            
            if self.render_graph:
                self.render()

            print(" yaw ", self.state, "| wd: ",self.wd, "| reward: ", self.rew, "| power: ",self.power_prev)

            self.state = [0]*self.num_turb
      
            self.power_prev = 1
            self.rew = 0

            if self.train:
                self.wd = random.randint(self.wd_low,self.wd_up)
            else:
                self.wd += 1

            self.fi.reinitialize_flow_field(wind_direction=self.wd)

            self.sim_length = 50

            return self._get_obs()