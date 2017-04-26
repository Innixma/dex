# By Nick Erickson
# Controls Environment

import numpy as np
import time
import skimage as skimage
from skimage import color # Tmp, make this a new class later
import skimage.transform as transf

class Environment_gym:
    def __init__(self, env_info):
        self.problem = env_info.problem
        #import gym # Lazy import to avoid dependency if not used
        import gym_preprocess
        
        #self.env = gym.make(problem)
        self.env = getattr(gym_preprocess,env_info.wrapper)(self.problem)
        #self.env = gym.make(self.problem)
        
    def run(self, agent):
        s = self.env.reset()
        #print(s.shape)
        R = 0 
        
        while True:         
            #self.env.render()
            
            a = agent.act(s)
            s_, r, t, info = self.env.step(a)
            
            agent.observe(s, a, r, s_, t)
            if agent.mode == 'train':
                agent.replay(debug=False)
                
                if agent.args.algorithm == 'a3c':
                    if agent.brain.brain_memory.isFull:
                        agent.brain.optimize_batch_full()

            s = s_
            R += r
            if t:
                return R, 1

class Environment_realtime_a3c:
    def __init__(self, emulator, img_channels=1):
        self.env = emulator
        self.timelapse = 1
        self.has_base_frame = False
        self.base_frame = None
        
    def framerate_check(self, start_time, frame):
        if time.time() - start_time < (self.timelapse * frame): # Cap framerate
            time.sleep(self.timelapse - (time.time() % self.timelapse))
        else:
            self.catchup_frames += 1
        
    def init_run(self, img_channels):
        x, r, t = self.env.gameState()
    
        stacking = [x for i in range(img_channels)]
        s = np.stack(stacking, axis=2)

        s = s.reshape(s.shape[0], s.shape[1], img_channels)
        return(s)
            
    def run(self, agent):
        frame_delay = int(agent.h.framerate*agent.args.memory_delay)
        self.catchup_frames = -1
        frame = 0
        frame_saved = 0
        useRate = np.zeros([agent.action_dim])
        
        self.timelapse = 1/agent.h.framerate
        
        self.env.start_game()
        start_time = time.time()
        s = self.init_run(agent.h.img_channels)
        
        if not self.has_base_frame:
            self.has_base_frame = True
            new_shape = [1] + list(s.shape)
            self.base_frame = s.reshape(new_shape)
        
        v_episode = []
        while self.env.alive:
            self.framerate_check(start_time, frame)
            #a = agent.act(s)
            a, v_cur = agent.act_v(s)
            
            
            
            x_, r, t = self.env.gameState(a)
            
            s_ = np.append(x_, s[:, :, :agent.h.img_channels-1], axis=2)

            if frame > frame_delay: # Don't store early useless frames
                agent.observe(s, a, r, s_, t)
                agent.replay()
                useRate[a] += 1
                frame_saved += 1
                v_episode.append(v_cur[0][0])
        
            s = s_
            frame += 1
                
            if frame > 80000: # Likely stuck, just go to new level
                print('Stuck! Moving on...')
                frame_saved = 0
                self.env.alive = False
                print('Deleting invalid memory...')
                agent.brain.brain_memory.isFull = False # Reset brain memory
                agent.brain.brain_memory.size = 0
        
        end_time = time.time()
        self.env.end_game()
        agent.run_count += 1
        
        agent.metrics.update(end_time-start_time)
        
        v = agent.brain.predict_v(self.base_frame)[0][0]
        print('V:', str(v), ', catchup:', str(self.catchup_frames))
        agent.metrics.V.append(v)
        agent.metrics.V_episode.extend(v_episode)
        return frame, useRate, frame_saved # Metrics

class Environment_realtime:
    def __init__(self, emulator):
        self.env = emulator
        self.timelapse = 1
        
    def framerate_check(self, start_time, frame):
        if time.time() - start_time < (self.timelapse * frame): # Cap framerate
            time.sleep(self.timelapse - (time.time() % self.timelapse))
        
    def init_run(self, img_channels):
        x, r, t = self.env.gameState()
    
        stacking = [x for i in range(img_channels)]
        s = np.stack(stacking, axis=2)

        s = s.reshape(s.shape[0], s.shape[1], img_channels)
        return(s)
            
    def run(self, agent):
        frame_delay = int(agent.h.framerate*agent.args.memory_delay)
        frame = 0
        frame_saved = 0
        useRate = np.zeros([agent.action_dim])
        
        self.timelapse = 1/agent.h.framerate
        
        self.env.start_game()
        start_time = time.time()
        s = self.init_run(agent.h.img_channels)
        
        while self.env.alive:
            self.framerate_check(start_time, frame)
            a = agent.act(s)      
            x_, r, t = self.env.gameState(a)
            
            s_ = np.append(x_, s[:, :, :agent.h.img_channels-1], axis=2)
            
            #if t: # Don't save t state itself, since it is pure white
            #    for i in range(agent.h.neg_regret_frames):
            #        if agent.memory.size > i:
            #            agent.memory.D[-1-i][2] = self.env.reward_terminal/(i+1)
            #    if agent.memory.size > 0:
            #        agent.memory.D[-1][4] = 1 # t State
            #else:
                
            if frame > frame_delay: # Don't store early useless frames
                agent.observe(s, a, r, s_, t)
                useRate[a] += 1
                frame_saved += 1
        
            s = s_
            frame += 1
                
            if frame > 80000: # Likely stuck, just go to new level
                print('Stuck! Moving on...')
                frame_saved = 0
                self.env.alive = False
                print('Deleting invalid memory...')
                agent.brain.brain_memory.isFull = False # Reset brain memory
                agent.brain.brain_memory.size = 0
        
        end_time = time.time()
        self.env.end_game()
        agent.run_count += 1
        
        agent.metrics.update(end_time-start_time)
        
        if agent.memory.total_saved > agent.h.observe:
            if agent.mode == 'observe':
                agent.mode = 'train'
                time.sleep(1)
        
        return frame, useRate, frame_saved # Metrics
