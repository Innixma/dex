# By Nick Erickson
# Controls Environment

import numpy as np
import time

class Environment_gym:
    def __init__(self, problem):
        self.problem = problem
        import gym # Lazy import to avoid dependency if not used
        self.env = gym.make(problem)
        
    def run(self, agent):
        s = self.env.reset()

        R = 0 
        
        while True:         
            #self.env.render()
            
            a = agent.act(s)
            s_, r, done, info = self.env.step(a)

            agent.observe( (s, a, r, s_, done) )
            if agent.mode == 'train':
                agent.replay(debug=False)

            s = s_
            R += r
            if done:
                return R, 1

class Environment_realtime:
    def __init__(self, emulator):
        self.env = emulator
        self.timelapse = 1
        
    def framerate_check(self, start_time, frame):
        if time.time() - start_time < (self.timelapse * frame): # Cap framerate
            time.sleep(self.timelapse - (time.time() % self.timelapse))
        
    def init_run(self, img_channels):
        # get the first state by doing nothing and preprocess the image
        x_t, r_0, terminal = self.env.gameState()
    
        stacking = [x_t for i in range(img_channels)]
        s_t = np.stack(stacking, axis=2)
        #In Keras, need to reshape
        s_t = s_t.reshape(s_t.shape[0], s_t.shape[1], img_channels)
        return(s_t)
            
    def run(self, agent):
        frame = 0
        frame_saved = 0
        useRate = np.zeros([agent.action_dim])
        
        self.timelapse = 1/agent.h.framerate
        
        self.env.start_game()
        start_time = time.time()
        s_t = self.init_run(agent.h.img_channels)

        
        while self.env.alive:
            self.framerate_check(start_time, frame)
            action_index = agent.act(s_t)      
            x_t1, r_t, terminal = self.env.gameState(action_index)
            
            if terminal: # Don't save terminal state itself, since it is pure white
                for i in range(agent.h.neg_regret_frames):
                    if agent.memory.size > i:
                        agent.memory.D[-1-i][2] = self.env.reward_terminal/(i+1)
                if agent.memory.size > 0:
                    agent.memory.D[-1][4] = 1 # Terminal State
            else:
                s_t1 = np.append(x_t1, s_t[:, :, :agent.h.img_channels-1], axis=2)
                if frame > agent.h.framerate*agent.args.memory_delay: # Don't store early useless frames
                    agent.observe([s_t, action_index, r_t, s_t1, terminal])
                    useRate[action_index] += 1
                    frame_saved += 1
        
                s_t = s_t1
                frame += 1
                
            if frame > 20000: # Likely stuck, just go to new level
                print('Stuck! Moving on...')
                frame_saved = 0 # TODO: Still have to delete bad memories
                self.env.alive = False
                print('Deleting invalid memory...')
                agent.memory.removeLastN(20000)
        
        end_time = time.time()
        self.env.end_game()
        agent.run_count += 1
        
        agent.metrics.update(end_time-start_time)
        
        if agent.memory.total_saved > agent.h.observe:
            if agent.mode == 'observe':
                agent.mode = 'train'
                time.sleep(1)
        
        return frame, useRate, frame_saved # Metrics
