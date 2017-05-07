# By Nick Erickson
# Contains functions for game play loops

from environment import Environment_gym, Environment_gym_rgb, Environment_realtime, Environment_realtime_a3c
from data_utils import saveAll, save_class, loadClass, save_weights, saveMemory_v2, loadMemory_v2, save_memory_subset
import time
try:
    import OpenHexagonEmulator
except:
    print("Can't import OpenHexagonEmulator, not a Windows environment, skipping...")
import models

import threading

MULTITHREAD = 7
MULTITHREAD_BRAIN = 1

def run(args, agent):   
    if args.env.type == 'real':
        if args.algorithm == 'ddqn':
            playGameReal_ddqn(args, agent)
        elif args.algorithm == 'a3c':
            playGameReal_a3c(args, agent)
    elif args.env.type == 'gym':
        if args.algorithm == 'ddqn':
            playGameGym_ddqn(args, agent)
        elif args.algorithm == 'a3c':
            playGameGym_a3c(args, agent)
    elif args.env.type == 'memory': 
        gatherMemory(args, agent)
    else:
        pass

def init_agents(args, agent_func):
    agents = []
    envs = []
    brain = None
    for i in range(MULTITHREAD):
        if args.hyper.img_channels > 1:
            env = Environment_gym_rgb(args.env, i)
            state_dim  = env.env.state_dim + [args.hyper.img_channels]
        else:
            env = Environment_gym(args.env, i)
            state_dim  = env.env.state_dim
        action_dim = env.env.action_dim
        agent = agent_func(args, state_dim, action_dim, getattr(models,args.model), brain=brain, idx=i)
        brain = agent.brain
        envs.append(env)
        agents.append(agent)
        
    return agents, envs
    
def playGameGym_a3c_multithread_init(args, agent_func):
    agents, envs = init_agents(args, agent_func)
    brain = agents[0].brain
    threads = []
    for i in range(MULTITHREAD):
        threads.append(Multithread_agent(agents[i], envs[i]))
        threads[-1].daemon = True

    threads_brain = []

    for i in range(MULTITHREAD_BRAIN):
        threads_brain.append(Multithread_brain(brain))
        threads_brain[-1].daemon = True
        
    for i in range(MULTITHREAD):
        threads[i].start()
        
        
    for i in range(MULTITHREAD_BRAIN):
        threads_brain[i].start()
        
    start = time.time()
    while True:
        time.sleep(20)
        print(brain.brain_memory.total_saved, 'saved')
        print(brain.brain_memory.total_saved / (time.time() - start), 'saved per second')
    """    
    time.sleep(20)
    
    print('stopping')
    
    for i in range(MULTITHREAD):
        threads[i].stop()
        
    for i in range(MULTITHREAD):
        threads[i].join()
        
    print('agents all stopped')
    
    for i in range(MULTITHREAD_BRAIN):
        threads_brain[i].stop()
        
    for i in range(MULTITHREAD_BRAIN):
        threads_brain[i].join()
        
    print('brains all stopped')
    print(brain.brain_memory.total_saved)
    print(brain.c)
    print(brain.brain_memory.total_saved / brain.c)
    """
class Multithread_agent(threading.Thread):
    stop_signal = False
    def __init__(self, agent, env):
        threading.Thread.__init__(self)
        self.agent = agent
        self.env = env
        
    def run(self):
        iteration = 0
        while not self.stop_signal:
            iteration += 1
        
            R, useRate = self.env.run(self.agent)
            
            if self.agent.mode == 'train':
                if iteration % 10 == 0:
                    print("Step:", self.agent.memory.total_saved, ", Total reward:", R, "idx:", self.agent.idx)
                
            #agent.metrics.display_metrics(frame, useRate, agent.memory.total_saved, agent.epsilon)
            
            if self.agent.idx == 0:
                if self.agent.h.save_rate < self.agent.save_iterator:
                    self.agent.save_iterator -= self.agent.h.save_rate
                    save_weights(self.agent, self.agent.brain.brain_memory.total_saved)
                    #self.agent.metrics.a3c.graph_all(self.agent.results_location)
                    
            #playGameGym_a3c_multithread(self.agent, self.env)
        
    def stop(self):
        self.stop_signal = True

class Multithread_brain(threading.Thread):
    stop_signal = False
    def __init__(self, brain):
        threading.Thread.__init__(self)
        self.brain = brain
                  
    def run(self):
        while not self.stop_signal:
            self.brain.optimize_batch_full_multithread()
        
    def stop(self):
        self.stop_signal = True
        
def playGameGym_a3c_multithread(agent, env):
    iteration = 0
    while (True):
        iteration += 1
        
        R, useRate = env.run(agent)
        
        if agent.mode == 'train':
            if iteration % 10 == 0:
                print("Step:", agent.memory.total_saved, ", Total reward:", R, "idx:", agent.idx)
            
        #agent.metrics.display_metrics(frame, useRate, agent.memory.total_saved, agent.epsilon)
        
        if agent.idx == 0:
            if agent.h.save_rate < agent.save_iterator:
                agent.save_iterator -= agent.h.save_rate
                save_weights(agent, agent.run_count)
                #agent.metrics.a3c.graph_all(agent.results_location)
                #if agent.mode == 'train': # Fix this later, not correct
                        #agent.metrics.save_metrics(agent.results_location)
                        #agent.metrics.save_metrics_training(agent.results_location)
    
    
def playGameGym_a3c(args, agent_func):
    
    playGameGym_a3c_multithread_init(args, agent_func)

    """
    if args.hyper.img_channels > 1:
        env = Environment_gym_rgb(args.env)
        state_dim  = env.env.state_dim + [args.hyper.img_channels]
    else:
        env = Environment_gym(args.env)
        state_dim  = env.env.state_dim
    action_dim = env.env.action_dim

    
    agent = agent_func(args, state_dim, action_dim, getattr(models,args.model))
    #brain = agent.brain
    
    iteration = 0
    while (True):
        iteration += 1
        
        R, useRate = env.run(agent)
        
        if agent.mode == 'train':
            if iteration % 10 == 0:
                print("Step:", agent.memory.total_saved, ", Total reward:", R, "idx:", agent.idx)
            
        #agent.metrics.display_metrics(frame, useRate, agent.memory.total_saved, agent.epsilon)
        
        if agent.h.save_rate < agent.save_iterator:
            agent.save_iterator -= agent.h.save_rate
            save_weights(agent, agent.run_count)
            agent.metrics.a3c.graph_all(agent.results_location)
            #if agent.mode == 'train': # Fix this later, not correct
                    #agent.metrics.save_metrics(agent.results_location)
                    #agent.metrics.save_metrics_training(agent.results_location)
    """
def playGameGym_ddqn(args, agent_func):
    if args.hyper.img_channels > 1:
        env = Environment_gym_rgb(args.env)
        state_dim  = env.env.state_dim + [args.hyper.img_channels]
    else:
        env = Environment_gym(args.env)
        state_dim  = env.env.state_dim
    action_dim = env.env.action_dim

    agent = agent_func(args, state_dim, action_dim, getattr(models,args.model))
    
    iteration = 0
    while (True):
        iteration += 1
        
        R, useRate = env.run(agent)
        
        if agent.memory.total_saved > agent.h.extra.observe:
            if agent.mode == 'observe':
                agent.mode = 'train'
                print('Training...')
                time.sleep(0.5)
        
        if agent.mode == 'train':
            if iteration % 10 == 0:
                print("Step:", agent.memory.total_saved, ", Total reward:", R)
            
        #agent.metrics.display_metrics(frame, useRate, agent.memory.total_saved, agent.epsilon)
        
        if agent.h.save_rate < agent.save_iterator:
            agent.save_iterator -= agent.h.save_rate
            save_weights(agent)
            #if agent.mode == 'train': # Fix this later, not correct
                    #agent.metrics.save_metrics(agent.results_location)
                    #agent.metrics.save_metrics_training(agent.results_location)

def playGameReal_a3c(args, agent_func, screen_number=0, screen_id=-1):

    img_channels = args.hyper.img_channels
    env = Environment_realtime_a3c(args.env)
    action_dim = env.env.action_dim()
    state_dim = list(env.env.state_dim()) + [img_channels]
    #env = Environment_realtime_a3c(emulator, img_channels)
    
    print(state_dim)
    print(action_dim)
    
    agent = agent_func(args, state_dim, action_dim, getattr(models,args.model))
    
    
    hasSavedMemory = False
    
    max_frame_saved = 300
    
    while (True):
        pointer_start = agent.brain.brain_memory.curIndex + 0
        frame, useRate, frame_saved = env.run(agent)
        pointer_end = agent.brain.brain_memory.curIndex + 0
        agent.metrics.display_metrics(frame, useRate, agent.memory.total_saved, agent.epsilon)
                    
        if frame_saved > max_frame_saved:
            print('New max time!')
            max_frame_saved = frame_saved
            
            save_memory_subset(agent, pointer_start, pointer_end, frame_saved, skip=8)
            save_weights(agent, 'frame_' + str(frame_saved))
            
        
        if agent.h.save_rate < agent.save_iterator:
            agent.save_iterator -= agent.h.save_rate
            save_weights(agent, agent.run_count)
            agent.metrics.save_metrics(agent.results_location)
            agent.metrics.save_metrics_v(agent.results_location)
            #agent.metrics.a3c.graph_all(agent.results_location)
            #agent.metrics.save_metrics_training(agent.results_location)
        
        if agent.brain.brain_memory.isFull and hasSavedMemory == False:
            hasSavedMemory = True
            saveMemory_v2(agent)
            if agent.args.mode == 'gather':
                print('Finished Gathering Data')
                break
        frame_saved = int(frame_saved / 4)
        if frame_saved > 400:
            frame_saved = 400
        if frame_saved < 60:
            frame_saved = 60
        if agent.brain.brain_memory.isFull:
            agent.brain.optimize_batch(frame_saved)
            #for i in range(frame_saved):
            #    if i % 10 == 0:
            #        print('\r', 'Learning', '(', i, '/', frame_saved, ')', end="")
            #    agent.brain.optimize()
            #print('\r', 'Learning', '(', frame_saved, '/', frame_saved, ')')
            
def playGameReal_ddqn(args, agent_func, screen_number=0, screen_id=-1):
    
    img_channels = args.hyper.img_channels
    env = Environment_realtime(args.env)
    action_dim = env.env.action_dim()
    state_dim = list(env.env.state_dim()) + [img_channels]
    
    agent = agent_func(args, state_dim, action_dim, getattr(models,args.hyper.model))
    
    if args.data:
        # Load Memory
        loadMemory_v2(agent, args.data)
        
        agent.mode = 'train'
        loaded_replays = int(agent.memory.size)
        print('Running', loaded_replays, 'replays')
        # Train on loaded memory
        for i in range(loaded_replays):
            agent.update_agent()
            if i % 1000 == 0:
                print(i, '/', loaded_replays, 'replays learned')
            if i % 100 == 0:
                agent.replay(debug=True)
            else:
                agent.replay(debug=False)
        
        agent.save_weights()
        
    time.sleep(1)

    while (True):
        frame, useRate, frame_saved = env.run(agent)
        
        agent.metrics.display_metrics(frame, useRate, agent.memory.total_saved, agent.epsilon)
        
        if agent.mode == 'train':
            print('Running', frame_saved, 'replays')
            for i in range(frame_saved):
                if i % 100 == 0:
                    agent.replay(debug=True)
                else:
                    agent.replay(debug=False)
                    
        if agent.h.save_rate < agent.save_iterator:
            agent.save_iterator -= agent.h.save_rate
            save_weights(agent)
            if agent.mode == 'train': # Fix this later, not correct
                agent.metrics.save_metrics(agent.results_location)
                agent.metrics.save_metrics_training(agent.results_location)

def gatherMemory(args, agent_func):
    emulator = OpenHexagonEmulator.HexagonEmulator(
                                                   args.screen.app,
                                                   args.screen.size,
                                                   args.screen.zoom
                                                  )
    img_rows , img_cols = emulator.capture_size[0], emulator.capture_size[1]
    img_channels = args.hyper.img_channels
    state_dim = [img_channels, img_rows, img_cols]
    action_dim = emulator.action_dim
    
    agent = agent_func(args, state_dim, action_dim)
    
    env = Environment_realtime(emulator)
    print('Gathering', agent.memory.max_size, 'states:')
    while (True):
        frame, useRate, frame_saved = env.run(agent)
        
        #agent.metrics.display_metrics(frame, useRate, agent.memory.total_saved, agent.epsilon)
        print(agent.memory.size, '/', agent.memory.max_size)
        
        
        if agent.memory.isFull:
            return saveAll(agent)