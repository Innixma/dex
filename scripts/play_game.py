# By Nick Erickson
# Contains functions for game play loops

from environment import Environment_gym, Environment_realtime
from data_utils import saveAll, saveMemory, saveClass, loadClass, loadMemory, save_weights
import time
import OpenHexagonEmulator
import models

def run(args, agent):   
    if args.env == 'real':
        playGameReal(args, agent)
    elif args.env == 'gym':
        playGameGym(args, agent)
    elif args.env == 'memory': 
        gatherMemory(args, agent)
    else:
        pass

def playGameGym(args, agent_func):
    env = Environment_gym(args.game)
    state_dim  = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.n
    state_dim = [state_dim]
    agent = agent_func(args, state_dim, action_dim)
    
    iteration = 0
    while (True):
        iteration += 1
        
        R, useRate = env.run(agent)
        
        if agent.memory.total_saved > agent.h.observe:
            if agent.mode == 'observe':
                agent.mode = 'train'
                print('training')
                time.sleep(0.5)
                
        if agent.mode == 'train':
            if iteration % 10 == 0:
                agent.replay(debug=True)
            else:
                agent.replay(debug=False)
            
            if iteration % 10 == 0:
                print("Step:", agent.memory.total_saved, ", Total reward:", R)
            
        #agent.metrics.display_metrics(frame, useRate, agent.memory.total_saved, agent.epsilon)
        
        if agent.h.save_rate < agent.save_iterator:
            agent.save_iterator -= agent.h.save_rate
            save_weights(agent)
            #if agent.mode == 'train': # Fix this later, not correct
                    #agent.metrics.save_metrics(agent.results_location)
                    #agent.metrics.save_metrics_training(agent.results_location)
                    
def playGameReal(args, agent_func, screen_number=0, screen_id=-1):
    
    emulator = OpenHexagonEmulator.HexagonEmulator(
                                                   args.screen.app,
                                                   args.screen.size,
                                                   args.screen.zoom,
                                                   screen_id,
                                                   screen_number
                                                  )
    img_rows , img_cols = emulator.capture_size[0], emulator.capture_size[1]
    img_channels = args.hyper.img_channels
    state_dim = [img_rows, img_cols, img_channels]
    action_dim = emulator.action_dim
    
    agent = agent_func(args, state_dim, action_dim, models.buildmodel_CNN_v3)
    
    if args.data != 'default':
        # Load Memory
        loadMemory(agent, args.data)
        
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
    
    env = Environment_realtime(emulator)
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