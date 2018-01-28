# By Nick Erickson
# Contains functions for game play loops

import time

from environments.environment import EnvironmentGym, EnvironmentGymRgb, EnvironmentRealtime, \
    EnvironmentRealtimeA3C
from utils.data_utils import save_all, save_weights, save_memory_v2, load_memory_v2, save_memory_subset

try:
    from environments.hexagon import open_hexagon_emulator
except:
    print("Can't import openHexagonEmulator, not a Windows environment, skipping...")
from agents import models

import threading

MULTITHREAD = 7
MULTITHREAD_BRAIN = 1


def run(args, agent):
    if args.env.type == 'real':
        if args.algorithm == 'ddqn':
            play_game_real_ddqn(args, agent)
        elif args.algorithm == 'a3c':
            play_game_real_a3c(args, agent)
    elif args.env.type == 'gym':
        if args.algorithm == 'ddqn':
            play_game_gym_ddqn(args, agent)
        elif args.algorithm == 'a3c':
            play_game_gym_a3c(args, agent)
    elif args.env.type == 'memory':
        gather_memory(args, agent)
    else:
        pass


def init_agents(args, agent_func):
    agents = []
    envs = []
    brain = None
    for i in range(MULTITHREAD):
        if args.hyper.img_channels > 1:
            env = EnvironmentGymRgb(args.env, i)
            state_dim = env.env.state_dim + [args.hyper.img_channels]
        else:
            env = EnvironmentGym(args.env, i)
            state_dim = env.env.state_dim
        action_dim = env.env.action_dim
        agent = agent_func(args, state_dim, action_dim, getattr(models, args.model), brain=brain, idx=i)
        brain = agent.brain
        envs.append(env)
        agents.append(agent)

    return agents, envs


def play_game_gym_a3c_multithread_init(args, agent_func):
    agents, envs = init_agents(args, agent_func)
    brain = agents[0].brain
    threads = []
    for i in range(MULTITHREAD):
        threads.append(MultithreadAgent(agents[i], envs[i]))
        threads[-1].daemon = True

    threads_brain = []

    for i in range(MULTITHREAD_BRAIN):
        threads_brain.append(MultithreadBrain(brain))
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


class MultithreadAgent(threading.Thread):
    stop_signal = False

    def __init__(self, agent, env):
        threading.Thread.__init__(self)
        self.agent = agent
        self.env = env

    def run(self):
        iteration = 0
        while not self.stop_signal:
            iteration += 1

            R, use_rate = self.env.run(self.agent)

            if self.agent.mode == 'train':
                if iteration % 10 == 0:
                    print("Step:", self.agent.memory.total_saved, ", Total reward:", R, "idx:", self.agent.idx)

            # agent.metrics.display_metrics(frame, use_rate, agent.memory.total_saved, agent.epsilon)

            if self.agent.idx == 0:
                if self.agent.h.save_rate < self.agent.save_iterator:
                    self.agent.save_iterator -= self.agent.h.save_rate
                    save_weights(self.agent, self.agent.brain.brain_memory.total_saved)
                    # self.agent.metrics.a3c.graph_all(self.agent.results_location)

            # playGameGym_a3c_multithread(self.agent, self.env)

    def stop(self):
        self.stop_signal = True


class MultithreadBrain(threading.Thread):
    stop_signal = False

    def __init__(self, brain):
        threading.Thread.__init__(self)
        self.brain = brain

    def run(self):
        while not self.stop_signal:
            self.brain.optimize_batch_full_multithread()

    def stop(self):
        self.stop_signal = True


def play_game_gym_a3c_multithread(agent, env):
    iteration = 0
    while True:
        iteration += 1

        R, use_rate = env.run(agent)

        if agent.mode == 'train':
            if iteration % 10 == 0:
                print("Step:", agent.memory.total_saved, ", Total reward:", R, "idx:", agent.idx)

        # agent.metrics.display_metrics(frame, use_rate, agent.memory.total_saved, agent.epsilon)

        if agent.idx == 0:
            if agent.h.save_rate < agent.save_iterator:
                agent.save_iterator -= agent.h.save_rate
                save_weights(agent, agent.run_count)
                # agent.metrics.a3c.graph_all(agent.results_location)
                # if agent.mode == 'train': # Fix this later, not correct
                #         agent.metrics.runs.graph(agent.results_location)
                #         agent.metrics.save_metrics_training(agent.results_location)


def play_game_gym_a3c(args, agent_func):

    play_game_gym_a3c_multithread_init(args, agent_func)

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
                    #agent.metrics.runs.graph(agent.results_location)
                    #agent.metrics.save_metrics_training(agent.results_location)
    """


def play_game_gym_ddqn(args, agent_func):
    if args.hyper.img_channels > 1:
        env = EnvironmentGymRgb(args.env)
        state_dim = env.env.state_dim + [args.hyper.img_channels]
    else:
        env = EnvironmentGym(args.env)
        state_dim = env.env.state_dim
    action_dim = env.env.action_dim

    agent = agent_func(args, state_dim, action_dim, getattr(models, args.model))

    iteration = 0
    while True:
        iteration += 1

        R, use_rate = env.run(agent)

        if agent.memory.total_saved > agent.h.extra.observe:
            if agent.mode == 'observe':
                agent.mode = 'train'
                print('Training...')
                time.sleep(0.5)

        if agent.mode == 'train':
            if iteration % 10 == 0:
                print("Step:", agent.memory.total_saved, ", Total reward:", R)

        # agent.metrics.display_metrics(frame, use_rate, agent.memory.total_saved, agent.epsilon)

        if agent.h.save_rate < agent.save_iterator:
            agent.save_iterator -= agent.h.save_rate
            save_weights(agent)
            # if agent.mode == 'train': # Fix this later, not correct
            #     agent.metrics.runs.graph(agent.results_location)
            #     agent.metrics.save_metrics_training(agent.results_location)

"""
def playGameReal_a3c_multithread_init(args, agent_func):
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


def play_game_real_a3c_incremental_init(args, agent_func, state_dim, action_dim):
    agent = agent_func(args, state_dim, action_dim, getattr(models, args.model))
    has_saved_memory = False
    max_frame_saved = 300
    return agent, has_saved_memory, max_frame_saved


def play_game_real_a3c_incremental(agent, env, state_dim, action_dim, has_saved_memory, max_frame_saved):

    pointer_start = agent.brain.brain_memory.cur_index + 0
    frame, use_rate, frame_saved = env.run(agent)
    pointer_end = agent.brain.brain_memory.cur_index + 0
    agent.metrics.display_metrics(frame, use_rate, agent.memory.total_saved, agent.epsilon)

    if frame_saved > max_frame_saved:
        print('New max time!')
        max_frame_saved = frame_saved

        save_memory_subset(agent, pointer_start, pointer_end, frame_saved, skip=1)
        save_weights(agent, 'frame_' + str(frame_saved))

    if agent.h.save_rate < agent.save_iterator:
        agent.save_iterator -= agent.h.save_rate
        if agent.args.mode != 'gather':
            save_weights(agent, agent.run_count)
            agent.metrics.save(agent.results_location, 'metrics')
            agent.metrics.runs.graph(agent.results_location)

        # agent.metrics.save_metrics_v(agent.results_location)
        # agent.metrics.a3c.graph_all(agent.results_location)
        # agent.metrics.save_metrics_training(agent.results_location)

    if agent.brain.brain_memory.isFull and has_saved_memory is False:
        has_saved_memory = True
        save_memory_v2(agent)

    frame_saved = int(frame_saved)
    if frame_saved > 3000:
        frame_saved = 3000
    if frame_saved < 300:
        frame_saved = 300
    batch_count = int(90000/frame_saved)
    batch_count = 15  # 75
    if agent.brain.brain_memory.isFull:
        if agent.args.mode != 'gather' and agent.args.mode != 'run':
            agent.brain.optimize_batch(batch_count)

    return has_saved_memory, max_frame_saved


def play_game_real_a3c(args, agent_func, screen_number=0, screen_id=-1):

    img_channels = args.hyper.img_channels
    env = EnvironmentRealtimeA3C(args.env)
    action_dim = env.env.action_dim()
    state_dim = list(env.env.state_dim()) + [img_channels]
    # env = Environment_realtime_a3c(emulator, img_channels)

    print(state_dim)
    print(action_dim)

    agent = agent_func(args, state_dim, action_dim, getattr(models, args.model))

    has_saved_memory = False

    max_frame_saved = 300
    total_saved = 0
    while True:
        pointer_start = agent.brain.brain_memory.cur_index + 0
        frame, use_rate, frame_saved = env.run(agent)
        pointer_end = agent.brain.brain_memory.cur_index + 0
        agent.metrics.display_metrics(frame, use_rate, agent.memory.total_saved, agent.epsilon)

        if frame_saved > max_frame_saved:
            print('New max time!')
            max_frame_saved = frame_saved

            save_memory_subset(agent, pointer_start, pointer_end, frame_saved, skip=8)
            save_weights(agent, 'frame_' + str(frame_saved))

        if agent.h.save_rate < agent.save_iterator:
            agent.save_iterator -= agent.h.save_rate
            save_weights(agent, agent.run_count)
            agent.metrics.save(agent.results_location, 'metrics')
            agent.metrics.runs.graph(agent.results_location)
            # agent.metrics.save_metrics_v(agent.results_location)
            # agent.metrics.a3c.graph_all(agent.results_location)
            # agent.metrics.save_metrics_training(agent.results_location)

        if agent.brain.brain_memory.is_full and has_saved_memory is False:
            has_saved_memory = True
            save_memory_v2(agent)
            if agent.args.mode == 'gather':
                print('Finished Gathering Data')
                break
        frame_saved = int(frame_saved)
        if frame_saved > 1000:
            frame_saved = 1000
        if frame_saved < 300:
            frame_saved = 300
        if total_saved > 100000:
            frame_saved = int(frame_saved / 4)
        if agent.brain.brain_memory.is_full:
            total_saved += frame_saved
            agent.brain.optimize_batch(frame_saved)


def play_game_real_ddqn(args, agent_func, screen_number=0, screen_id=-1):

    img_channels = args.hyper.img_channels
    env = EnvironmentRealtime(args.env)
    action_dim = env.env.action_dim()
    state_dim = list(env.env.state_dim()) + [img_channels]

    agent = agent_func(args, state_dim, action_dim, getattr(models, args.hyper.model))

    if args.data:
        # Load Memory
        load_memory_v2(agent, args.data)

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

    while True:
        frame, use_rate, frame_saved = env.run(agent)

        agent.metrics.display_metrics(frame, use_rate, agent.memory.total_saved, agent.epsilon)

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
            if agent.mode == 'train':  # Fix this later, not correct
                agent.metrics.save(agent.results_location, 'metrics')
                agent.metrics.runs.graph(agent.results_location)
                agent.metrics.save_metrics_training(agent.results_location)


def gather_memory(args, agent_func):
    emulator = open_hexagon_emulator.HexagonEmulator(
                                                   args.screen.app,
                                                   args.screen.size,
                                                   args.screen.zoom
                                                  )
    img_rows , img_cols = emulator.capture_size[0], emulator.capture_size[1]
    img_channels = args.hyper.img_channels
    state_dim = [img_channels, img_rows, img_cols]
    action_dim = emulator.action_dim

    agent = agent_func(args, state_dim, action_dim)

    env = EnvironmentRealtime(emulator)
    print('Gathering', agent.memory.max_size, 'states:')
    while True:
        frame, use_rate, frame_saved = env.run(agent)

        # agent.metrics.display_metrics(frame, use_rate, agent.memory.total_saved, agent.epsilon)
        print(agent.memory.size, '/', agent.memory.max_size)

        if agent.memory.is_full:
            return save_all(agent)
