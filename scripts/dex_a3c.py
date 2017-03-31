#!/usr/bin/env python

# Run with Tensorflow v12 and Keras v2.02

# Done: Target Network https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/
# TODO: Prioritized Experience Replay
# TODO: Random agent saving memory/loading memory (Mostly done)
    
from __future__ import print_function

#%%

# Experimental
#import threading

# Utilities
import param_const
import play_game
import agent_a3c

#%%

def runSimulation(args):
     
    if args.env == 'real':
        hyper = args.hyper
        screen = args.screen
        play_game.playGameReal(args, screen, hyper, agent_a3c.Agent)
        
    elif args.env == 'gym':
        hyper = args.hyper
        game = 'CartPole-v0'
        
        if args.game != 'default':
            game = args.game

        play_game.playGameGym(args, game, hyper, agent_a3c.Agent)
        
    elif args.env == 'memory':
        hyper = args.hyper
        screen = args.screen
            
        play_game.gatherMemory(args, screen, hyper, agent_a3c.Agent)
    
    else:
        pass

def main(args):    
    runSimulation(args)
    # TODO: Use finally clause to save model weights

if __name__ == "__main__":
    args = param_const.hex_base
    #args = param_const.gym_cart
    main(args) 

