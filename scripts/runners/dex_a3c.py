#!/usr/bin/env python
# By Nick Erickson
# A3C Implementation

# Run with TensorFlow 1.1.0 and Keras v2.03

# %%

# Experimental
# import threading

import copy

from agents.a3c import agent_a3c
from environments import play_game
# Utilities
from parameters import hex


def main(args):
    play_game.run(args, agent_a3c.Agent)
    # TODO: Use finally clause to save model weights

if __name__ == "__main__":
    args = hex.base_a3c

    args = copy.deepcopy(args)
    main(args)

