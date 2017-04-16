# Deep Hexagon

This is a deep learning repository for playing the game Open Hexagon with screen pixel information. It can also be generalized to any program, given a proper emulator.

:Videos: https://www.youtube.com/channel/UCvC0mlbgqEURZF7IkQKfbYg
:License: MIT

# Requirements

Runs on Windows, currently working on a Linux version.

Python 3.5

Tensorflow 0.12 (GPU) (TODO: Upgrade to Tensorflow r1.0 at some point)

Keras v2.0.2

# Algorithms

:DDQN: dex_ddqn.py
:ACER: dex_a3c.py

# Setup

Extract OpenHexagonV1.92.7z and lauch the game to the level you wish to learn.

Then run the script of the desired algorithm and it will detect and begin playing the game.

# Run

Run dex_ddqn.py for DDQN

Run dex_a3c.py for ACER

Currently, you will need to edit the code in these files to run the specific parameters you want.

Note: ACER here is not identical to the paper, and is my own implementation.