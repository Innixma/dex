# Deep Hexagon (WIP)

Deep Hexagon (dex) is the first reinforcement learning environment toolkit specialized for continual learning. It contains hundreds of environments ranging drastically in difficulty.

This is a deep learning repository containing state-of-the-art algorithms that run out of the box on Keras.

![Demo](videos/dex_rotation3_222s_inc.gif)

![Demo2](videos/dex_saliency_showcase.gif)

It is compatible with OpenAI Gym, as well as a custom environment 'Open Hexagon' played with screen pixel information, which is included with the repo.

It can also be generalized to any program, given a proper emulator. (WIP)

- Videos: https://www.youtube.com/channel/UCvC0mlbgqEURZF7IkQKfbYg
- License: MIT

## Requirements

- Runs on Windows and Linux (Linux Open Hexagon Environment WIP).
- Python 3.5
- [Tensorflow v1.0.1 (GPU)](https://github.com/tensorflow/tensorflow)
- [Keras v2.0.2](https://github.com/fchollet/keras)

### Optional Requirements

- OpenAI Gym

### Optional Visualization Requirements (For running visualization.py)

- cv2
- vis
- imageio

## Algorithms

- DDQN | dex_ddqn.py
- ACER | dex_a3c.py

## Setup Open Hexagon Environment

- Extract OpenHexagonV1.92.7z and launch the game to the level you wish to learn.
- Run the script of the desired algorithm and it will detect and begin playing the game.

## Run

- Run dex_ddqn.py for DDQN
- Run dex_a3c.py for ACER

Currently, you will need to edit the code in these files to run the specific parameters you want. (WIP)

Note: ACER here is not identical to the paper, and is my own implementation.

Visualization is based on saliency.