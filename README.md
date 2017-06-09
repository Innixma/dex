# Deep Hexagon (WIP)

Deep Hexagon (dex) is the first reinforcement learning environment toolkit specialized for continual learning, with an OpenAI gym like API. It contains hundreds of environments ranging drastically in difficulty, but using the same basic objectives and obstacles.

Currently this repository is a work in progress, and thus may not work as intended. A more indepth readme will be created once development has stabilized.

![Demo](videos/dex_rotation3_222s_inc.gif)

![Demo2](videos/dex_environment_showcase.gif)

Dex comes with state-of-the-art algorithms such as DDQN, A3C, and ACER to rapidly learn environments, and also supports integration with any other methods. The algorithms are compatible with OpenAI Gym, as well as the custom environment 'Open Hexagon' played with screen pixel information, which is included with the repo.

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

Note: ACER here is not identical to the paper, and is my own implementation. Visualization is based on saliency.