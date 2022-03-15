# Krishan's instructions to run

## Setup
- create a new conda environment.
- mac
  - install brew on macbook
  - brew install swig and other deps
- linux

    sudo apt update
    sudo apt install swig

- install dependeics
    pip install gym[box2d]==0.23.0 torch numpy matplotlib

## Run heuristic model

    python lunar_lander.py

## Run reinforcement learning models 

    python trainLander.py