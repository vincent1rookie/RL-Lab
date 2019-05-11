# RL-Lab

This repository is to store some implementation on basic Reinforcement learning algorithms. The task is to find a solution for LunarLander-v2 gym environment. 

## Project Detail

Action space covers 4 discrete actions. 

1. Do nothing 
2. fire left engine
3. fire main engine
4. fire right engine

Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. 

## Instruction

The model is trained on Google Colab, using tensorflow. In order to run the code locally, you may need to configure your environment.


