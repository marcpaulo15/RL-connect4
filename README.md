# RL-connect4

The main objective of this project is to compare how different Reinforcement Learning algorithms learn to play Connect4. 

We propose a training pipeline that combines Supervised Learning and self-play Reinforcement Learning. 
  - **Part 1)** A convolutional neural network learns to mimic the actions of a mid-level player
    - supervised learning, one-class classification
    - our mid-level player is a hand-crafted heuristic (1-Step minimax search)
  - **Part2)** Starting from the pre-trained network (with some knowledge of the game), we apply different Deep Reinforcement Learning algorithms (separately) to improve the performance of the pre-trained network (from Part 1).

We transfer the learning from Part 1 to Part 2. The pre-trained convolutional block (from Part 1) is regarded as a feature extractor and is frozen in Part 2. The rest of Fully Connected layers are trained to solve the RL task of each Deep RL algorithm. With this approach, the Reinforcement Learning algorithms do not have to learn from scratch. The training becomes more stable because the first self-play games are not random.

![Our training pipeline (transfer learning)](https://github.com/marcpaulo15/RL-connect4/assets/94222660/8b0f851b-2092-4ec4-a81b-4c80e0cc11ec)


To evaluate the trained agents, they compete against each other, so we can compare them and conclude which algorithm has achieved the highest level of play. 

Finally, we present a simple User Interface to let the user play Connect 4 against all the agents trained in this project.

**KEY WORDS**: connect4, zero-sum games, deep learning, supervised learning, transfer learning, reinforcement learning, self-play, Proximal Policy Optimization, PPO, REINFORCE, Deep Q-Network, DQN, Dueling Deep Q-Network, Dueling DQN.

## Agents

Implementation of the agents. There are two types of agents: Baseline Agents, and Trainable Agents.
- Baseline Agents: implement a non-trainable heuristic to play the game.
  - Random Agent: selects columns at random.
  - Leftmost Agent: selects the leftmost column.
  - N-Step Lookahead Agent: simulates N turns ahead and runs a minimax search to select actions.
- Trainable Agents: implement a trainable model (neural network) to play the game.
  - Vanilla and Dueling DQN Agents: use a model to estimate the optimal Q-values.
  - REINFORCE and PPO Agents: use a model to estimate the optimal policy.

## Data

Implementation of the classes to store and process training data (games).
  - part1 data: synthetic dataset used in the Supervised Learning task.
  - part1 dataset generator: notebook to generate the synthetic dataset used in the Supervised Learning task.
    - 200k (state, actions) pairs played by the 1-Step Lookahead Agent (baseline, mid-level player).
    - Supervised Learning task (classification): predict the actions of the 1-StepLA at each turn.
  - replay memory: a class that serves as an Experience Replay Memory or as an Episode Buffer.
    - backpropagates the last rewards to the intermediate steps.

## Environment
Implementation of the Connect4 game as a Reinforcement Learning environment.
  - connect game env: implements the environment (OpenAI gym structure).
  - env utils: some auxiliary functions used by the environment and the agents.

## Eval
Implementation of the competition system to evaluate the agents.
  - run episode: implements the logic to let two agents play a Connect4 game.
  - competition: implements the competition system to let two agents play several Connect4 games.
![ranking](https://github.com/marcpaulo15/RL-connect4/assets/94222660/a728072e-b2dd-4c23-9596-70bc287c1dcb)

## Game
Implementation of a simple User Interface (using Pygame) to let the readers play against the best agents defined in this project.
  - game config: configuration files to customize the application.
  - game logic: classes to implement the logic of the application.
  - connect game main: run this Python script to run the application and play the game.
![Game Menu](https://github.com/marcpaulo15/RL-connect4/assets/94222660/a25a449a-28c0-4569-b66f-569aa909c2f9)
![Example of an ongoing game](https://github.com/marcpaulo15/RL-connect4/assets/94222660/2d9ad3b1-1b72-4a5f-b575-61d77755e8b2)


## Models
Implementation of the neural network architecture. Contains the best models of each agent.
  - architectures: list of predefined neural network architectures.
  - saved models: weights and training hyper-parameters of the best models trained in this project.
  - custom network: a class to implement a wide range of different neural network architectures

## Train
Implements the training pipeline to train each agent (Supervised Learning, self-play Reinforcement Learning)
  - part 1 supervised learning: a policy network learns to predict the actions of a mid-level player:
    - see also: Data/part1 data; Data/part1 dataset generator; Agents/1-Step Lookahead Agent.
  - ppo training: a PPO Agent learns a policy to maximize the expected return.
    - depends on: Train/part 1 supervised learning.
  - reinforce training: a REINFORCE Agent learns a policy to maximize the expected return.
    - depends on: Train/part 1 supervised learning.
  - vanilla and dueling dqn training: a Vanilla DQN or Dueling DQN Agent learns the Q-values.
    - depends on: Train/part 1 supervised learning.
    
    
