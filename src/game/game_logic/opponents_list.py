from typing import List

import torch

from src.agents.agent import Agent
from src.agents.baselines.random_agent import RandomAgent
from src.agents.baselines.n_step_lookahead_agent import NStepLookaheadAgent
from src.agents.trainable.trainable_agent import CustomNetwork
from src.agents.trainable.dueling_dqn_agent import DuelingDQNAgent
from src.agents.trainable.dqn_agent import DQNAgent
from src.agents.trainable.pg_agent import PGAgent


def get_opponents_list() -> List[Agent]:
    """
    Returns the list of candidates to be the opponent of the game.
    This function is used when initializing a ConnectGameMenu instances.

    :return: List of different Agent instances
    """

    # 1) Random Agent
    a1 = RandomAgent(name="Random Agent")

    # 2) 1-step LookAhead Agent
    a2 = NStepLookaheadAgent(
        n=1,
        prefer_central_columns=True,
        name="1-Step Lookahead Agent"
    )

    # 3) 2-step LookAhead Agent
    a3 = NStepLookaheadAgent(
        n=2,
        prefer_central_columns=True,
        name="2-Step Lookahead Agent"
    )

    # 4) 3-step LookAhead Agent
    a4 = NStepLookaheadAgent(
        n=2,
        prefer_central_columns=True,
        name="3-Step Lookahead Agent"
    )

    # 5) best DQN agent
    model5 = CustomNetwork.from_architecture(
        file_path='./src/models/architectures/cnet128.json'
    )
    model5.second_head = torch.nn.Sequential()
    model5.load_weights('./src/models/saved_models/best_dqn.pt')
    model5.eval()
    a5 = DQNAgent(model=model5,
                  avg_symmetric_q_vals=True,
                  name='Vanilla DQN Agent')

    # 6) best Dueling DQN agent
    model6 = CustomNetwork.from_architecture(
        file_path='./src/models/architectures/cnet128.json'
    )
    model6.load_weights('./src/models/saved_models/best_dueling_dqn.pt')
    model6.eval()
    a6 = DuelingDQNAgent(model=model6,
                         avg_symmetric_q_vals=True,
                         name='Dueling DQN Agent')

    # 7) best PPO agent
    model7 = CustomNetwork.from_architecture(
        file_path='./src/models/architectures/cnet128.json'
    )
    model7.load_weights('./src/models/saved_models/best_ppo.pt')
    model7.eval()
    a7 = PGAgent(model=model7,
                 avg_symmetric_probs=True,
                 stochastic_mode=True,
                 name='PPO Agent')

    # 8) best PPO agent
    model8 = CustomNetwork.from_architecture(
        file_path='./src/models/architectures/cnet128.json'
    )
    model8.load_weights('./src/models/saved_models/best_reinforce.pt')
    model8.eval()
    a8 = PGAgent(model=model7,
                 avg_symmetric_probs=True,
                 stochastic_mode=True,
                 name='REINFORCE Agent')
    return [a1, a2, a3, a4, a5, a6, a7, a8]


if __name__ == '__main__':
    import os
    os.chdir('/home/marc/Escritorio/RL-connect4')

    opponents_list = get_opponents_list()
    for opp in opponents_list:
        print(opp.name, '->', opp)
