import numpy as np
import torch
from torch.distributions import Categorical

from src.agents.baselines.baseline_agent import BaselineAgent
from src.environment.env_utils import get_legal_actions


class LeftmostAgent(BaselineAgent):
    """
    LeftMostAgent: BaselineAgent that selects the leftmost column of the board
    """

    def __init__(self, name: str = "Leftmost Agent", **kwargs) -> None:
        super(LeftmostAgent, self).__init__(name=name, **kwargs)

    def get_exploitation_policy(self, obs: np.ndarray) -> Categorical:
        """
        Returns the policy to follow in order to exploit the environment
        in the given observation: select the left-most column

        :param obs: environment observation (game board)
        :return: distribution over the action space to exploit the environment
        """

        probs = torch.zeros(obs.shape[-1])
        if self.allow_illegal_actions:
            probs[0] = 1
        else:
            legal_actions = get_legal_actions(board=obs)
            probs[legal_actions[0]] = 1
        policy = Categorical(probs=probs)
        return policy


if __name__ == "__main__":
    # DEMO
    from src.environment.connect_game_env import ConnectGameEnv

    agent = LeftmostAgent()
    print(agent.name)
    obs = ConnectGameEnv.random_observation()
    print('obs =\n', obs)
    transition = agent.get_transition(state=obs)
    policy = agent.get_exploitation_policy(obs=obs)
    print('policy =\n', policy.probs)
    vis_policy = agent.get_policy_scores_to_visualize(obs=obs)
    print('visualization policy =\n', vis_policy)
    print('action =', transition['action'])
    print('\ntransition =\n', transition)
    print('\nsymmetric transition=\n', agent.get_symmetric_transition(transition))
