import numpy as np
from torch.distributions import Categorical

from src.agents.baselines.baseline_agent import BaselineAgent


class RandomAgent(BaselineAgent):
    """
    RandomAgent: BaselineAgent that selects one of the columns at random
    """

    def __init__(self, name: str = "Random Agent", **kwargs) -> None:
        super(RandomAgent, self).__init__(name=name, **kwargs)

    def get_exploitation_policy(self, obs: np.ndarray) -> Categorical:
        """
        Returns the policy to follow in order to exploit the environment
        in the given observation. The RandomAgent only knows exploration.

        :param obs: environment observation (game board)
        :return: distribution over the action space to exploit the environment
        """

        policy = self.get_exploration_policy(obs=obs)
        return policy


if __name__ == "__main__":
    # DEMO
    from src.environment.connect_game_env import ConnectGameEnv

    agent = RandomAgent()
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
