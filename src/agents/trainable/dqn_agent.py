from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from src.agents.trainable.trainable_agent import TrainableAgent
from src.environment.env_utils import get_illegal_actions


class DQNAgent(TrainableAgent):
    """
    DQNAgent: TrainableAgent that aims at estimating the optimal action-state
    values (Q-vals) to define the policy to follow. The estimation is done
    using a Deep Q-Network (DQN).

    The training phase of the model will be carried out in a dedicated python
    notebook, not within this class implementation (refer to 'src/train')

    Connect4 has a vertical advantage that can be used by the DQN to hopefully
    get more stable results. When computing the Q-vals (network forward pass),
    we can also create the symmetric board and predict the symmetric Q-values.
    Then, we can average the Q-value for each pair of symmetric actions and
    prevent the network from overestimating the higher values. This behaviour
    is controlled by the 'avg_symmetric_q_vals' attribute.
    """

    def __init__(self,
                 name: str = 'DQN Agent',
                 avg_symmetric_q_vals: bool = False,
                 **kwargs) -> None:
        """
        Initialize a DQNAgent instance

        :param name: DQNAgent name
        :param avg_symmetric_q_vals: whether to average symmetric Q-vals
        :param kwargs: 'exploration_rate' and 'allowed_illegal_actions'
        """

        super(DQNAgent, self).__init__(name=name, **kwargs)
        self.avg_symmetric_q_vals = avg_symmetric_q_vals
        self.model.second_head = torch.nn.Sequential()  # just one head

    def get_q_vals(self, obs: np.ndarray) -> torch.Tensor:
        """
        Performs a forward pass and returns the output (Q-vals).
        If 'avg_symmetric_q_vals', averages the symmetric Q-vals to get a more
        stable results. If not 'allow_illegal_actions', set the Q-vals of
        illegal actions to minus infinite. This way they will never be chosen
        in the argmax operation.

        :param obs: environment observation (game board)
        :return: Q-vals. state-action value for each action
        """

        model_input = self.model.obs_to_model_input(obs=obs)
        with torch.no_grad():
            q_vals = self.model(model_input).squeeze()

        if self.avg_symmetric_q_vals:
            sym_obs = np.flip(obs, axis=-1)
            sym_model_input = self.model.obs_to_model_input(obs=sym_obs)
            with torch.no_grad():
                sym_q_vals = self.model(sym_model_input).squeeze()
            q_vals = (q_vals + sym_q_vals.flip(dims=[0])) / 2

        if not self.allow_illegal_actions:
            illegal_actions = get_illegal_actions(board=obs)
            q_vals[illegal_actions] = - torch.inf
        return q_vals

    def get_exploitation_policy(self, obs: np.ndarray) -> Categorical:
        """
        Returns the policy to follow in order to exploit the environment
        in the given observation. The DQNAgent takes the action with the
        highest expected return. The DQN exploitation policy is deterministic.
.
        :param obs: environment observation (game board)
        :return: exploitation policy
        """

        q_vals = self.get_q_vals(obs=obs)
        best_action = q_vals.max(0)[1]
        probs = torch.zeros_like(q_vals)
        probs[best_action] = 1
        policy = Categorical(probs=probs)
        return policy

    def get_policy_scores_to_visualize(self, obs: np.ndarray) -> Tuple:
        """
        Returns the 'score' that the Agent assigns to each action in the
        given observation. This method is intended to be called within the User
        Interface (refer to the 'src/game' directory) so the players can
        visualize the logic underlying the Agent's behaviour. The DQNAgent uses
        the predicted Q-vals.

        :param obs: environment observation
        :return: q-vals
        """

        q_vals = self.get_q_vals(obs=obs)
        return tuple(q_vals.cpu().tolist())


if __name__ == "__main__":
    # DEMO
    from src.models.custom_network import CustomNetwork
    from src.environment.connect_game_env import ConnectGameEnv

    model = CustomNetwork(conv_block=[[32, 4, 0], 'relu'],
                          fc_block=[64, 'relu'],
                          first_head=[64, 'relu', 7])
    agent = DQNAgent(model=model,
                     avg_symmetric_q_vals=True,
                     exploration_rate=0)
    print(agent.name)
    print(agent.model)
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
