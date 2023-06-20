import numpy as np
import torch
from torch.distributions import Categorical

from src.agents.trainable.trainable_agent import TrainableAgent
from src.environment.env_utils import get_illegal_actions


class PGAgent(TrainableAgent):
    """
    PGAgent (Policy-Gradient Agent): TrainableAgent that aims at estimating the
    optimal policy to follow. The estimation is done using a Neural Network.
    This class is intended to be used with the Policy-Gradient algorithms:
        - REINFORCE with a baseline
        - Proximal Policy Optimization

    The exploration / exploitation trade-off is controlled by the 'exploration_
    rate' and 'stochastic_mode' attributes. The 'exploration_rate' is the
    probability to explore (at random) the environment at any step.
    If the 'stochastic_mode' is True, the policy is stochastic (get actions by
    sampling), otherwise it is deterministic (get greedy actions by taking the
    argmax(prob)).

    Connect4 has a vertical advantage that can be used by the PGAgent to
    hopefully get more stable results. When computing the logits (network
    forward pass), we can also create the symmetric board and predict the
    symmetric logits. Then, we can average both values for each pair of
    symmetric actions and prevent the network from overestimating the higher
    values. This behaviour is controlled by the 'avg_symmetric_probs' attribute.

    The network ('model' attribute) has two heads. The first one outputs the
    policy (actor), and the second one outputs the state-value (critic). Only
    the actor head is used when choosing actions. The critic head is only used
    in the training process.

    The training phase of the model will be carried out in a dedicated python
    notebook, not within this class implementation (refer to 'src/train').
    """

    def __init__(self,
                 stochastic_mode: bool = True,
                 avg_symmetric_probs: bool = True,
                 name: str = 'Policy-Gradient Agent',
                 **kwargs) -> None:
        """
        Initialize a PGAgent

        :param stochastic_mode: whether the policy is stochastic or deterministic
        :param avg_symmetric_probs: whether to average symmetric probabilities
        :param name: PGAgent name
        :param kwargs: 'exploration_rate' or 'allow_illegal_actions'
        """

        super().__init__(name=name, **kwargs)
        self.stochastic_mode = stochastic_mode
        self.avg_symmetric_probs = avg_symmetric_probs

    def get_log_prob(self, obs: np.ndarray, action: int) -> int:
        """
        Returns the natural logarithm of the probability of playing the given
        'action' in the given observation ('obs')

        :param obs: environment observation (game board)
        :param action: action to compute its log prob
        :return: log_prob for the given 'action'
        """

        model_input = self.model.obs_to_model_input(obs=obs)
        with torch.no_grad():
            logits, _ = self.model(model_input)  # critic head is not used
        policy = torch.distributions.Categorical(logits=logits)
        action_ = torch.tensor(action, device=self.model.device)
        log_prob = policy.log_prob(action_)
        return log_prob.item()

    def get_exploitation_policy(self, obs: np.ndarray) -> Categorical:
        """
        Returns the policy that must be followed to exploit the environment
        from the given observation. The PGAgent takes the action with the
        highest probability if 'stochastic_mode' is False. Else, it samples an
        action from the policy.
.
        :param obs: environment observation (game board)
        :return: exploitation policy (stochastic or deterministic)
        """

        model_input = self.model.obs_to_model_input(obs=obs)
        with torch.no_grad():
            logits, _ = self.model(model_input)
            logits = logits.squeeze()

        if self.avg_symmetric_probs:
            sym_obs = np.flip(obs, axis=-1)
            sym_model_input = self.model.obs_to_model_input(obs=sym_obs)
            with torch.no_grad():
                sym_logits, _ = self.model(sym_model_input)
                sym_logits = sym_logits.squeeze()
            logits = (logits + sym_logits.flip(dims=[0])) / 2

        if not self.allow_illegal_actions:
            illegal_actions = get_illegal_actions(board=obs)
            if len(illegal_actions) != 0:
                logits[illegal_actions] = - torch.inf
                if (logits != -torch.inf).sum().item() == 0:
                    # if no legal action has a non-zero prob, explore at random
                    return self.get_exploration_policy(obs=obs)

        if self.stochastic_mode:
            policy = Categorical(logits=logits)
        else:  # deterministic mode
            best_action = torch.argmax(logits)
            probs = torch.zeros_like(logits)
            probs[best_action] = 1
            policy = Categorical(probs=probs)
        return policy

    def get_transition(self,
                       state: np.ndarray,
                       exploration_rate: float = None) -> dict:
        """
        Returns the basic transition that represents the agent taking action in
        the given observation: (state, action, log_prob)

        NOTE: the 'reward', 'next_state', and 'done' elements have to be filled
        outside this method.

        :param state: environment observation (game board)
        :param exploration_rate: probability of exploring the environment
        :return: transition (s,a,log_prob)
        """

        with torch.no_grad():
            action = self.choose_action(obs=state,
                                        exploration_rate=exploration_rate)
            log_prob = self.get_log_prob(obs=state, action=action)
        return {'state': state.copy(), 'action': action, 'log_prob': log_prob}

    def get_symmetric_transition(self, transition: dict) -> dict:
        """
        Connect4 has a vertical board symmetry that can be used to double the
        number of transitions and also to help the agent learn this symmetry.
        Compute the log_prob for the symmetric transition.

        :param transition: transitions that will be horizontally flipped
        :return: basic symmetric transition
        """

        symmetric = self._get_basic_symmetric_transition(transition=transition)
        symmetric['log_prob'] = self.get_log_prob(obs=symmetric['state'],
                                                  action=symmetric['action'])
        return symmetric


if __name__ == "__main__":
    # DEMO
    from src.models.custom_network import CustomNetwork
    from src.environment.connect_game_env import ConnectGameEnv

    model = CustomNetwork(conv_block=[[32, 4, 0], 'relu'],
                          fc_block=[64, 'relu'],
                          first_head=[64, 'relu', 7],
                          second_head=[64, 'relu', 1])
    agent = PGAgent(model=model,
                    stochastic_mode=True,
                    avg_symmetric_probs=True)
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
