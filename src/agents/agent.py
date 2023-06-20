import random
from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from src.environment.env_utils import get_legal_actions


class Agent:
    """
    Basic Agent to play Connect4. Parent Class.
    Implements the functionalities that all the Agents in this project share.
    Some of the methods have to be filled by the children classes. Some of them
    might be overwritten to implement a specific technique that depends on a
    particular type of agent.

    - choose_action(obs, exploration_rate), the main method here:
        Since Connect4 is a perfect information game, what the Agent observes
        is the entire state of the environment: the current board. Each board
        value is {0: empty, 1: you, -1: opponent}, so the Agent must choose
        in which column it will drop one of its pieces (value 1). It also
        implements an epsilon-greedy strategy to deal with the exploration-
        exploitation dilemma. The Agent has an exploration_rate by default (as
        an attribute) that is used when a specific exploration_rate is not
        provided in the 'choose_action' method.
    """

    def __init__(self,
                 name: str = "Agent",
                 exploration_rate: float = 0.,
                 allow_illegal_actions: bool = False) -> None:
        """
        Initialize and Agent instance

        :param name: name of the Agent
        :param exploration_rate: default prob of exploring the environment
        :param allow_illegal_actions: whether to allow illegal actions
        """

        self.name = name
        self.exploration_rate = exploration_rate
        self.allow_illegal_actions = allow_illegal_actions

    def get_exploration_policy(self, obs: np.ndarray) -> Categorical:
        """
        Returns the random policy to follow in order to explore the environment
        from the given observation: a uniform policy over the available actions

        :param obs: game board
        :return: distribution over the action space to explore the environment
        """

        n_actions = obs.shape[-1]
        if self.allow_illegal_actions:
            probs = torch.full([n_actions], fill_value=1 / n_actions)
        else:  # only legal actions are allowed
            probs = torch.zeros(n_actions)
            legal_actions = get_legal_actions(board=obs)
            probs[legal_actions] = 1 / len(legal_actions)
        policy = Categorical(probs=probs)
        return policy

    def get_exploitation_policy(self, obs: np.array) -> Categorical:
        """
        Returns the policy to follow in order to exploit the environment
        in the given observation. It depends on the child Agent class.

        :param obs: environment observation (game board)
        :return: distribution over the action space to exploit the environment
        """
        pass

    def get_policy_scores_to_visualize(self, obs: np.ndarray) -> Tuple:
        """
        Returns the 'score' that the Agent assigns to each action in the
        given observation. This method is intended to be called within the User
        Interface (refer to the 'src/game' directory) so the user (human) can
        visualize the logic underlying the Agent's behaviour.

        These 'scores' completely depend on the children Agent classes. For
        instance, in value-based Agents, the 'scores' are be the Q-values. On
        the other hand, in policy-based methods the 'scores' are the
        probabilities (implemented here by default).

        A child Agent class have to overwrite this method if their scores are
        not their exploitation policy.

        :return: scores that define the policy to follow
        """

        policy = self.get_exploitation_policy(obs=obs)
        return tuple(policy.probs.cpu().tolist())

    def _process_exploration_rate(self, exploration_rate: float) -> float:
        """
        Process the exploration rate (probability of exploring the environment)

        :param exploration_rate: value between [0,1]
        :return: processed exploration_rate
        """

        if exploration_rate is not None:  # if provided, use the given value
            return exploration_rate
        else:  # else, use the default value (defined within the __init__)
            return self.exploration_rate

    def choose_action(self,
                      obs: np.ndarray,
                      exploration_rate: float = None,
                      ) -> int:
        """
        Epsilon-greedy strategy to choose action based on the given obs.
        An observation is a game board with values {0: empty, 1: active player,
        -1: opponent}. The Agent chooses where to drop a '1' piece.

        :param obs: environment observation (game board)
        :param exploration_rate: probability of exploring the environment
        :return: action and (optional) policy used to sample actions from
        """

        exploration_rate_ = self._process_exploration_rate(exploration_rate)
        if random.random() < exploration_rate_:  # exploration
            policy = self.get_exploration_policy(obs=obs)
        else:  # exploitation
            policy = self.get_exploitation_policy(obs=obs)
        action = policy.sample()
        return action.item()

    def get_transition(self,
                       state: np.ndarray,
                       exploration_rate: float = None) -> dict:
        """
        Returns the basic transition that represents the agent taking action in
        the given observation: (state, action, log_prob)

        NOTE: the 'log_prob' is only useful for policy-gradient Agents. So this
        type of agents may overwrite this method to include it.
        NOTE: the 'reward', 'next_state', and 'done' elements have to be filled
        outside this method.

        :param state: environment observation (game board)
        :param exploration_rate: probability of exploring the environment
        :return: transition (s,a) or (s,a,log_prob)
        """

        with torch.no_grad():
            action = self.choose_action(obs=state,
                                        exploration_rate=exploration_rate)
        return {'state': state.copy(), 'action': action, 'log_prob': None}

    @staticmethod
    def _get_basic_symmetric_transition(transition: dict):
        """
        Connect4 has a vertical board symmetry that can be used to double the
        number of transitions and also to help the agent learn this symmetry.
        NOTE: the 'log_prob' has to be computed outside this method

        :param transition: transitions that will be horizontally flipped
        :return: basic symmetric transition
        """

        symmetric = {}
        if 'state' in transition:
            symmetric['state'] = np.flip(transition['state'], axis=-1)
        if 'action' in transition:
            symmetric['action'] = 6 - transition['action']
        if 'reward' in transition:
            symmetric['reward'] = transition['reward']
        if 'next_state' in transition:
            symmetric['next_state'] = np.flip(transition['next_state'],
                                              axis=-1)
        if 'done' in transition:
            symmetric['done'] = transition['done']
        if 'log_prob' in transition:
            symmetric['log_prob'] = None  # it must be re-computed by the Agent
        return symmetric

    def get_symmetric_transition(self, transition: dict) -> dict:
        """
        Connect4 has a vertical board symmetry that can be used to double the
        number of transitions and also to help the agent learn this symmetry.
        NOTE: if the Agent is policy-based, it may overwrite this method to
        add the natural logarithm of the probability.

        :param transition: transitions that will be horizontally flipped
        :return: basic symmetric transition
        """

        symmetric = self._get_basic_symmetric_transition(transition=transition)
        return symmetric


if __name__ == "__main__":
    agent = Agent()
    print(agent.name)
    print("ok")
