from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from src.agents.baselines.baseline_agent import BaselineAgent
from src.environment.env_utils import \
    count_n_in_row, is_terminal, get_legal_actions, drop_piece


class NStepLookaheadAgent(BaselineAgent):
    """
    NStepLookaheadAgent: BaselineAgent that uses a minimax search to choose
    the action that will guarantee the best score in the next 'N' steps.

    - IDEA FROM: https://www.kaggle.com/code/alexisbcook/n-step-lookahead
    - MINIMAX SEARCH: minimizing the possible loss for a worst case scenario

    The minimax search is carried out without interacting with the environment.
    Connect4 is a perfect information game and the model of the environment is
    known, so the agent can simulate 'N' steps ahead.

    How to break ties between the best scored actions?
        - Option 1) choose one of them at random (stochastic)
        - Option 2) choose the most central one (deterministic)
    This behaviour is chosen by the 'prefer_central_columns' attribute.
    """

    # Define the patterns (and their scores) to look for when evaluating boards
    _pattern_scores = {
        4: 1e10,   # four of your tokens in a row
        3: 1e4,    # three of your tokens in a row
        2: 1e2,    # two of your tokens in a row
        -2: -1,    # two of the opponent's tokens in a row
        -3: -1e6,  # three of the opponent's tokens in a row
        -4: -1e8   # four of the opponent's tokens in a row
    }
    # values from: https://www.kaggle.com/code/alexisbcook/n-step-lookahead

    def __init__(
            self,
            n: int,
            name="{n}-Step Lookahead Agent",
            prefer_central_columns: bool = False,
            **kwargs) -> None:
        """
        Initialize a NStepLookaheadAgent instance

        :param n: depth of the minimax search
        :param name: name of the Agent
        :param prefer_central_columns: whether the Agent prefers central column
        :param kwargs: 'exploration_rate' or 'allow_illegal_actions'
        """

        super().__init__(name.format(n=n), **kwargs)
        self.n = n
        self.prefer_central_columns = prefer_central_columns

    def _compute_scores(self, board: np.ndarray) -> np.array:
        """
        Given an observation, go 'N' steps down the game tree, score the
        leaf nodes, and then go up applying the minimax search. Return
        the scores of the available actions in the current turn.

        :param board: game board
        :return: scores of the available actions in the current turn
        """

        scores = np.full(board.shape[1], -np.Inf)
        legal_actions = get_legal_actions(board)
        # illegal actions will receive a -inf score and won't be chosen
        for action in legal_actions:
            next_board = drop_piece(board=board, column=action, mark=1)
            scores[action] = self._minmax_search(
                board=next_board, depth=self.n-1, is_max_player=False
            )
        return scores

    def _score_leaf_board(self, board: np.array) -> float:
        """
        Computes and returns the score of the given "leaf" board.
        The higher the score, the more valuable the board is.
        Count how many times each pattern appears in the board, and use this
        counter to weight their pattern_score.

        :param board: game board
        :return: score assigned to the given observation
        """

        # the score is a weighted sum (counts) of the _pattern_scores
        score = 0
        for pattern, pattern_score in self._pattern_scores.items():
            mark = 1 if pattern > 0 else -1
            counts = count_n_in_row(board=board, n=abs(pattern), mark=mark)
            score += counts * pattern_score
        return score

    def _minmax_search(self,
                       board: np.ndarray,
                       depth: int,
                       is_max_player: bool) -> float:
        """
        Runs a recursive min-max search. It goes 'N-1' steps down the game tree,
        starting at 'board'. It scores the leaf nodes, and then goes up applying
        the minimax search to minimize the possible loss for a worst case scenario

        :param board: game board
        :param depth: depth to grow the search tree
        :param is_max_player: whether player aims to minimize of maximize score
        :return: score of the given observation (board, mark)
        """

        # base case:
        if depth == 0 or is_terminal(board=board):
            return self._score_leaf_board(board=board)

        # recursive steps:
        legal_actions = get_legal_actions(board=board)
        if is_max_player:
            value = - np.Inf
            for action in legal_actions:
                child_board = drop_piece(board=board, column=action, mark=1)
                value = max(value, self._minmax_search(
                    board=child_board, depth=depth-1, is_max_player=False
                ))
            return value
        else:  # min-player
            value = np.Inf
            for action in legal_actions:
                child_board = drop_piece(board=board, column=action, mark=-1)
                value = min(value, self._minmax_search(
                    board=child_board, depth=depth-1, is_max_player=True
                ))
            return value

    def get_exploitation_policy(self, obs: np.ndarray) -> Categorical:
        """
        Returns the policy that must be followed to exploit the environment
        from the given observation. If 'prefer_central_columns', select the most
        central column among the best scored columns, else: choose one of the
        best scored columns at random.

        :param obs: environment observation (game board)
        :return: distribution over the action space to exploit the environment
        """

        scores = self._compute_scores(board=obs)
        best_scored_actions = np.where(scores == np.amax(scores))[0]
        probs = torch.zeros(obs.shape[1])
        if self.prefer_central_columns:
            best_central_action = best_scored_actions[
                np.argmin(abs(best_scored_actions - len(probs)//2))
            ]
            probs[best_central_action] = 1
        else:  # random sample among the best_scored_actions
            probs[best_scored_actions] = 1 / len(best_scored_actions)
        policy = Categorical(probs=probs)
        return policy

    def get_policy_scores_to_visualize(self, obs: np.ndarray) -> Tuple:
        """
        Returns the policy scores that will be shown in the User Interface
        (refer to the 'src/game' directory). In the case of the NStepLookahead
        Agent, the policy scores are the scores computed in the minimax search.
        The current board score is subtracted in order to provide with the real
        impact of each action.

        :return: scores from the minimax search
        """

        obs_score = self._score_leaf_board(board=obs)
        action_scores = self._compute_scores(board=obs)
        scores = action_scores - obs_score
        return tuple(scores.tolist())


if __name__ == "__main__":
    from src.environment.connect_game_env import ConnectGameEnv

    agent = NStepLookaheadAgent(n=1, prefer_central_columns=False)
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
