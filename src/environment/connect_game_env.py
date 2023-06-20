import copy
from typing import Tuple

import numpy as np

from src.environment.env_utils import is_game_winner, is_a_draw, \
    drop_piece, is_illegal_action, get_winning_cols, random_action,  \
    is_terminal


class ConnectGameEnv:
    """
    ConnectGameEnv. Implements ConnectX game as a Reinforcement Learning
    environment. Two players (agents) alternate turns to play the game. After
    each turn, the active player receives a reward indicating the quality of
    the chosen action. The environment keeps some information about the ongoing
    episode.

    - Observation: game board with values {1: active player, 0: empty,
        -1: opponent}. The active player chooses a colum to drop a '1' piece.
    - State: Internally, the board is represented differently : its values are
        {0: empty, 1: player1, 2: player2} and 'active_player' is the player
        1|2 taking the current turn .

    - ConnectX is a perfect-information two-player zero-sum game
    - This class follows the basic structure as an OpenAI gym environment
    - the size of the board and the number of pieces in row to win the game are
        parameters of the class (and can be changed)
    - the initial board can be random, not just an empty board ('reset' method)
    """

    # information about the ongoing episode
    _init_episode_info = {
            'init_obs': {},  # initial observation ('reset' method is called)
            'game_len': 0,   # game length
            'is_a_draw': False,  # whether the game is tied
            'winner': 0,  # 0: no winner (yet), 1: player1, 2: player2
            'actions1': [],  # actions played by player1 (first player)
            'actions2': [],  # actions played by player2 (second player)
            'disqualified': 0,  # 0: no disq, 1: player1, 2: player2
            'rewards1': [],  # player1 rewards
            'rewards2': [],  # player2 rewards
        }

    def __init__(self,
                 nrows: int = 6,
                 ncols: int = 7,
                 inrow: int = 4) -> None:
        """
        Initialize a ConnectGameEnv instance

        :param nrows: number of board rows
        :param ncols: number of board columns
        :param inrow: number of tokens in line to win the game
        """

        self.nrows = nrows
        self.ncols = ncols
        self.inrow = inrow
        self.board_shape = (self.nrows, self.ncols)
        self.max_game_len = self.nrows * self.ncols
        # Call the 'reset' method to initialize 'board' and 'active_mark'
        self.board = None
        self.active_mark = None
        # Information about the ongoing episode
        self._episode_info = copy.deepcopy(self._init_episode_info)

    def _get_obs(self) -> np.ndarray:
        """
        From the 'board' and 'active_mark' attributes, create a board with
        values: {1: active player, -1: opponent, 0: empty}.
        The active player chooses a colum to drop a '1' piece.

        :return: game board to play the next turn
        """

        obs_ = self.board.copy()
        opponent = 1 if self.active_mark == 2 else 2
        obs_[obs_ == opponent] = -1
        obs_[obs_ == self.active_mark] = 1
        return obs_

    def _get_info(self) -> dict:
        """
        Returns information about the status of the ongoing game (episode)

        :return: dictionary{str: Any}
        """

        return self._episode_info

    def reset(self, init_random_obs: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Starts a new episode. If 'init_random_obs', the initial board is
        randomly chosen. A Random board is a non-terminal board (it is not game
        over) so at least one more move can be played. If not 'init_random_obs'
        the initial board is the common empty board (zeros).
        NOTE: this method empties the '_episode_info' attribute

        :return: 2-tuple (initial observation, initial information)
        """

        if init_random_obs:
            random_obs_ = self._random_observation(
                terminal=False,
                nrows=self.nrows,
                ncols=self.ncols,
                inrow=self.inrow)
            self.board = random_obs_['board']
            self.active_mark = random_obs_['mark']
        else:  # empty board to start the game
            self.board = np.zeros(self.board_shape, dtype=int)
            self.active_mark = 1
        self._episode_info = copy.deepcopy(self._init_episode_info)
        self._episode_info['init_obs'] = self._get_obs()
        return self._get_obs(), self._get_info()

    @staticmethod
    def _get_non_terminal_state_reward(board: np.ndarray,
                                       mark: int,
                                       action: int) -> float:
        """
        Computes and returns the reward that the active player receives if it
        takes the given 'action' in the given 'obs'. The idea is to penalize
        bad actions and reward good ones.

        NOTE: this method assumes that 'action' has NOT been played in 'obs'
        yet, and that 'action' is not to win the game.

        - Scenarios that receive a negative reward:
        (1) if the active player wastes a clear opportunity to win -> -0.5
        (2) if the active player does not prevent the opp from winning -> -1
        if reward(1)+reward(2) != 0: return the sum of rewards. Else:
        - Scenarios that receive a positive reward:
        (3) if the active player has more than one option to win -> +1
        (4) if the active player blocks an opponent's 3-inrow -> +0.5
        return reward(3)+reward(4)

        :param board: game board {0: empty, 1: player1, 2: player2}
        :param mark: token of the active player
        :param action: chosen column
        """

        opp = 1 if mark == 2 else 2  # opponent

        action_reward = 0  # value to return
        # (1) if the active player wastes a clear opportunity to win, it
        # receives a negative reward.
        # NOTE: this method assumes that 'action' is not to win the game
        active_winning_columns = get_winning_cols(board=board, mark=mark)
        if len(active_winning_columns) != 0:
            action_reward = action_reward - 0.5

        next_board = drop_piece(board=board, column=action, mark=mark)
        # (2) if the opponent can win in his next turn, the active
        # player losses the game (it should have blocked it)
        opp_winning_columns = get_winning_cols(board=next_board, mark=opp)
        if len(opp_winning_columns) != 0:
            action_reward = action_reward - 1

        if action_reward != 0:
            return action_reward

        # (3) if the active player has more than one option to win, it means
        # it can force a victory in its next turn (it's like winning)
        next_winning_columns = get_winning_cols(board=next_board, mark=mark)
        if len(next_winning_columns) >= 2:
            action_reward = action_reward + 1

        # (4) if the player blocks a chance for the opponent to win, it
        # receives a positive reward.
        if_no_action_board = drop_piece(board=board, column=action, mark=opp)
        if is_game_winner(board=if_no_action_board, mark=opp):
            action_reward = action_reward + 0.5

        return action_reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Updates the state of the environment after the active player drops one
        of its pieces in the given column ('action')

        :param action: column where the active mark is dropped
        :return: 4-tuple (next observation, reward, done, info)
        """

        # action value: sanity check
        if (not str(action).isnumeric() or
                not (0 <= action < self.board_shape[-1])):
            raise ValueError(
                f"'action' must be an positive integer, not {action}")

        self._episode_info['game_len'] += 1
        self._episode_info[f'actions{self.active_mark}'].append(action)

        reward_ = None  # value updated in the following lines

        if is_illegal_action(action=action, board=self.board):
            self._episode_info['disqualified'] = self.active_mark
            next_board = self.board
            reward_ = - 5
        else:  # if 'action' is a legal action:
            next_board = drop_piece(
                board=self.board, column=action, mark=self.active_mark)
            # check if the game is over (winner or drawn)
            if is_game_winner(board=next_board, mark=self.active_mark):
                self._episode_info['winner'] = self.active_mark
                reward_ = 1
            elif is_a_draw(board=next_board):
                self._episode_info['is_a_draw'] = True
                reward_ = 0

        if reward_ is None:  # game is not over
            reward_ = self._get_non_terminal_state_reward(
                board=self.board, mark=self.active_mark, action=action)

        # update the cumulative reward of the active player
        self._episode_info[f'rewards{self.active_mark}'].append(reward_)

        # Update the next observation
        self.board = next_board
        self.active_mark = 2 if self.active_mark == 1 else 1

        # compute the values to return
        next_obs, info = self._get_obs(), self._get_info()
        done = (self._episode_info['winner'] != 0 or
                self._episode_info['is_a_draw'] or
                self._episode_info['disqualified'] != 0)

        return next_obs, reward_, done, info

    @staticmethod
    def _random_observation(terminal: bool = False,
                            nrows: int = 6,
                            ncols: int = 7,
                            inrow: int = 4) -> dict:
        """
        Generates a random game board. Starting from an empty board, play a
        sequence of random actions and return one of the game states at random
        (or the last one if 'terminal'). Returns {'board', 'mark'}
        - board values: {0: empty, 1: player1, 2: player2}
        - mark: 1|2

        :param terminal: whether the random observation is terminal
        :param nrows: number of rows in the board
        :param ncols: number of board columns
        :param inrow: number of tokens in row to win the game
        :return: random observation {'board': np.ndarray, 'mark': 1|2}
        """

        board, mark = np.zeros((nrows, ncols)), 1
        random_obs_list = [{'board': board, 'mark': mark}]
        done = False
        while not done:
            action = random_action(board=random_obs_list[-1]['board'])
            board = drop_piece(board=board, column=action, mark=mark)
            mark = 1 if mark == 2 else 2
            random_obs_list.append({'board': board, 'mark': mark})
            done = is_terminal(board=board, inrow=inrow)
        idx = -1 if terminal else np.random.randint(0, len(random_obs_list)-1)
        return random_obs_list[idx]

    @staticmethod
    def random_observation(terminal: bool = False,
                           nrows: int = 6,
                           ncols: int = 7,
                           inrow: int = 4) -> np.array:
        """
        Generates a random game board. Starting from an empty board, play a
        sequence of random actions and return one of the game states at random
        (or the last one if 'terminal').
        - board values: {0: empty, 1: active player, -1: opponent}

        :param terminal: whether the random board is done (game over)
        :param nrows: number of rows in the board
        :param ncols: number of board columns
        :param inrow: number of tokens in row to win the game
        :return: random game board {0: empty, 1: active_player, 2: opponent}
        """

        obs_dict = ConnectGameEnv._random_observation(
            terminal=terminal, nrows=nrows, ncols=ncols, inrow=inrow)
        obs = obs_dict['board'].copy()
        opponent = 1 if obs_dict['mark'] == 2 else 2
        obs[obs == opponent] = -1
        obs[obs == obs_dict['mark']] = 1
        return obs


if __name__ == "__main__":
    # DEMO
    from pprint import pprint

    env = ConnectGameEnv()
    obs, info = env.reset(init_random_obs=False)
    done = False
    while not done:
        action = random_action(board=obs)
        next_obs, reward, done, info = env.step(action=action)
        print(f'obs:\n{obs},\naction: {action}, reward: {reward}, '
              f'done: {done}, next_obs:\n{next_obs}\n' + '-'*60)
        obs = next_obs
    pprint(info)

    random_obs = ConnectGameEnv.random_observation(terminal=False)
    print('\nrandom_obs:\n', random_obs)
    terminal_random_obs = ConnectGameEnv.random_observation(terminal=True)
    print('\nterminal_random_obs:\n', terminal_random_obs)
