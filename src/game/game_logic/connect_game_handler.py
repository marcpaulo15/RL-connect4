import random
import json
import time
from typing import List

import pygame
import numpy as np

from src.agents.agent import Agent
from src.agents.baselines.n_step_lookahead_agent import NStepLookaheadAgent
from src.agents.trainable.dqn_agent import DQNAgent
from src.agents.trainable.dueling_dqn_agent import DuelingDQNAgent
from src.environment.connect_game_env import ConnectGameEnv


class ConnectGameHandler:
    """
    ConnectGameHandler. Implements the user interface to play a ConnectX game
    against one of the Agents defined (and trained) in this project.

    This class enables the user to interact with different types of Agents in
    order to test their performance (and try to beat them).
    When the Agent takes turn, the strategy it follows will be displayed above
    the game board (on top of each column). When the user takes turn, the
    strategy that the Agent would follow will be displayed above the game board.
    The user does not have to follow that strategy, it is only for evaluation
    purposes.
    The kind of 'strategy' depends on the type of Agent. In most cases, it is
    the policy. But, for instance, the DQN Agent uses the predicted Q-values.

    INSTRUCTIONS:
        - when it is your turn, use your mouse to right-click on the (legal)
            column where you want to drop one of your pieces.
        - when the agent is taking turn, wait until it drops its piece.
        - whenever you want to restart the game, click on the 'restart' button
        - press the left-arrow key in your keyboard to go one step backward in
            time and review past moves.
        - press the right-arrow key in your keyboard to undo the backward steps
            in time and go back to the last game board.

    Frontend python package: pygame
    """

    general_config_path = "./src/game/game_config/general_config.json"
    handler_config_path = "./src/game/game_config/handler_config.json"

    def __init__(self, config: dict, opp_agent: Agent) -> None:
        """
        Initialize a ConnectGameHandler instance

        :param config: configuration (dictionary)
        :param opp_agent: your opponent, an Agent instance
        """

        self.config = config
        self.opp_agent = opp_agent
        self._active_player = random.choice(['human', self.opp_agent])
        self._mouse_pos = None  # (x,y) coordinates of the last mouse click
        # store the game turns in order to move back and forth in time (turns)
        self._history = {'obs': [], 'actions': [], 'policy_vals': []}

        board_width, board_height = self.config['board_size']
        board_center_x, board_center_y = self.config['board_center']

        # Create the board structure (a set of pygame rectangles)
        column_width = int(board_width / self.config['ncols'])
        column_height = board_height
        board_left = board_center_x - board_width // 2
        columns_left_list = [board_left + i * column_width
                             for i in range(self.config['ncols'])]
        column_top = board_center_y - board_height // 2
        # a pygame rectangle is composed of a pygame surface and its position
        self._column_surfaces_list = []
        self._column_rects_list = []
        for column_left in columns_left_list:
            surface = pygame.Surface([column_width, column_height])
            surface.fill(self.config['board_colour'])
            self._column_surfaces_list.append(surface)
            rect = surface.get_rect(topleft=(column_left, column_top))
            self._column_rects_list.append(rect)

        # create a map of from board positions to coordinates
        #  :(row,col) -> (x-coord, y-coord)
        self._centers_grid = np.zeros(
            (self.config['nrows'], self.config['ncols'], 2))
        for j in range(self.config['ncols']):
            column_x = self._column_rects_list[j].centerx
            row_y = column_top + column_height // (2*self.config['nrows'])
            for i in range(self.config['nrows']):
                self._centers_grid[i, j, :] = (column_x, row_y)
                row_y += column_height // self.config['nrows']

        # create and initialize the environment
        self.env = ConnectGameEnv()
        obs, _ = self.env.reset()
        self._history['obs'].append(
            {'board': self.env.board, 'mark': self.env.active_mark})
        self._history['policy_vals'].append(
            self.opp_agent.get_policy_scores_to_visualize(obs=obs)
        )
        self._history['actions'].append(None)
        self._turn_to_display = 0  # which turn is shown
        self._game_len = 1  # number of turns played

        self._winner = 0  # 1 if agent1 wins, 2 if agent2 wins
        self._is_tied = False  # whether the game is a draw

        # create the 'restart' button (text and rectangle) to reset everything
        font = pygame.font.SysFont(name=self.config['font_name'],
                                   size=self.config['font_size'])
        self._restart_button = font.render(
            'restart', True, config['restart_button_text_color'],
            config['new_game_button_color']
        )
        self._restart_button_rect = self._restart_button.get_rect(
            x=(self.config['screen_size'][0]
               - self._restart_button.get_width() - self.config['margin']),
            y=self.config['margin']
        )

    def is_game_over(self) -> bool:
        """
        Checks whether the game is over (there is a winner or a tie)

        :return: True if the game is over, otherwise returns False
        """
        return self._winner != 0 or self._is_tied

    def process_events(self) -> bool:
        """
        Capture any mouse clicks or keyboard pulses (left and right arrows).
        Returns whether the game window is closed.

        :return: True if the game is closed, False if the game is still running
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
                elif event.key == pygame.K_LEFT:  # move backward in time
                    self._turn_to_display = max(0, self._turn_to_display-1)
                elif event.key == pygame.K_RIGHT:  # move forward in time
                    self._turn_to_display = min(self._game_len-1,
                                                self._turn_to_display+1)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._mouse_pos = pygame.mouse.get_pos()  # mouse click
        return False

    def _is_legal_action(self, action: int) -> bool:
        """
        Checks whether the given action is legal if played in the current board
        An action is legal if the column where it is played is not full

        :param action: column
        :return: True if the action is legal
        """

        if action is None:
            return False
        last_obs = self._history['obs'][-1]
        return last_obs['board'][0, action] == 0

    def _process_turn(self, action: int) -> None:
        """
        Process the current turn and runs the environment 'step' method with
        the given 'action' in order to move on to the next turn.
        NOTE: This method assumes that the game is not over and that the
        step_to_visualize attribute is the last played turn (i.e. the displayed
        board is the last one, the present game state).

        :param action: action taken by the active player
        :return: None
        """

        obs, _, done_, info = self.env.step(action=action)
        self._history['obs'].append(
            {'board': self.env.board, 'mark': self.env.active_mark})
        if done_:
            self._history['policy_vals'].append([0]*self.config['ncols'])
        else:
            self._history['policy_vals'].append(
                self.opp_agent.get_policy_scores_to_visualize(obs=obs)
            )
        self._turn_to_display += 1
        self._game_len += 1
        self._history['actions'].append(action)
        self._winner = info['winner']
        self._is_tied = info['is_a_draw']

        if self._active_player == 'human':
            self._active_player = self.opp_agent
        else:
            self._active_player = 'human'
            # wait a second to create a smoother transition from Agent to human
            time.sleep(self.config['opp_turn_sleep'])

    def _restart_game(self) -> None:
        """
        Restart the attributes to their initial value

        :return: None
        """

        self._mouse_pos = None
        self._active_player = random.choice(['human', self.opp_agent])
        self._history = {'obs': [], 'actions': [], 'policy_vals': []}
        obs, _ = self.env.reset()
        self._history['obs'].append(
            {'board': self.env.board, 'mark': self.env.active_mark})
        self._history['policy_vals'].append(
            self.opp_agent.get_policy_scores_to_visualize(obs=obs)
        )
        self._history['actions'].append(None)
        self._turn_to_display = 0
        self._game_len = 1
        self._winner = 0
        self._is_a_draw = False

    def run_logic(self) -> None:
        """
        Runs the logic of the game. If the opponent is the active player, it
        takes turn. Otherwise, wait until the user (human) clicks on a legal
        action (column) to play. If the user clicks 'restart' at any moment,
        the game starts over.

        :return: None
        """

        action = None
        obs = self._history['obs'][-1]  # last observation, current board
        if self._active_player == self.opp_agent and not self.is_game_over():
            obs_ = obs['board'].copy()
            opp = 1 if obs['mark'] == 2 else 2
            obs_[obs_ == opp] = -1
            obs_[obs_ == obs['mark']] = 1
            action = self.opp_agent.choose_action(obs=obs_)
        if self._mouse_pos is not None:  # process mouse click from user
            if self._restart_button_rect.collidepoint(self._mouse_pos):
                self._restart_game()  # restart the game (board)
                return
            elif self._turn_to_display == self._game_len - 1 and \
                    self._active_player == 'human' and not self.is_game_over():
                for col_id, col_rect in enumerate(self._column_rects_list):
                    if col_rect.collidepoint(self._mouse_pos):
                        action = col_id
            self._mouse_pos = None

        # if 'action' is None or illegal, do nothing
        if self._is_legal_action(action=action):
            self._process_turn(action=action)

    def _display_policy_vals(self, screen: pygame.Surface) -> None:
        """
        Display the policy that the 'opponent' uses in his turns and the one
        that would use in the turns it is not playing (i.e. played by the user)

        :param screen: pygame surface where the policy values will be displayed
        :return: None.
        """

        # Define the format depending on the type of Agent class
        if isinstance(self.opp_agent, NStepLookaheadAgent):
            var, transform = 'Scores:', lambda v: '{:.2g}'.format(v)
        elif (isinstance(self.opp_agent, DQNAgent) or
                isinstance(self.opp_agent, DuelingDQNAgent)):
            var, transform = 'Qvals:', lambda v: str(round(v, 2))
        else:  # PGAgents and other BaselineAgents
            var, transform = 'Policy:', lambda v: str(round(v, 2))
        # retrieve the policy values for the turn_to_display
        policy_vals_ = self._history['policy_vals'][self._turn_to_display]
        policy_vals = [transform(v) for v in policy_vals_]  # apply format

        font = pygame.font.SysFont(name=self.config['font_name'],
                                   size=self.config['policy_info_font_size'])
        column_centers_x = [x for x, y in self._centers_grid[0]]
        y = self._column_rects_list[0].top   # y coord is shared
        for x, val in zip(column_centers_x, policy_vals):
            val_text = font.render(val, True, self.config['font_colour'])
            val_text_topleft = (x-val_text.get_width()//2,
                                y-val_text.get_height()-self.config['margin'])
            screen.blit(val_text, val_text_topleft)

        board_left = self._column_rects_list[0].left
        mark = self._history['obs'][self._turn_to_display]['mark']
        var_text_colour = self.config['colour1'] if mark == 1\
            else self.config['colour2']
        var_text = font.render(var, True, var_text_colour)
        var_text_topleft = (board_left - var_text.get_width(),
                            y - var_text.get_height() - self.config['margin'])
        screen.blit(var_text, var_text_topleft)

    def draw_text(self, screen: pygame.Surface) -> None:
        """
        Display all the game elements in text format on the screen.

        :param screen: pygame surface where the text elements will be displayed
        :return:
        """

        colour1, colour2 = self.config['colour1'], self.config['colour2']
        font = pygame.font.SysFont(name=self.config['font_name'],
                                   size=self.config['font_size'])
        info_text_tl = [self._column_rects_list[0].left, self.config['margin']]
        if self._is_tied:
            colour_ = self.config['font_colour']
            info_text = self.config['tie_text']
        else:  # winner or ongoing game
            info_text_ = font.render('Player  ', True, (0, 0, 0))
            screen.blit(info_text_, info_text_tl)
            info_text_tl[0] += info_text_.get_width()
            if self._winner != 0:
                colour_ = colour1 if self._winner == 1 else colour2
                info_text = ' , you win!!!'
            else:  # current turn information
                current_player = self._history['obs'][-1]['mark']
                colour_ = colour1 if current_player == 1 else colour2
                info_text = " , it's your turn!"
            center = (info_text_tl[0]+10, info_text_tl[1]+info_text_.get_height()//2)
            pygame.draw.circle(
                screen, colour_, center, 15)
            colour_ = (0, 0, 0)
            info_text_tl[0] += 20

        n_turns_behind = self._game_len - self._turn_to_display - 1
        if n_turns_behind > 0:
            info_text += self.config['n_turns_behind_text'].format(n=n_turns_behind)
        info_text = font.render(info_text, True, colour_)
        screen.blit(info_text, info_text_tl)

        font2 = pygame.font.SysFont(name=self.config['font_name'],
                                    size=self.config['policy_info_font_size'])

        opp_name_text = font2.render(f'vs {self.opp_agent.name}',
                                     True,
                                     self.config['policy_info_font_colour']
                                     )
        opp_name_text_topleft = (
            self.config['margin'],
            self.config['screen_size'][1] - opp_name_text.get_height()
            - self.config['margin']
        )

        # display the opponent's name
        screen.blit(opp_name_text, opp_name_text_topleft)

        self._display_policy_vals(screen=screen)

    def draw(self, screen: pygame.Surface) -> None:
        """
        Display the game content on the screen
        NOTE: the order in which the elements are drawn matters

        :param screen: pygame surface where the game is displayed
        :return: None. Fills the given screen.
        """

        screen.fill(self.config['screen_colour'])  # fill the background
        # draw the board structure (columns)
        for column_surface, column_rect in zip(self._column_surfaces_list,
                                               self._column_rects_list):
            screen.blit(column_surface, column_rect)

        # draw the pieces on the board
        obs_to_display = self._history['obs'][self._turn_to_display]
        for i in range(self.config['nrows']):
            for j in range(self.config['ncols']):
                colour = self.config['empty_cell_colour']
                if obs_to_display['board'][i, j] == 1:
                    colour = self.config['colour1']
                elif obs_to_display['board'][i, j] == 2:
                    colour = self.config['colour2']
                center = self._centers_grid[i, j, :]
                # token (inner circle)
                pygame.draw.circle(
                    screen, colour, center, self.config['circle_radius'])
                # outer circumference (contour)
                pygame.draw.circle(
                    screen, (0, 0, 0), center, self.config['circle_radius'], 4)

        # add the 'restart' button to start over the game
        screen.blit(self._restart_button, self._restart_button_rect)

        # highlight the last token that has been dropped
        if self._turn_to_display != 0:
            action = self._history['actions'][self._turn_to_display]
            action_row = \
                np.where(obs_to_display['board'][:, action] != 0)[0][0]
            action_center = self._centers_grid[action_row, action]
            pygame.draw.circle(
                screen, self.config['highlight_position_colour'], action_center,
                self.config['circle_radius'], width=4
            )
        # if there is a winner, draw a line highlighting the winning combination
        if self._winner != 0 and self._turn_to_display == self._game_len-1:
            winning_pos = self._get_winning_line(board=obs_to_display['board'],
                                                 inrow=self.config['inrow'])
            start_pos = self._centers_grid[winning_pos[0]]
            end_pos = self._centers_grid[winning_pos[-1]]
            pygame.draw.line(screen,
                             self.config['winning_line_colour'],
                             start_pos,
                             end_pos,
                             self.config['winning_line_width'])

        # add the text elements
        self.draw_text(screen=screen)
        # update the screen content
        pygame.display.update()

    @staticmethod
    def _get_winning_line(board: np.ndarray, inrow: int = 4) -> List:
        """
        Returns a list of the coordinates (row_idx, col_idx) of the positions
        that form the winning line. If there is no winner, return an empty list.

        :return: [(row_idx{1},col_idx{1}), ..., (row_idx{inrow},col_idx{inrow})
        """

        nrows, ncols = board.shape
        for mark in (1, 2):  # for both player1 and player2
            mark_board_mask = (board == mark).astype(int)
            # search for a winning combination by rows
            for i in range(nrows):
                for j in range(ncols - inrow + 1):
                    row_window = mark_board_mask[i, j:j+inrow]
                    if row_window.sum() == inrow:
                        return [(i, jj) for jj in range(j, j+inrow)]
            # search for a winning combination by columns
            for j in range(ncols):
                for i in range(nrows - inrow + 1):
                    col_window = mark_board_mask[i:i+inrow, j]
                    if col_window.sum() == inrow:
                        return [(ii, j) for ii in range(i, i+inrow)]
            # search for a winning combination by diagonals
            for i in range(nrows - inrow + 1):
                for j in range(ncols - inrow + 1):
                    square_window = mark_board_mask[i:i+inrow, j:j+inrow]
                    des_diag = square_window.diagonal()
                    asc_diag = np.fliplr(square_window).diagonal()
                    if des_diag.sum() == inrow:  # descending column
                        return [(i+k, j+k) for k in range(inrow)]
                    elif asc_diag.sum() == inrow:  # ascending column
                        return [(i+k, j+inrow-k-1) for k in range(inrow)]
        return []  # there is no winner yet


if __name__ == "__main__":
    # DEMO
    import os
    os.chdir('/home/marc/Escritorio/RL-connect4')

    opponent = NStepLookaheadAgent(n=2)

    with open(ConnectGameHandler.general_config_path, 'r') as config_file:
        handler_config = json.load(config_file)
    with open(ConnectGameHandler.handler_config_path, 'r') as menu_config_file:
        handler_config_update = json.load(menu_config_file)

    handler_config.update(handler_config_update)

    pygame.init()
    screen = pygame.display.set_mode(handler_config['screen_size'])
    pygame.display.set_caption(handler_config['game_caption'])
    clock = pygame.time.Clock()

    game = ConnectGameHandler(config=handler_config, opp_agent=opponent)

    done = False
    while not done:
        done = game.process_events()
        game.run_logic()
        game.draw(screen=screen)
        clock.tick(handler_config['pygame_tick'])

    pygame.quit()
