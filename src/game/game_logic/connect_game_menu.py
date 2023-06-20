import pygame
import json

from src.game.game_logic.opponents_list import get_opponents_list


class ConnectGameMenu:
    """
    ConnectGameMenu. Allows the user to choose his/her opponent for the game.
    Once the opponent is chosen, a ConnectGameHandler instance refers to the
    'chosen_opponent' attribute to know the opponent.

    The user must use the mouse to right-click on the colored box with the name
    of the Agent that will be playing against.

    Define the candidates to be the opponent in the file:
        - 'src/game/game_logit/opponents_list.py'
    """

    general_config_path = "./src/game/game_config/general_config.json"
    menu_config_path = "./src/game/game_config/menu_config.json"
    _box_colours = (
        (207, 236, 207), (253, 222, 238), (204, 236, 239), (255, 250, 129),
        (240, 232, 205), (253, 202, 162), (179, 226, 221), (221, 212, 232)
    )

    def __init__(self, config: dict) -> None:
        """
        Initialize a ConnectGameMenu instance.

        :param config: configuration
        """

        self._opponents_list = get_opponents_list()
        self.chosen_opponent = None
        self._mouse_pos = None
        self._screen_colour = config['screen_colour']

        font = pygame.font.SysFont(name=config['font_name'],
                                   size=config['font_size'])

        font.bold = pygame.font.Font.bold
        self._choose_your_opponent_txt = font.render(
            "Choose your opponent:", True, config['font_colour']
        )
        font.bold = None
        self._choose_your_opponent_rect = \
            self._choose_your_opponent_txt.get_rect()
        self._choose_your_opponent_rect.centerx = config['screen_size'][0] // 2
        self._choose_your_opponent_rect.top = config['upper_margin']

        boxes_top = self._choose_your_opponent_rect.bottom
        boxes_vertical_space = (config['screen_size'][1]
                                - config['lower_margin']
                                - self._choose_your_opponent_rect.height // 2
                                - boxes_top)
        intra_box_space = boxes_vertical_space // len(self._opponents_list)
        self._box_rect_list = []
        self._box_surface_list = []

        for i, opp in enumerate(self._opponents_list):
            surface = font.render(
                " "*config['whitespace_padding_box_name'] +
                opp.name +
                " "*config['whitespace_padding_box_name'],
                True,
                config['font_colour'],
                self._box_colours[i % len(self._box_colours)]
            )
            self._box_surface_list.append(surface)
            rect = surface.get_rect()
            rect.center = (config['screen_size'][0] // 2,
                           boxes_top + (i+1) * intra_box_space)
            self._box_rect_list.append(rect)

    def process_events(self) -> bool:
        """
        Capture mouse clicks. Returns whether the menu window is closed.

        :return: True if the menu window is closed
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._mouse_pos = pygame.mouse.get_pos()
        return False

    def run_logic(self) -> None:
        """
        Process the mouse click (if any). If any Agent (box) is selected, it
        updates the 'chosen_opponent' attribute

        :return: None
        """

        if self._mouse_pos is None:
            return
        chosen_rect_idx = None
        for idx, rect in enumerate(self._box_rect_list):
            if rect.collidepoint(self._mouse_pos):
                chosen_rect_idx = idx
                break
        if chosen_rect_idx is not None:
            self.chosen_opponent = self._opponents_list[chosen_rect_idx]
        self._mouse_pos = None

    def draw(self, screen: pygame.Surface) -> None:
        """
        Display everything on the given screen

        :param screen: pygame surface where the menu is displayed
        :return: None
        """

        screen.fill(self._screen_colour)  # background
        screen.blit(self._choose_your_opponent_txt,
                    self._choose_your_opponent_rect)
        for surface, rect in zip(self._box_surface_list, self._box_rect_list):
            screen.blit(surface, rect)
        pygame.display.update()


if __name__ == "__main__":
    # DEMO
    import os
    # YOUR PATH
    os.chdir('/home/marc/Escritorio/RL-connect4')

    # 1) load the configuration file
    with open(ConnectGameMenu.general_config_path, 'r') as config_file:
        menu_config = json.load(config_file)
    with open(ConnectGameMenu.menu_config_path, 'r') as menu_config_file:
        menu_config_update = json.load(menu_config_file)

    menu_config.update(menu_config_update)

    # 2) Initialize pygame and create the window
    pygame.init()
    screen = pygame.display.set_mode(menu_config['screen_size'])
    pygame.display.set_caption(menu_config['game_caption'])
    clock = pygame.time.Clock()

    # 3) Create a Menu instance to choose the opponent
    menu = ConnectGameMenu(config=menu_config)
    done = False
    while not done and menu.chosen_opponent is None:
        done = menu.process_events()
        menu.run_logic()
        menu.draw(screen=screen)
        clock.tick(menu_config['pygame_tick'])

    print("chosen opponent is", menu.chosen_opponent)

    pygame.quit()
