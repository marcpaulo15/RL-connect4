import json
import copy

import pygame

from src.game.game_logic.connect_game_menu import ConnectGameMenu
from src.game.game_logic.connect_game_handler import ConnectGameHandler


if __name__ == "__main__":

    # 1) load the configuration files
    with open(ConnectGameMenu.general_config_path, 'r') as config_file:
        menu_config = json.load(config_file)
        handler_config = copy.deepcopy(menu_config)
    with open(ConnectGameMenu.menu_config_path, 'r') as menu_config_file:
        menu_config_update = json.load(menu_config_file)
    with open(ConnectGameHandler.handler_config_path, 'r') as handler_config_file:
        handler_config_update = json.load(handler_config_file)

    menu_config.update(menu_config_update)
    handler_config.update(handler_config_update)

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

    if not done and menu.chosen_opponent is not None:
        # 4) Create a game versus the chosen opponent
        game = ConnectGameHandler(config=handler_config,
                                  opp_agent=menu.chosen_opponent)
        while not done:
            done = game.process_events()
            game.run_logic()
            game.draw(screen=screen)
            clock.tick(handler_config['pygame_tick'])

    pygame.quit()
