from typing import Tuple, List

import torch

from src.agents.agent import Agent
from src.environment.connect_game_env import ConnectGameEnv


def run_episode(env: ConnectGameEnv,
                agent1: Agent,
                agent2: Agent,
                exploration_rate: float = None,
                print_transitions: bool = False,
                initial_actions: List[int] = ()) -> Tuple[dict, List]:
    """
    Runs an entire episode: agent1 versus agent2 in the given environment.
    Returns the episode information and the list of observations (game boards)
    If 'initial_actions' is provided, run those actions before starting the
    episode. The reason to do this is to prevent two deterministic agents from
    playing the exact same episode every time. With a non-empty initial board,
    deterministic agents are forced to play new moves rather than always
    play the same game starting with an empty board. Assume that player1 and
    player2 alternate turns to play the given 'initial_actions' (stating with
    player1).

    :param env: environment where the episode (game) takes place
    :param agent1: represents player1 in the game (first player)
    :param agent2: represents player2 in the game (second player)
    :param exploration_rate: fixed exploration rate for both players
    :param print_transitions: whether to display the episode transitions
    :param initial_actions: initial actions to create the initial board
    :return: information at the end of the episode and list of game boards
    """

    obs, info = env.reset()
    done = False
    # run the initial actions (if any)
    for init_action in initial_actions:
        obs, _, done, _ = env.step(action=init_action)
        if done:
            raise ValueError("initial_actions lead to a terminal board")

    obs_list = [obs]
    if print_transitions:
        print('obs:', obs)

    active_player = agent1 if len(initial_actions) % 2 == 0 else agent2

    while not done:
        with torch.no_grad():
            action = active_player.choose_action(
                obs=obs, exploration_rate=exploration_rate)
        obs, reward, done, info = env.step(action=action)
        obs_list.append(obs)
        active_player = agent2 if active_player == agent1 else agent1
        if print_transitions:
            print(f'action: {action}, reward: {reward}\n'+'-'*30+f'\nobs: {obs}')

    return info, obs_list


if __name__ == "__main__":
    # DEMO
    from pprint import pprint

    # 1) import agents and environment
    from src.agents.baselines.random_agent import RandomAgent
    from src.agents.baselines.leftmost_agent import LeftmostAgent
    from src.environment.connect_game_env import ConnectGameEnv

    # 2) initialize environment
    env = ConnectGameEnv()

    # 3) initialize agent1 and agent2
    agent1 = RandomAgent()
    agent2 = LeftmostAgent()

    # 4) run an episode and display the results
    res, obs_list = run_episode(
        env=env,
        agent1=agent1,
        agent2=agent2,
        print_transitions=True,
        initial_actions=[0, 2]
    )
    print()
    pprint(res)
