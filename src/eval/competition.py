from tqdm import trange
from typing import Tuple, List

import torch

from src.agents.agent import Agent
from src.environment.connect_game_env import ConnectGameEnv
from src.eval.run_episode import run_episode


def _get_initial_actions(game_id: int, ncols: int = 7) -> List[int]:
    """
    Compute which actions will be played to initialize the board before the
    game starts. The game_id (value from 0 to 99) determines the initial actions.
    This function is a mapping: game_id -> initial_actions to make sure that
    all the combinations of the first zero and two turns are used to initialize
    the games.

    :param game_id: game id from the competition main loop [0-99]
    :param ncols: number of board columns
    :return: sequence of initial actions to initialize the board
    """

    game_id_ = game_id % 50
    if game_id_ == 49:
        return []  # the empty board
    return [game_id_ % ncols, game_id_ // ncols]  # [for player1, for player2]


@torch.no_grad()
def competition(
        env: ConnectGameEnv,
        agent1: Agent,
        agent2: Agent,
        progress_bar: bool = True) -> Tuple[dict, List]:
    """
    Runs a fair competition between agent1 and agent2 in the given environment.
    - fair competition: each agent plays first (player1) in half the games.
    Returns the competition results and the last board of each game.

    - competition system: 100 games. Each game starts with an initial
    board that has zero or two moves already played. The idea is to introduce
    some exploration (initial randomness) so the agents have to prove that they
    know how to play in different scenarios that they would not have chosen.
    WHY 100 games? Since there are 7 columns (possible actions at each turn),
    there are 7*7=49 possible board configuration after the first two turns
    (one per player). If we also take into account the empty board, there are
    50 different board configurations. For each player playing first, one of
    these initializations is used => 50+50=100 different combinations.

    :param env: environment where the episodes take place
    :param agent1: agent1 that takes part in the competition
    :param agent2: agent2 that takes part in the competition
    :param progress_bar: whether to display a tqdm progress bar
    :return: information about the results of the competition
    """

    if agent1.allow_illegal_actions or agent2.allow_illegal_actions:
        raise Exception('illegal actions are not allowed in this competition')

    n_episodes = 100

    # Define the keys that will be tracked regardless of who plays first
    general_info = {
        'avg_cum_reward1': 0, 'avg_cum_reward2': 0, 'n_wins1': 0,  'n_wins2': 0
    }
    # Define the keys tracked separately depending on who plays first
    starting_player_dependent_vars = (
        'win_rate1', 'win_rate2', 'draw_rate', 'avg_game_len'
    )

    starting_player_vars = []
    for key_ in starting_player_dependent_vars:
        general_info[key_] = 0
        general_info[key_ + '_s1'] = 0
        general_info[key_ + '_s2'] = 0
        starting_player_vars.append(key_ + '_s1')
        starting_player_vars.append(key_ + '_s2')

    last_obs_list = []

    if progress_bar:  # display tdqm progress bar
        iter_ = trange(n_episodes, desc=f'{agent1.name} vs {agent2.name}')
    else:
        iter_ = range(n_episodes)

    for i in iter_:
        # for a fair competition, agent1 plays first half of the times
        initial_actions = _get_initial_actions(game_id=i)
        if i < n_episodes // 2:
            info, obs_list = run_episode(env=env, agent1=agent1, agent2=agent2,
                                         initial_actions=initial_actions)
            general_info['draw_rate_s1'] += int(info['is_a_draw'])
            general_info['win_rate1_s1'] += int(info['winner'] == 1)
            general_info['win_rate2_s1'] += int(info['winner'] == 2)
            general_info['avg_game_len_s1'] += info['game_len']
            general_info['avg_cum_reward1'] += sum(info['rewards1'])
            general_info['avg_cum_reward2'] += sum(info['rewards2'])

            last_obs_list.append(-obs_list[-1])  # from winner's perspective

        else:  # for a fair competition, agent2 plays first half of the times
            # NOTE: agent1 and agent2 exchange roles (player1 and player2)
            info, obs_list = run_episode(env=env, agent1=agent2, agent2=agent1,
                                         initial_actions=initial_actions)
            general_info['draw_rate_s2'] += int(info['is_a_draw'])
            general_info['win_rate1_s2'] += int(info['winner'] == 2)
            general_info['win_rate2_s2'] += int(info['winner'] == 1)
            general_info['avg_game_len_s2'] += info['game_len']
            general_info['avg_cum_reward1'] += sum(info['rewards2'])
            general_info['avg_cum_reward2'] += sum(info['rewards1'])

            last_obs_list.append(-obs_list[-1])  # from winner's perspective

    # gather partial information to compute the general results
    for k in starting_player_dependent_vars:
        general_info[k] = general_info[k+'_s1'] + general_info[k+'_s2']
    general_info['n_wins1'] = general_info['win_rate1']
    general_info['n_wins2'] = general_info['win_rate2']

    # finish the win_rates computation (add draw games)
    for s in (1, 2):  # starting_player
        general_info[f'win_rate1_s{s}'] += 0.5*general_info[f'draw_rate_s{s}']
        general_info[f'win_rate2_s{s}'] += 0.5*general_info[f'draw_rate_s{s}']

    general_info['avg_game_len_s1'] /= n_episodes//2
    general_info['avg_game_len_s2'] /= n_episodes//2
    general_info['win_rate1_s1'] /= n_episodes//2
    general_info['win_rate1_s2'] /= n_episodes//2
    general_info['win_rate2_s1'] /= n_episodes//2
    general_info['win_rate2_s2'] /= n_episodes//2
    general_info['draw_rate_s1'] /= n_episodes//2
    general_info['draw_rate_s2'] /= n_episodes//2

    general_info['avg_game_len'] /= n_episodes
    general_info['avg_cum_reward1'] /= n_episodes
    general_info['avg_cum_reward2'] /= n_episodes
    general_info['draw_rate'] /= n_episodes
    general_info['win_rate1'] /= n_episodes
    general_info['win_rate2'] /= n_episodes

    for k in general_info.keys():
        general_info[k] = round(general_info[k], 5)

    general_info['n_episodes'] = n_episodes
    general_info['agent1_name'] = agent1.name
    general_info['agent2_name'] = agent2.name

    return general_info, last_obs_list


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
    agent1 = LeftmostAgent()
    agent2 = RandomAgent()

    # 4) run a competition and display the results
    res, last_obs_list = competition(
        env=env,
        agent1=agent1,
        agent2=agent2,
        progress_bar=True,
    )

    pprint(res)
    # pprint(last_obs_list)
    print("OK!")
