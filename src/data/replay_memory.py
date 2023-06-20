import copy
from collections import deque, namedtuple
from typing import List
import random

from src.agents.agent import Agent
from src.environment.connect_game_env import ConnectGameEnv
from src.environment.env_utils import is_illegal_action


class ReplayMemory:
    """
    ReplayMemory. Implements the basic functionalities of an Experience Replay
    Memory. It is used to train Neural Networks in Reinforcement Learning tasks.
    When sampling from the memory, the correlation between different samples
    within the same episode is broken (provided that there are enough samples).
    On the other hand, it can also serve as a buffer to store a set of the
    transitions from one or more episode and retrieve them already processed.

    The ReplayMemory receives an entire episode to process and store. This way,
    it can back-propagate the last rewards to the intermediate transitions
    before storing them in the main memory.

    DEALING WITH THE CREDIT ASSIGNMENT PROBLEM:
        - 'reward_backprop_exponent': each intermediate step has some impact
            on the game outcome. The actions played by the winner are arguably
            better than the ones played by the loser. And the latest moves of
            the game have more impact on the game outcome. For these reasons,
            the last reward (+1 or -1) can be back-propagated to the turns
            played by the same player (the winner or the loser).
            This parameter controls this "reward back-propagation".
            if None: the last reward is not back-propagated
            if =0: constant backprop (all turns receive the same credit)
            if =1: linear backprop.
            And so on.

    FIFO structure (deque): First In, First Out.
    """

    Transition = namedtuple(
        'Transition',
        ('state', 'action', 'reward', 'next_state', 'done', 'log_prob')
    )

    def __init__(self,
                 capacity: int,
                 reward_backprop_exponent: float = None) -> None:
        """
        Initialize a ReplayMemory instance

        :param capacity: deque max capacity
        :param reward_backprop_exponent: exponent to back-propagate terminal
               rewards
        """

        self.capacity = capacity
        self.reward_backprop_exponent = reward_backprop_exponent

        self.memory = deque(maxlen=self.capacity)
        self.original_rewards = deque(maxlen=self.capacity)

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self) -> int:
        return len(self.memory)

    def all_data(self) -> List:
        return list(self.memory)

    def is_empty(self) -> bool:
        return len(self.memory) == 0

    def sample(self, batch_size: int) -> List:
        return random.sample(self.memory, batch_size)

    def reset(self) -> None:
        self.memory.clear()
        self.original_rewards.clear()

    def _reward_backprop(
            self, turn: int, n_turns: int, last_reward: float) -> float:
        """
        Computes the reward that is given at 'turn' when the game has 'n_turns'
        played with 'last_reward' assigned to the last terminal transition
        (the last transition is at n_turns).

        :param turn: an intermediate turn of the game.
        :param n_turns: game length (total number of turns)
        :param last_reward: reward given in the last transition of the game
        :return: back-propagated reward
        """

        assert 1 <= turn <= n_turns, f'{turn}, {n_turns}, {last_reward}'
        # First, check whether the given 'turn' is played by the winner
        # we know that the last turn ('n_turns') is always played by the winner
        if (n_turns - turn) % 2 == 1:
            turn_played_by_winner = last_reward < 0
        else:
            # the current 'turn' and the last one are played by the same player
            turn_played_by_winner = last_reward > 0
        # if 'turn' is played by the loser, it actually loses in the next turn
        turn_ = turn if turn_played_by_winner else turn+1
        # back-propagate the (last) reward
        reward = abs(last_reward) * (turn_/n_turns) ** self.reward_backprop_exponent
        if not turn_played_by_winner:
            reward = - reward
        return round(reward, 4)

    def _backprop_episode_rewards(self, episode_transitions: List[dict]
                                  ) -> List[dict]:
        """
        Propagate the last rewards of the (terminal) states backwards.
        The idea is to give some credit to all those actions that lead to a
        certain outcome (winning or losing a game).
        The 'reward_backprop_exponent' controls how the last reward decreases
        for the earlier steps of the game.

        NOTE: certain wins and certain loses are considered terminal actions

        :param episode_transitions: sequence of transitions that form one episode
        :return: List of episode_transitions with the updated rewards
        """

        if not episode_transitions[-1]['done']:
            raise Exception('the game (episode) is not over', episode_transitions)

        backprop_episode_transitions = copy.deepcopy(episode_transitions)

        if len(backprop_episode_transitions) == 1:
            return backprop_episode_transitions

        max_reward = abs(episode_transitions[-1]['reward'])

        if self.reward_backprop_exponent is None:  # do not back-propagate
            return backprop_episode_transitions

        init_turn_ = (backprop_episode_transitions[0]['state'] != 0).sum() + 1
        init_turn = init_turn_
        for turn in range(init_turn_, init_turn_+len(backprop_episode_transitions)):
            last_reward = backprop_episode_transitions[turn-init_turn_]['reward']
            if (abs(abs(last_reward) - max_reward)) < 1e-4:  # terminal state
                # back-propagate the reward until init_turn
                for i in range(init_turn, turn):
                    # if the reward is zero, back-propagate the last reward
                    if abs(backprop_episode_transitions[i-init_turn]['reward']) < 1e-4:
                        backprop_episode_transitions[i-init_turn]['reward'] = \
                            self._reward_backprop(
                                turn=i, n_turns=turn, last_reward=last_reward
                            )
                init_turn = turn+1
        return backprop_episode_transitions

    def push(self, episode_transitions: List[dict]) -> None:
        """
        Adds a new episode (sequence of transitions) to the main memory.
        Assume that all the transitions in episode_transitions belong
        to the same episode (game) and are ordered.
        The terminal rewards are back-propagated before saving the episode in
        the main memory.
        A transition is a dictionary containing info (s,a,r,s',done,log_prob)

        :param episode_transitions: sequence of transitions
        :return: None
        """

        if not episode_transitions[-1]['done']:
            raise Exception('the game (episode) is not over', episode_transitions)

        episode_transitions_ = copy.deepcopy(episode_transitions)

        if is_illegal_action(action=episode_transitions_[-1]['action'],
                             board=episode_transitions_[-1]['state']):
            # if illegal actions are allowed and a player is disqualified,
            # only learn from the last move (the illegal one)
            episode_transitions_ = [episode_transitions_[-1]]

        _original_rewards = [t['reward'] for t in episode_transitions_]
        self.original_rewards.extend(_original_rewards)

        # back-propagate the last rewards
        episode_transitions_ = self._backprop_episode_rewards(
            episode_transitions=episode_transitions_
        )
        # add the episode to the memory
        for tt in episode_transitions_:
            self.memory.append(self.Transition(**tt))

    def push_self_play_episode_transitions(self,
                                           agent: Agent,
                                           env: ConnectGameEnv,
                                           init_random_obs: bool,
                                           push_symmetric: bool = True,
                                           exploration_rate: float = None) -> None:
        """
        Generate a sequence of episode transitions by letting the 'agent' play
        against itself (self-play) in the given environment 'env'. If 'init_
        random_obs' the game starts with a random board. If 'push_symmetric',
        compute the symmetric episode and push the symmetric transitions as well.

        :param agent: Agent that plays against itself (self-play)
        :param env: environment where the episode takes place
        :param init_random_obs: whether the episode start with a random board
        :param push_symmetric: whether to add the symmetric transitions as well
        :param exploration_rate: probability of exploring the environment
        :return: None. It fills the memory.
        """

        episode_transitions_ = []

        obs, _ = env.reset(init_random_obs=init_random_obs)
        done = False
        while not done:
            transition_ = agent.get_transition(state=obs,
                                               exploration_rate=exploration_rate)
            next_obs, reward, done, info = env.step(transition_['action'])
            transition_['next_state'] = next_obs.copy()
            transition_['reward'] = reward
            transition_['done'] = done
            episode_transitions_.append(transition_)
            obs = next_obs

            if info['is_a_draw']:
                # the memory assumes that there is a winner and the loser
                # if there is a draw, play again
                done = False
                obs, info = env.reset()
                episode_transitions_.clear()

            elif info['disqualified'] != 0:
                # if illegal actions are allowed and a player is disqualified,
                # only learn from the last move (the illegal one)
                episode_transitions_ = [episode_transitions_[-1]]

        self.push(episode_transitions=episode_transitions_)

        if push_symmetric:
            sym_episode_transitions_ = []
            for tt in episode_transitions_:
                sym_episode_transitions_.append(
                    agent.get_symmetric_transition(transition=tt)
                )
            self.push(episode_transitions=sym_episode_transitions_)


def __plot_reward_values(memory_: ReplayMemory) -> None:
    """
    Visualization function to plot the back-propagated rewards of an entire
    episode. The given memory is expected to have just one completed episode.

    :param memory_: non-empty ReplayMemory instance
    :return: None
    """

    import matplotlib.pyplot as plt

    plt.figure()
    reward_vals1 = [memory_[i].reward for i in range(len(memory_)) if i%2==0]
    reward_vals2 = [memory_[i].reward for i in range(len(memory_)) if i%2==1]
    if len(memory_) % 2 == 1:
        label1, label2 = 'winner rewards', 'loser rewards'
    else:
        label1, label2 = 'loser', 'winner'
    plt.scatter(range(1, len(memory_)+1, 2), reward_vals1, label=label1, zorder=5, s=60)
    plt.scatter(range(2, len(memory_)+1, 2), reward_vals2, label=label2, zorder=5, s=60)
    title = 'Rewards backpropagation (one episode)'
    if memory_.reward_backprop_exponent is not None:
        title += f', exp={memory_.reward_backprop_exponent}'
    plt.title(title)

    plt.scatter(range(1, len(memory_)+1), memory_.original_rewards,
                label='original rewards', marker='x', color='lime',
                zorder=3, s=100, alpha=0.6)
    plt.legend()
    for turn in range(len(memory_)+1):
        plt.axvline(x=turn, color='whitesmoke', linestyle='--', zorder=1)
    max_reward = max([abs(t.reward) for t in memory_])
    plt.axhline(y=max_reward, color='silver', linestyle='--', zorder=1)
    plt.axhline(y=-max_reward, color='silver', linestyle='--', zorder=1)
    plt.xlabel('turn')
    plt.ylabel('reward')
    plt.show()


if __name__ == "__main__":
    # DEMO

    from src.agents.baselines.n_step_lookahead_agent import NStepLookaheadAgent

    env = ConnectGameEnv()
    agent = NStepLookaheadAgent(n=1)

    memory = ReplayMemory(capacity=100,
                          reward_backprop_exponent=3)

    memory.push_self_play_episode_transitions(
        agent=agent,
        env=env,
        init_random_obs=False,
        push_symmetric=False,
        exploration_rate=0
    )

    print('->', len(memory), 'transitions:')
    print(memory.all_data())
    print(memory.original_rewards)
    __plot_reward_values(memory_=memory)
