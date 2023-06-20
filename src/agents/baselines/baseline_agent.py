from src.agents.agent import Agent


class BaselineAgent(Agent):
    """
    Baseline Agent. Parent Class. Inherits methods and attributes from the
    Agent class defined in 'src/agents/agent.py'.

    Implements the functionalities that all the Baseline Agents in this project
    share. A Baseline Agent implements a heuristic to define the policy to
    follow. The heuristic does not have any trainable parameter. The heuristic
    may be deterministic or may implement a random component.

    BaselineAgents:
        - RandomAgent  (src/agents/baselines/random_agent.py)
        - LeftmostAgent  (src/agents/baselines/leftmost_agent.py)
        - NStepLookAheadAgent  (src/agents/baselines/n_step_look_ahead_agent.py)
    """

    def __init__(self, name: str = "BaselineAgent", **kwargs) -> None:
        super(BaselineAgent, self).__init__(name=name, **kwargs)


if __name__ == "__main__":
    agent = BaselineAgent()
    print(agent.name)
    print("ok")
