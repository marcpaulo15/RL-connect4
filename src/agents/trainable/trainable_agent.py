from src.agents.agent import Agent
from src.models.custom_network import CustomNetwork


class TrainableAgent(Agent):
    """
    Trainable Agent. Parent Class.

    Implements the functionalities that all the Trainable Agents in this
    project share. The TrainableAgent class implements a trainable model (a
    neural network) to define the Agent's behaviour: the strategy to choose
    actions.
    The training phase of the model will be carried out in a dedicated python
    notebook, not within this class implementation (refer to 'src/train')
    """

    def __init__(self,
                 model: CustomNetwork,
                 name: str = 'Trainable Agent',
                 **kwargs) -> None:
        """
        Initialize a TrainableAgent instance

        :param model: pytorch neural network that defines the Agent's behaviour
        :param name: name of the Trainable Agent instance
        :param kwargs: 'exploration_rate' or 'allow_illegal_actions'
        """

        super(TrainableAgent, self).__init__(name=name, **kwargs)
        self.model = model

    def save_weights(self,
                     file_path: str,
                     training_hparams: dict = None) -> None:
        """
        Save the network weights in the given file_path. If the training hyper
        parameters are provided, they are stored as a json file, using the
        same file_path name but with json extension.

        :param file_path: path where the weights will be saved
        :param training_hparams: [optional] training hyper-parameters
        :return: None
        """

        self.model.save_weights(file_path=file_path,
                                training_hparams=training_hparams,
                                )

    def load_weights(self, file_path: str) -> None:
        """
        Load the network weights from the given file_path. Before loading the
        weights, the 'model' attribute must be defined with the same network
        architecture according to these weights.

        :param file_path: location where the weights will be loaded from.
        :return: None
        """

        self.model.load_weights(file_path=file_path)


if __name__ == "__main__":
    print('ok')
