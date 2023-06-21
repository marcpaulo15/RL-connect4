import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class CustomNetwork(nn.Module):
    """
    Custom Network: Create a Neural Network that follows the structure:
    ConvBlock -> FullyConnBlock |--> FirstHead (policy / Q-vals / Advantages)
                                |--> SecondHead (state-value)

    How layers are initialized:
        - when initializing a new CustomNetwork instance, each module (block)
        is received as a python list whose elements are strings and numbers.
        These elements are translated into a Pytorch layer.
        - numbers (or tuples of numbers) represent the number of hidden units /
            kernel sizes / padding / ...
        - strings represent activation functions ('tanh' or 'relu')
        - in the class documentation there are more details about the different
        types of layer that are allowed for each module.

    Preprocessing observations before feeding the Network:
        - environment observations are two-dimensional Numpy arrays (boards)
        - the board is one-hot encoded so the first channel contains the active
        player's pieces and the second channel contains the one's from the
        opponent.
        - Apart from that, the positions that can be filled in the current turn
        are highlighted with a -1 value.
    """

    def __init__(self,
                 conv_block: List = (),
                 fc_block: List = (),
                 first_head: List = (),
                 second_head: List = (),
                 name: str = 'CustomNetwork({n_params})',
                 board_shape: Tuple[int, int] = (6, 7)
                 ):
        """
        Initialize a CustomNetwork instance

        :param conv_block: convolutional layers (backbone network)
        :param fc_block:  fully connected layers (backbone network)
        :param first_head: first prediction head
        :param second_head: second prediction head
        :param name: name of the network (architecture)
        :param board_shape: shape of the Connect4 game board
        """

        super(CustomNetwork, self).__init__()
        self.arch = {'conv_block': conv_block,
                     'fc_block': fc_block,
                     'first_head': first_head,
                     'second_head': second_head}
        self.board_shape = board_shape
        self.input_shape = (2, *self.board_shape)
        # 2 channels from the one-hot encoding of the players' pieces
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CONVOLUTIONAL BLOCK
        self.conv_block = nn.Sequential()
        in_channels_ = 2
        for layer in conv_block:
            in_channels_ = self._add_conv_layer(
                layer=layer,
                module=self.conv_block,
                in_channels=in_channels_
            )

        # compute the number of input features of the first FullyConn layer
        with torch.no_grad():
            conv_out = self.conv_block(
                torch.zeros((1, *self.input_shape))
            )
            in_features = torch.prod(torch.tensor(conv_out.shape)).item()

        # FULLY CONNECTED LAYER BLOCK
        self.fc_block = nn.Sequential()
        for layer in fc_block:
            in_features = self._add_fully_connected_layer(
                layer=layer, module=self.fc_block, in_features=in_features)

        # FIRST HEAD
        self.first_head = nn.Sequential()
        in_features_head1 = in_features
        for layer in list(first_head):
            in_features_head1 = self._add_fully_connected_layer(
                layer=layer,
                module=self.first_head,
                in_features=in_features_head1
            )

        # SECOND HEAD [optional]
        self.second_head = nn.Sequential()
        in_features_head2 = in_features
        for layer in list(second_head):
            in_features_head2 = self._add_fully_connected_layer(
                layer=layer,
                module=self.second_head,
                in_features=in_features_head2
            )

        # compute the total number of trainable parameters in the network
        self.conv_block_n_params = sum(
            p.numel() for p in self.conv_block.parameters() if p.requires_grad
        )
        self.fc_block_n_params = sum(
            p.numel() for p in self.fc_block.parameters() if p.requires_grad
        )
        self.first_head_n_params = sum(
            p.numel() for p in self.first_head.parameters() if p.requires_grad
        )
        self.second_head_n_params = sum(
            p.numel() for p in self.second_head.parameters() if p.requires_grad
        )
        self.n_params = (self.conv_block_n_params + self.fc_block_n_params +
                         self.first_head_n_params + self.second_head_n_params)
        # create the name of the network based on its structure

        if '{n_params}' in name:
            self.name = name.format(n_params=self.n_params)
        else:
            self.name = name

        # move th model to 'device'
        self.to(self.device)

    def print_blocks(self) -> None:
        """
        Prints the number of parameters for each module

        :return: None
        """

        print(f' -> {self.n_params} weights in total:')
        for b in ('conv_block', 'fc_block', 'first_head', 'second_head'):
            n_layers = len(getattr(self, b))
            n_params = getattr(self, b+'_n_params')
            print(f' -> {b}: {n_layers} layers, {n_params} params')

    @staticmethod
    def _add_conv_layer(layer, module, in_channels) -> int:
        """
        Adds a 2D (convolutional) layer to the given module.
        - [<in_channels>, <kernel_size>, <padding>]: convolutional layer
        - 'tanh': activation function
        - 'relu': activation function

        :param layer: layer
        :param module: module in which the layer is added
        :param in_channels: number of input channels
        :return: number of output channels at the output of the layer
        """

        next_in_channels = in_channels
        n_layers = len(module)
        if isinstance(layer, (list, tuple)) and len(layer) == 3:
            # 2D convolutional layer = (out_channels, kernel_size, padding)
            module.append(nn.Conv2d(
                in_channels=in_channels, out_channels=layer[0],
                kernel_size=layer[1], padding=layer[2]
            ))
            next_in_channels = layer[0]
        elif isinstance(layer, str):
            if layer.lower() == 'relu':
                module.append(nn.ReLU())
            elif layer.lower() == 'tanh':
                module.append(nn.Tanh())
            # check that one layer has been added to the given module
            if n_layers == len(module):
                raise ValueError(f"(conv_block) layer {layer} is not available")
        return next_in_channels

    @staticmethod
    def _add_fully_connected_layer(layer, module, in_features) -> int:
        """
        Adds a fully connected layer to the given module
        - <int>: number of hidden units
        - 'tanh': activation function
        - 'relu': activation function

        :param layer: fully connected layer type
        :param module: module in which the layer is added
        :param in_features: number of input features
        :return: number of output features at the output of the layer
        """

        n_layers = len(module)
        next_in_features = in_features
        if isinstance(layer, int):
            module.append(nn.Linear(
                in_features=in_features, out_features=layer))
            next_in_features = layer
        elif isinstance(layer, str):
            if layer.lower() == 'relu':
                module.append(nn.ReLU())
            elif layer.lower() == 'tanh':
                module.append(nn.Tanh())

        if n_layers == len(module):
            raise ValueError(f"FC layer {layer} is not available")
        return next_in_features

    def forward(self, x):
        """
        If the second head is empty, the prediction is just the first head.

        :param x:
        :return:
        """

        if len(self.conv_block) != 0:
            x = self.conv_block(x)
            x = torch.flatten(x, start_dim=1)
        x = self.fc_block(x)
        output1 = self.first_head(x)
        if len(self.second_head) != 0:
            output2 = self.second_head(x)
            return output1, output2
        else:
            return output1

    @classmethod
    def from_architecture(cls, file_path: str, n_heads: int = 2):
        """
        Creates a CustomNetwork instance following a pre-defined architecture
        that is stored in the given file_path as a json file.

        :param file_path: location of the file where the architecture is stored
        :param n_heads: number of prediction heads
        :return: a class instance, with the given architecture
        """

        with open(file_path, 'r') as file:
            architecture = json.load(file)
        network = cls(**architecture)
        if n_heads == 1:
            network.second_head = nn.Sequential()
        return network

    def obs_to_model_input(self, obs: np.array) -> torch.tensor:
        """
        Preprocess an observation from the environment to feed the Network:
        First, one-hot encode the board (players' pieces):
            - first channel: active player;    second channel: opponent
        (2) highlight the cells that can be filled in the next turn (value -1)

        :param obs: observation (game board)
        :return: obs turned into a model input
        """

        model_input_ = np.zeros((2, *obs.shape))
        # one-hot encoder
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if obs[i, j] == 1:
                    model_input_[0, i, j] = 1
                elif obs[i, j] == -1:
                    model_input_[1, i, j] = 1
        # highlight available cells that can be filled in the next turn
        filled_positions = model_input_[0] + model_input_[1]
        for channel_vals in model_input_:
            for col in range(channel_vals.shape[-1]):
                first_empty_row = np.where(filled_positions[:, col] == 0)[0]
                if len(first_empty_row) > 0:
                    channel_vals[first_empty_row[-1], col] = -1
        model_input_ = torch.from_numpy(model_input_).float().unsqueeze(0)
        return model_input_.to(self.device)

    def save_weights(
            self,
            file_path: str,
            training_hparams: dict = None,
    ) -> None:
        """
        Save the network weights in the given file_path. If the training hyper
        parameters are provided, they are stored as a json file using the same
        file_path name but with json extension.

        :param file_path: path where the weights will be saved
        :param training_hparams: [optional] training hyper-parameters
        :return: None
        """

        # save the weights in file_path
        torch.save(self.state_dict(), file_path)
        if training_hparams is not None:
            # from the file_path (weights), retrieve the hparams file path
            hparams_file_path = ''.join(file_path.split('.')[:-1]) \
                                + '_hparams.json'
            if file_path[0] == '.':
                hparams_file_path = '.' + hparams_file_path
            with open(hparams_file_path, 'w') as hparams_file:
                json.dump(training_hparams, hparams_file, indent=4)

    def load_weights(self, file_path: str) -> None:
        """
        Load the network weights from the given file_path. Before loading the
        weights, the 'model' attribute must be defined with the same network
        architecture according to these weights.

        :param file_path: location where the weights will be loaded from.
        :return: None
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(file_path, map_location=device))


if __name__ == "__main__":
    # DEMO
    import os
    # YOUR PATH HERE
    os.chdir('/home/marc/Escritorio/RL-connect4')

    from pprint import pprint
    from torchsummary import summary
    from src.environment.connect_game_env import ConnectGameEnv

    # architecture from 'src/models/architectures/demo_net.json'
    demo_architecture = {
        'conv_block': [[32, 4, 0], "relu", [64, 3, 0], "tanh"],
        'fc_block': [64, 'relu'],
        'first_head': [32, 'relu', 7],
        'second_head': [16, 'relu', 1],
        'name': 'demo_net({n_params})'
    }

    # 1) Create a network with the architecture defined above
    model = CustomNetwork(**demo_architecture)

    # 2) Check that all the layers have been properly created and added
    print(model.name)
    print()
    pprint(model.arch)
    print()
    model.print_blocks()
    print()
    print(summary(model, input_size=model.input_shape), '\n')

    # 3) Test that 'obs_to_model_input' and 'forward' methods work well
    obs = ConnectGameEnv.random_observation()
    model_input = model.obs_to_model_input(obs=obs)  # .to(device)
    random_out = model(model_input)
    print('(2-heads) output for a random observation:', random_out, '\n')

    # 4) Print the first parameters of the network (they are random values)
    for param in model.parameters():
        if param.requires_grad:
            print('first parameters (model):', param[0].squeeze()[0])
            break

    # 5) Create a set of random training hparams to test 'save_weights' method
    test_training_hparams = {
        'num_steps': 100, 'lr': 0.45, 'mse_loss': False, 'rule': 'q_learning'}
    weights_file_path = './src/models/saved_models/demo_net.pt'
    model.save_weights(
        file_path=weights_file_path,
        training_hparams=test_training_hparams,
    )
    print(f"model weights successfully saved in '{weights_file_path}'", '\n')

    # 6) Create a new CustomNetwork that will follow the same architecture as
    # the previous network. But now, the architecture will be loaded from a
    # json file where the layers are already specified. This json file was
    # already created with the exact same architecture as 'model'
    model2 = CustomNetwork.from_architecture(
        file_path='src/models/architectures/demo_net.json',
    )

    # 7) The new network (model2) loads the weights saved by the previous model
    model2.load_weights(file_path=weights_file_path)
    print(f"model weights successfully loaded from '{weights_file_path}'")

    # 8) Print the first parameters of the new model and check that they match
    # the old weights (saved and loaded correctly)
    for param in model2.parameters():
        if param.requires_grad:
            print('first parameters (model2):', param[0].squeeze()[0])
            break

    print('are the saved and the loaded weights exactly the same?')
    print("yess?")
