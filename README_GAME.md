# PLAY THE GAME

Here we explain the steps to follow in order to play Connect4 against our trained Agents.
-----------------------------------

**(1)** Open a Terminal and go to the directory where you want to clone the repository.

```
cd <YOUR PATH>
```


**(2)** Clone the Github repository

```
git clone https://github.com/marcpaulo15/RL-connect4
```


**(3)** Create and activate a Python virtual environment. Install the requirements.

```
python3 -m venv .connect4_env &&
source .connect4_env/bin/activate &&
pip install -r RL-connect4/requirements.txt
```


**(4)** Add the project libraries to the PYTHONPATH variable (environment variable).

```
cd RL-connect4 &&
export PYTHONPATH="$PWD"
```

If you don't do this, you will get the following error:
```
File "src/game/connect_game_main.py", line 7, in <module>
    from src.game.game_logic.connect_game_menu import ConnectGameMenu
ModuleNotFoundError: No module named 'src'
```


**(5)** Run the Python code to play the game

```
python3 src/game/connect_game_main.py
```


**(SHORTCUT)** If you already have the code and the environment, you can run:

```
source .connect4_env/bin/activate &&
cd RL-connect4 &&
export PYTHONPATH="$PWD" &&
python3 src/game/connect_game_main.py
```
