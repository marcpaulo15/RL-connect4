from typing import List

import numpy as np


def random_action(board: np.ndarray) -> int:
    """
    Returns a random (legal) action for the given board

    :param board: game board
    :return: random (legal) action
    """

    action = np.random.choice(get_legal_actions(board=board))
    return action


def is_game_winner(board: np.ndarray, mark: int, inrow: int = 4) -> bool:
    """
    Checks if the given player ('mark') is the winner of the game ('board').

    :param board: connectX game board
    :param mark: player
    :param inrow: how many pieces in row in order to win
    :return: True if 'mark' is the winner of the game
    """

    target = np.array([mark] * inrow)

    # check rows
    for j in range(board.shape[1] - inrow + 1):
        # look at n-length horizontal sequences starting at column 'j'
        found = np.any(
            np.all(target == board[:, j:j+inrow], axis=1)
        )
        if found:
            return True

    # check columns
    for i in range(board.shape[0] - inrow + 1):
        # look at n-length vertical sequences starting at row 'i'
        found = np.any(
            np.all(target == board[i:i+inrow, :].T, axis=1)
        )
        if found:
            return True

    # check diagonals (ascending and descending)
    for i in range(board.shape[0] - inrow + 1):
        for j in range(board.shape[1] - inrow + 1):
            asc_diag = board[i:i+inrow, j:j+inrow].diagonal()
            des_diag = np.fliplr(board[i:i+inrow, j:j+inrow]).diagonal()
            if np.all(asc_diag == target) or np.all(des_diag == target):
                return True

    return False


def is_a_draw(board: np.ndarray, inrow: int = 4) -> bool:
    """
    Checks if the given game (board) is a draw: the board is full and there is
    no winner

    :param board: connect4 game board
    :param inrow: how many tokens in line to win the game
    :return: whether the game is a draw
    """

    board_is_not_full = (board == 0).sum() != 0
    if board_is_not_full:
        return False  # there are still moves to play
    else:
        return not is_game_winner(board=board, mark=1, inrow=inrow) and \
            not is_game_winner(board=board, mark=2, inrow=inrow)


def is_terminal(board: np.ndarray, inrow: int = 4) -> bool:
    """
    Checks if the given state (board) is a terminal state (game over)

    :param board: game board
    :param inrow: how many tokens in line to win the game
    :return:
    """

    return (board == 0).sum() == 0 or \
        is_game_winner(board=board, mark=1, inrow=inrow) or \
        is_game_winner(board=board, mark=2, inrow=inrow)


def drop_piece(board: np.ndarray, column: int, mark: int) -> np.ndarray:
    """
    Plays one turn and returns the next board. The given action ('column') must
    be a legal action.

    :param board: game board
    :param column: action
    :param mark: player
    :return: next board after the turn ('column') is played
    """

    if is_illegal_action(board=board, action=column):
        raise Exception("illegal action! can't drop piece", f"{board}; {column}")
    next_board = board.copy()
    row = np.where(board[:, column] == 0)[0][-1]
    next_board[row, column] = mark
    return next_board


def get_legal_actions(board: np.ndarray) -> List[int]:
    """
    Returns the list of legal actions for the given board

    :param board: game board
    :return: List of legal actions (int values)
    """

    return np.where(board[0] == 0)[0]


def get_illegal_actions(board: np.ndarray) -> List[int]:
    """
    Returns the list of illegal actions for the given board

    :param board: game board
    :return: List of legal actions (int values)
    """

    return np.where(board[0] != 0)[0]


def is_legal_action(action: int, board: np.ndarray) -> bool:
    """
    Checks if 'action' is legal if played in the given 'board'

    :param action: column (int)
    :param board: game board
    :return: True if the action is legal
    """

    return board[0, action] == 0


def is_illegal_action(action: int, board: np.ndarray) -> bool:
    """
    Checks if 'action' is illegal if played in the given 'board'

    :param action: column (int)
    :param board: game board
    :return: True if the action is illegal
    """

    return board[0, action] != 0


def count_n_in_row(board: np.ndarray, n: int, mark: int, inrow: int = 4) -> int:
    """
    Returns the number of occurrences of 'n' 'marks' in row, with the
    possibility of completing 'inrow' pieces in that line (i.e. the opponent is
    not blocking the line). This function is used by the NStepLookAheadAgent to
    evaluate boards

    :param board: game board
    :param n: number of 'marks' in a row
    :param mark: player's marks
    :param inrow: how many marks in a row are required to win the game
    :return: number of occurrences of the given pattern
    """

    count = 0
    # functions to check that there are 'n' 'marks' and 'n'-1 empty positions
    check_marks = lambda ww: (ww == mark).sum(axis=1) == n
    check_spaces = lambda ww: (ww == 0).sum(axis=1) == inrow-n

    # check rows
    for j in range(board.shape[1] - inrow + 1):
        w = board[:, j:j+inrow]  # window
        count += np.sum(check_spaces(w) & check_marks(w))

    # check columns
    for i in range(board.shape[0] - inrow + 1):
        w = board[i:i+inrow, :].T  # window
        count += np.sum(check_spaces(w) & check_marks(w))

    # check diagonals
    check_marks = lambda ww: (ww == mark).sum() == n
    check_spaces = lambda ww: (ww == 0).sum() == inrow-n
    for i in range(board.shape[0] - inrow + 1):
        for j in range(board.shape[1] - inrow + 1):
            asc_diag = board[i:i+inrow, j:j+inrow].diagonal()
            des_diag = np.fliplr(board[i:i+inrow, j:j+inrow]).diagonal()
            count += np.sum(check_spaces(asc_diag) & check_marks(asc_diag))
            count += np.sum(check_spaces(des_diag) & check_marks(des_diag))

    return count


def get_winning_cols(board: np.array, mark: int, inrow: int = 4) -> List[int]:
    """
    Returns the list of the actions (columns) that allow the player 'mark' to
    win the game if one of these columns is played in that turn.

    :param board: game board
    :param mark: player's mark
    :param inrow: how many marks in a row are required to win the game
    :return: List of winning columns for the given 'mark' in the given 'board'
    """

    winning_columns = []
    for column in get_legal_actions(board=board):
        next_board = drop_piece(board=board, column=column, mark=mark)
        if is_game_winner(board=next_board, mark=mark, inrow=inrow):
            winning_columns.append(column)
    return winning_columns


if __name__ == "__main__":
    # DEMO

    board = np.array([
        [2, 0, 0, 0, 1],
        [2, 0, 2, 0, 2],
        [1, 2, 1, 0, 2],
        [1, 1, 2, 0, 2],
        [1, 1, 1, 0, 1],
    ])
    print('board:\n', board)
    random_board = np.random.randint(0, 3, 42).reshape((6, 7))
    print('random_board:\n', random_board)

    print('\ncount_n_in_row(board=board):')
    for mark in (1, 2):
        for n in range(1, 5):
            res = count_n_in_row(board=board, n=n, mark=mark)
            print(f"mark={mark}, n={n}: {res}")

    print('\ncount_n_in_row(board=random_board):')
    for mark in (1, 2):
        for n in range(1, 5):
            res = count_n_in_row(board=random_board, n=n, mark=mark)
            print(f"mark={mark}, n={n}: {res}")

    print()
    print("get_legal_actions =", get_legal_actions(board=board))
    print("get_illegal_actions =", get_illegal_actions(board=board))
    print()
    print("is_game_winner(1)? [NO] ", is_game_winner(mark=1, board=board))
    print("is_game_winner(2)? [NO] ", is_game_winner(mark=2, board=board))
    print("is_terminal? [NO]", is_terminal(board=board))
    print()
    b2 = drop_piece(board=board, column=3, mark=1)
    print("is_game_winner(1, b2)? [YES] ", is_game_winner(mark=1, board=b2))
    print("is_terminal(b2)? [YES]", is_terminal(board=b2))

    draw_board = np.array([
        [1, 2, 1, 2, 1, 2],
        [1, 2, 1, 2, 1, 2],
        [1, 2, 1, 2, 1, 2],
        [2, 1, 2, 1, 2, 1],
        [1, 2, 1, 2, 1, 2],
        [1, 2, 1, 2, 1, 2],
    ])
    print('\ndraw_board is a draw:\n', draw_board)
    print("is_a_draw(draw_board)? [YES]", is_a_draw(board=draw_board))
    print("is_terminal(draw_board)? [YES]", is_terminal(board=draw_board))
    print("is_game_winner(mark=1)? [NO]", is_game_winner(draw_board, mark=1))
    print("is_game_winner(mark=2)? [NO]", is_game_winner(draw_board, mark=1))
