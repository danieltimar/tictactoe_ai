import numpy as np
import random
import re
import pickle
import os.path
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

players = ['X', 'O']


def set_up_board(board_size, human_plays=False):
    empty_board = np.array([[' '] * board_size for i in range(board_size)])
    current_state = "Not Done"
    #print_board(empty_board)
    #print("\nNew Game!")
    step_count = 0
    player_choice = input("Choose which player goes first - X (You) or O (AI): ") if human_plays else None
    if player_choice is None:
        current_player_idx = random.choice([0, 1])
    elif player_choice.lower() == 'x':
        current_player_idx = 0
    else:
        current_player_idx = 1
    winner = None
    return empty_board, current_state, current_player_idx, winner, step_count


def get_input(prompt='Your turn, type a row and a column number:') -> (int, int):

    input_digits = re.search(pattern='(?P<row>\d+)[ ,]+(?P<col>\d+)', string=input(prompt))
    while input_digits is None:
        input_digits = re.search(pattern='(?P<row>\d+)[ ,]+(?P<col>\d+)', string=input("Type a row and a column number, separated, e.g. 2,3 or 2 3"))

    row_number = int(input_digits.group('row'))-1  # 0-based indexing
    col_number = int(input_digits.group('col'))-1  # 0-based indexing

    return row_number, col_number


def find_empty_fields(state: np.ndarray) -> list:
    empty_cells = [(i, j) for j in range(len(state)) for i in range(len(state)) if state[i, j] == ' ']
    return empty_cells


def play_move(state, player, block_coord):
    if state[block_coord] == ' ':
        state[block_coord] = player
    else:
        block_coord = get_input("Block is not empty! Choose again: ")
        play_move(state=state, player=player, block_coord=block_coord)


def copy_game_state(state):
    state_copy = np.array(state)
    return state_copy


def check_current_state(state):

    # Check if any of the players won
    for player in players:
        all_moves_for_player = [(row, col) for row in range(board_size) for col in range(board_size)
                                if state[row, col] == player]
        all_moves_for_player.sort()

        sequence_counter = 0
        for mark in all_moves_for_player:
            # Check horizontal
            for i in range(mark_to_win):
                if (mark[0], mark[1]+i) in all_moves_for_player:
                    sequence_counter += 1
                    if sequence_counter == mark_to_win:
                        return player, "Done"
                else:
                    sequence_counter = 0
                    break
            # Check vertical
            for i in range(mark_to_win):
                if (mark[0]+i, mark[1]) in all_moves_for_player:
                    sequence_counter += 1
                    if sequence_counter == mark_to_win:
                        return player, "Done"
                else:
                    sequence_counter = 0
                    break

            # Check diagonal right
            for i in range(mark_to_win):
                if (mark[0]+i, mark[1]+i) in all_moves_for_player:
                    sequence_counter += 1
                    if sequence_counter == mark_to_win:
                        return player, "Done"
                else:
                    sequence_counter = 0
                    break

            # Check diagonal left
            for i in range(mark_to_win):
                if (mark[0]+i, mark[1]-i) in all_moves_for_player:
                    sequence_counter += 1
                    if sequence_counter == mark_to_win:
                        return player, "Done"
                else:
                    sequence_counter = 0
                    break

    # Check if draw (board is full)
    draw_flag = 0
    for i in range(len(state)):
        for j in range(len(state)):
            if state[i, j] == ' ':
                draw_flag = 1
    if draw_flag is 0:
        return None, "Draw"

    return None, "Not Done"


def print_board(state):

    board_side = len(state)

    col_num_to_print = [str(i+1) for i in range(board_side)]
    print('    ', '    '.join(col_num_to_print), '  ', sep='')
    print('  ', '-----'*board_side, sep='')
    for i, row in enumerate(state):
        cols_to_print = [str(col) for col in row]
        print(i+1, ' | ', ' || '.join(cols_to_print), ' |', sep='')
        print('  ', '-----'*board_side, sep='')


def find_all_equivalent_states(state, state_dict):
    rotate0 = state
    rotate90 = np.rot90(rotate0)
    rotate180 = np.rot90(rotate90)
    rotate270 = np.rot90(rotate180)
    flip0 = np.flipud(rotate0)
    flip90 = np.rot90(flip0)
    flip180 = np.rot90(flip90)
    flip270 = np.rot90(flip180)

    state_vars = [rotate0, rotate90, rotate180, rotate270, flip0, flip90, flip180, flip270]
    convert_state_vars = [convert_state_to_key(var) for var in state_vars]
    found_in_history = [var in state_dict for var in convert_state_vars]

    if any(found_in_history):
        key_in_history = convert_state_vars[found_in_history.index(True)]
        return True, key_in_history
    else:
        return False, None


def convert_state_to_key(state):
    return tuple(state.flatten())


def load_historical_state_values(path_to_files=f'state_values/'):

    board_size_dir = f'{board_size}_x_{board_size}/'
    path_to_file_x = path_to_files+board_size_dir+f'all_game_states_dict_x.pkl'

    if os.path.exists(path_to_file_x):
        with open(file=path_to_file_x, mode='rb') as file_x:
            all_game_states_dict_x = pickle.load(file_x)
    else:
        starting_game_state = np.array([[' '] * board_size for i in range(board_size)])
        game_state_as_key = convert_state_to_key(starting_game_state)
        all_game_states_dict_x = {game_state_as_key: 0}
        # with open(path_to_file_x, 'wb') as file_x:
        #     pickle.dump(all_game_states_dict_x, file_x)

    path_to_file_o = path_to_files+board_size_dir+f'all_game_states_dict_o.pkl'
    if os.path.exists(path_to_file_o):
        with open(file=path_to_file_o, mode='rb') as file_o:
            all_game_states_dict_o = pickle.load(file_o)
    else:
        starting_game_state = np.array([[' '] * board_size for i in range(board_size)])
        game_state_as_key = convert_state_to_key(starting_game_state)
        all_game_states_dict_o = {game_state_as_key: 0}
        # with open(path_to_file_o, 'wb') as file_o:
        #     pickle.dump(all_game_states_dict_o, file_o)

    return all_game_states_dict_x, all_game_states_dict_o


def save_historical_state_values(states_x, states_o, path_to_files=f'state_values/'):
    board_size_dir = f'{board_size}_x_{board_size}/'

    if not os.path.exists(path_to_files+board_size_dir):
        os.mkdir(path_to_files+board_size_dir)

    path_to_file_x = path_to_files+board_size_dir+f'all_game_states_dict_x.pkl'
    with open(path_to_file_x, 'wb') as file_x:
        pickle.dump(states_x, file_x)

    path_to_file_o = path_to_files+board_size_dir+f'all_game_states_dict_o.pkl'
    with open(path_to_file_o, 'wb') as file_o:
        pickle.dump(states_o, file_o)

    print('Updated state values saved!')


def update_state_values(winning_player, game_history, learning_rate=0.2):

    game_length = len(game_history)
    winning_state = game_history[0]

    if winning_player == 'X':
        all_game_states_dict_x[convert_state_to_key(winning_state)] = 1
        all_game_states_dict_o[convert_state_to_key(winning_state)] = -1

    elif winning_player == 'O':
        all_game_states_dict_x[convert_state_to_key(winning_state)] = -1
        all_game_states_dict_o[convert_state_to_key(winning_state)] = 1
    else:
        all_game_states_dict_x[convert_state_to_key(winning_state)] = 0
        all_game_states_dict_o[convert_state_to_key(winning_state)] = 0

    for i in range(game_length-2):  # No point in modifying first entry
        new_status = game_history[i]
        prev_status = game_history[i+1]

        for all_game_states_dict in (all_game_states_dict_x, all_game_states_dict_o):

            found_prev, key_prev = find_all_equivalent_states(prev_status, all_game_states_dict)
            found_new, key_new = find_all_equivalent_states(new_status, all_game_states_dict)

            if not found_prev:
                prev_status_as_key = convert_state_to_key(prev_status)
                all_game_states_dict[prev_status_as_key] = 0
                new_value = all_game_states_dict[prev_status_as_key] + learning_rate * (
                        all_game_states_dict[key_new] -
                        all_game_states_dict[prev_status_as_key])
                all_game_states_dict[prev_status_as_key] = new_value
            else:
                new_value = all_game_states_dict[key_prev] + learning_rate * (
                        all_game_states_dict[key_new] - all_game_states_dict[key_prev])
                all_game_states_dict[key_prev] = new_value


def find_best_move(state, player):

    potential_moves = find_empty_fields(state)

    global epsilon
    # global step_count

    # Explore
    if (np.random.uniform(0, 1) <= epsilon) or (step_count == 1):
        best_move = random.choice(potential_moves)
        epsilon *= 0.99
        #  print('AI decides to explore! Takes action = ' + str(best_move))

    # Exploit
    else:
        curr_state_values = []
        all_game_states_dict = all_game_states_dict_x if player.lower() == 'x' else all_game_states_dict_o

        for move in potential_moves:
            new_state = copy_game_state(state)
            play_move(new_state, player, move)

            found, new_state_key = find_all_equivalent_states(new_state, all_game_states_dict)

            if found:
                state_value = all_game_states_dict[new_state_key]
            else:
                state_value = 0

            curr_state_values.append(state_value)

        #print('Possible moves = ' + str(potential_moves))
        #print('Move values = ' + str(curr_state_values))

        best_move_idx = np.argmax(curr_state_values)
        best_move = potential_moves[best_move_idx]
        #print('AI decides to exploit! Takes action = ' + str(best_move))

    return best_move


# =================== GAMEPLAY ===================

# 1. LOAD TRAINED STATE VALUES

if __name__ == '__main__':

    board_size = None
    while type(board_size) != int:
        board_size_str = input('Set board size (3, 5, 7, 10): ')
        while board_size_str not in ('3', '5', '7', '10'):
            board_size_str = input('Size has to be 3, 5, 7, 10:')
        board_size = int(board_size_str)

    if board_size == 3:
        mark_to_win = 3
    elif board_size in (5, 7):
        mark_to_win = 4
    else:
        mark_to_win = 5

    # 1. LOAD TRAINED STATE VALUES
    all_game_states_dict_x, all_game_states_dict_o = load_historical_state_values()

    iterations = 100000
    epsilon = 0.2
    learning_rate = 0.2

    winning_history = []
    step_count_history = []

    for iteration in range(iterations):
        # 2. SETTING UP BOARD
        print(f'Iteration {iteration+1}!')
        game_state, current_state, current_player_idx, winner, step_count = set_up_board(board_size, human_plays=False)

        game_states = []

        while current_state == "Not Done":
            # 3. AI X
            step_count += 1
            if current_player_idx == 0:
                #print("AI X's turn")
                block_choice = find_best_move(game_state, players[current_player_idx])
                play_move(game_state, players[current_player_idx], block_choice)
                game_states.insert(0, copy_game_state(game_state))
            # 4. AI O
            else:
                #print("AI O's turn")
                block_choice = find_best_move(game_state, players[current_player_idx])
                play_move(game_state, players[current_player_idx], block_choice)
                game_states.insert(0, copy_game_state(game_state))

            #print_board(game_state)
            winner, current_state = check_current_state(game_state)
            if winner is not None:
                update_state_values(winning_player=winner, game_history=game_states, learning_rate=learning_rate)
                winning_history.append(str(winner))
                # print(str(winner) + " WON!")

            else:
                current_player_idx = (current_player_idx + 1) % 2

            if current_state is "Draw":
                update_state_values(winning_player=None, game_history=game_states, learning_rate=learning_rate)
                winning_history.append('draw')
                # print("DRAW!")

        step_count_history.append(step_count)

    print(f'{iterations} iterations ran successfully!')
    save_historical_state_values(all_game_states_dict_x, all_game_states_dict_o)

    print(Series(winning_history).value_counts(normalize=True))

    x_counter, o_counter, d_counter = 0, 0, 0
    winning_history_cum = []
    for i in winning_history:
        if i == 'X':
            x_counter += 1
        elif i == 'O':
            o_counter += 1
        else:
            d_counter += 1

        winning_history_cum.append((x_counter, o_counter, d_counter))

    winners_df = DataFrame(winning_history_cum, columns=('x', 'o', 'd'))
    winners_df['iteration'] = np.arange(iterations)
    sns.lineplot('iteration', 'x', data=winners_df, label='x won')
    sns.lineplot('iteration', 'o', data=winners_df, label='o won')
    sns.lineplot('iteration', 'd', data=winners_df, label='draw')
    plt.legend()
    plt.show()

    sns.distplot(step_count_history, hist=True, kde=False, bins=(board_size*board_size))
    plt.xlim(mark_to_win*2-1, board_size*board_size)
    plt.show()




