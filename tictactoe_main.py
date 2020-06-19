import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import pickle
import re
import os
from math import inf

#BOARD_SIZE = 3

class Board:
    def __init__(self, player1, player2, board_size=3):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size))
        self.player1 = player1
        self.player2 = player2
        self.gameover = False
        self.board_as_key = None
        self.player_mark = 1

        if board_size == 3:
            self.mark_to_win = 3
        elif board_size in (5, 7):
            self.mark_to_win = 4
        else:
            self.mark_to_win = 5

    def encode_to_key(self):
        self.board_as_key = str(self.board.reshape(self.board_size * self.board_size))
        return self.board_as_key

    def winner(self):

        # Check if any of the players won
        for p in [1, -1]:
            all_moves_for_player = [(row, col) for row in range(self.board_size) for col in range(self.board_size)
                                    if self.board[row, col] == p]
            all_moves_for_player.sort()

            sequence_counter = 0
            for mark in all_moves_for_player:
                # Check horizontal
                for i in range(self.mark_to_win):
                    if (mark[0], mark[1] + i) in all_moves_for_player:
                        sequence_counter += 1
                        if sequence_counter == self.mark_to_win:
                            self.gameover = True
                            return p
                    else:
                        sequence_counter = 0
                        break
                # Check vertical
                for i in range(self.mark_to_win):
                    if (mark[0] + i, mark[1]) in all_moves_for_player:
                        sequence_counter += 1
                        if sequence_counter == self.mark_to_win:
                            self.gameover = True
                            return p
                    else:
                        sequence_counter = 0
                        break

                # Check diagonal right
                for i in range(self.mark_to_win):
                    if (mark[0] + i, mark[1] + i) in all_moves_for_player:
                        sequence_counter += 1
                        if sequence_counter == self.mark_to_win:
                            self.gameover = True
                            return p
                    else:
                        sequence_counter = 0
                        break

                # Check diagonal left
                for i in range(self.mark_to_win):
                    if (mark[0] + i, mark[1] - i) in all_moves_for_player:
                        sequence_counter += 1
                        if sequence_counter == self.mark_to_win:
                            self.gameover = True
                            return p
                    else:
                        sequence_counter = 0
                        break

        # draw
        draw_flag = None
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    draw_flag = 1
        if draw_flag is None:
            self.gameover = True
            return 0

        self.gameover = False
        return None

    def potential_moves(self):
        positions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def make_move(self, position):
        if self.board[position] == 0:
            self.board[position] = self.player_mark
        else:
            position = self.player2.pick_next_move()
            self.board[position] = self.player_mark
        self.player_mark = -1 if self.player_mark == 1 else 1

    def giveReward(self):
        result = self.winner()

        if result == 1:
            self.player1.backpropagate_reward(1)
            self.player2.backpropagate_reward(-1)
        elif result == -1:
            self.player1.backpropagate_reward(-1)
            self.player2.backpropagate_reward(1)
        else:
            self.player1.backpropagate_reward(-0.5)
            self.player2.backpropagate_reward(-0.5)

    def reset_board(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.board_as_key = None
        self.gameover = False
        self.player_mark = 1

    def play(self, rounds=100, print_stats=False):
        winning_history = []
        for i in range(rounds):
            if i % 100 == 0:
                print("Rounds {}".format(i))
            while not self.gameover:
                positions = self.potential_moves()
                player1_action = self.player1.pick_next_move(positions, self.board, self.player_mark)
                self.make_move(player1_action)
                board_hash = self.encode_to_key()
                self.player1.append_board_state(board_hash)

                win = self.winner()
                if win is not None:
                    # self.print_board()
                    winning_history.append(win)
                    self.giveReward()
                    self.player1.reset_player()
                    self.player2.reset_player()
                    self.reset_board()
                    break

                else:
                    positions = self.potential_moves()
                    player2_action = self.player2.pick_next_move(positions, self.board, self.player_mark)
                    self.make_move(player2_action)
                    board_hash = self.encode_to_key()
                    self.player2.append_board_state(board_hash)

                    win = self.winner()
                    if win is not None:
                        winning_history.append(win)
                        # ended with player2 either win or draw
                        self.giveReward()
                        self.player1.reset_player()
                        self.player2.reset_player()
                        self.reset_board()
                        break

        self.player1.save_state_values()
        self.player2.save_state_values()

        if print_stats:
            winning_series = Series(winning_history)
            print(winning_series.value_counts(normalize=True, ascending=False))
            winning_df = DataFrame({'wins': winning_series, 'values': np.ones(len(winning_series))})
            df_to_plot = winning_df.pivot(columns='wins', values='values').fillna(0).cumsum()
            for c in df_to_plot.columns.values:
                plt.plot(df_to_plot[c], label=f'Winner:{c}')
            plt.legend()
            plt.show()


    # play with human player
    def play2(self):
        while not self.gameover:
            # Player 1
            positions = self.potential_moves()
            player1_action = self.player1.pick_next_move(positions, self.board, self.player_mark)
            self.make_move(player1_action)
            self.print_board()
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.player1.name, "wins!")
                else:
                    print("tie!")
                self.reset_board()
                break

            else:
                # Player 2
                positions = self.potential_moves()
                player2_action = self.player2.pick_next_move(positions)

                self.make_move(player2_action)
                self.print_board()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.player2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset_board()
                    break

    def print_board(self):
        col_num_to_print = [str(i + 1) for i in range(self.board_size)]
        print('    ', '    '.join(col_num_to_print), '  ', sep='')
        print('  ', '-----' * self.board_size, sep='')
        for i, row in enumerate(self.board):
            marks_in_row=[]
            for col in row:
                if col == 1:
                    marks_in_row.append('X')
                elif col == -1:
                    marks_in_row.append('O')
                else:
                    marks_in_row.append(' ')
            print(i + 1, ' | ', ' || '.join(marks_in_row), ' |', sep='')
            print('  ', '-----' * self.board_size, sep='')


class Player:
    def __init__(self, name, exp_rate=0.3, lr_rate=0.2, board_size=3):
        self.name = name
        self.states = []
        self.lr = lr_rate
        self.exp_rate_initial = exp_rate
        self.exp_rate = exp_rate
        self.decay_gamma = 1
        self.states_value = {}
        self.board_size = board_size

    def encode_to_key(self, board):
        board_as_key = str(board.reshape(self.board_size * self.board_size))
        return board_as_key

    def decode_from_key(self, board_hashed):
        unhashed_list = re.findall(pattern=r'-?\d', string=board_hashed)
        return np.array(unhashed_list).reshape(self.board_size, self.board_size)

    def pick_next_move(self, positions, current_board, symbol):
        if (np.random.uniform(0, 1) <= self.exp_rate) or (len(positions) == self.board_size**2):
            idx = np.random.choice(len(positions))
            action = positions[idx]
            # self.exp_rate = self.exp_rate * 0.999
        else:
            value_max = -inf
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                st_equivalent = self.find_equivalent_state(next_board)
                if st_equivalent is None:
                    value = 0
                else:
                    value = self.states_value.get(st_equivalent)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p

        return action

    def append_board_state(self, state):
        self.states.append(state)

    def find_equivalent_state(self, state):
        rotate0 = state
        rotate90 = np.rot90(state, k=1)
        rotate180 = np.rot90(state, k=2)
        rotate270 = np.rot90(state, k=3)
        flip0 = np.flipud(state)
        flip90 = np.rot90(flip0, k=1)
        fliplayer180 = np.rot90(flip0, k=2)
        fliplayer270 = np.rot90(flip0, k=3)

        equivalent_states = [rotate0, rotate90, rotate180, rotate270, flip0, flip90, fliplayer180, fliplayer270]
        # print(equivalent_states)
        hashed_equivalent_states = [self.encode_to_key(var) for var in equivalent_states]
        in_states_value = [var in self.states_value for var in hashed_equivalent_states]

        if any(in_states_value):
            key_in_history = hashed_equivalent_states[in_states_value.index(True)]
            return key_in_history
        else:
            return None

    def backpropagate_reward(self, reward):
        for i, state in enumerate(reversed(self.states)):
            state_decoded = self.decode_from_key(state)
            state_equivalent = self.find_equivalent_state(state_decoded)
            if state_equivalent is None:
                if i == 0:
                    self.states_value[state] = reward
                else:
                    self.states_value[state] = self.lr * (self.decay_gamma * reward) # initial value 0 = 0
                    reward = self.states_value[state]
            else:
                if i == 0:
                    continue
                else:
                    self.states_value[state_equivalent] += self.lr * (self.decay_gamma * reward - self.states_value[state_equivalent])
                    reward = self.states_value[state_equivalent]

    def reset_player(self):
        self.states = []
        self.exp_rate = self.exp_rate_initial

    def save_state_values(self):

        folder_path = f'state_values/{self.board_size}_x_{self.board_size}'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        fw = open(f'{folder_path}/ttt_state_values_{str(self.name)}', 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def load_state_values(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def get_input(self):
        input_digits = re.search(pattern='(?P<row>\d+)[ ,]+(?P<col>\d+)', string=input('Type row and column number: '))
        while input_digits is None:
            input_digits = re.search(pattern='(?P<row>\d+)[ ,]+(?P<col>\d+)',
                                     string=input("Type a row and a column number, separated, e.g. 2,3 or 2 3"))

        row_number = int(input_digits.group('row')) - 1  # 0-based indexing
        col_number = int(input_digits.group('col')) - 1  # 0-based indexing

        return row_number, col_number

    def pick_next_move(self, positions):
        while True:
            row, col = self.get_input()
            action = (row, col)
            if action in positions:
                return action

    def append_board_state(self, state):
        pass

    def backpropagate_reward(self, reward):
        pass

    def reset_player(self):
        pass


if __name__ == "__main__":

    play_mode = int(input('Select game mode(0=AIvAI, 1=AIvHuman, 2=Both):'))

    board_size = None
    while board_size not in (3, 5, 7, 10):
        board_size = int(input('Select board size (3, 5, 7 or 10):'))

    exploration_rate = None
    while (exploration_rate is None) or (not 0 <= exploration_rate <= 1):
        if play_mode in (0, 2):
            exploration_rate = float(input('Select rate of exploration (between 0 and 1): '))
        else:
            break

    learning_rate = None
    while (learning_rate is None) or (not 0 <= learning_rate <= 1):
        if play_mode in (0, 2):
            learning_rate = float(input('Select learning rate (between 0 and 1): '))
        else:
            break

    iterations = None
    while (iterations is None) or (not 0 <= iterations <= 100000):
        if play_mode in (0, 2):
            iterations = int(input('Type number of iterations: '))
        else:
            break


    # training with AI
    if play_mode in (0, 2):
        iterations = iterations
        player1 = Player("player1", board_size=board_size, exp_rate=exploration_rate, lr_rate=learning_rate)
        player2 = Player("player2", board_size=board_size, exp_rate=exploration_rate, lr_rate=learning_rate)

        st = Board(player1, player2, board_size=board_size)
        print("training...")
        st.play(iterations, print_stats=True)

    # play with human
    if play_mode in (1, 2):
        play_again = 'y'
        while play_again.lower() == 'y':
            player1 = Player("computer", exp_rate=0, board_size=board_size)
            player1.load_state_values(f"state_values/{player1.board_size}_x_{player1.board_size}/ttt_state_values_player1")

            player2 = HumanPlayer("human")

            st = Board(player1, player2, board_size=board_size)
            st.play2()
            play_again = input('Want to play again? (y/n)')

