import torch
import random
import cv2
import numpy as np
from PIL import Image
from matplotlib import style

# set plot style to "ggplot" for visual consistency
style.use("ggplot")


class Tetris:
    # define colors for different tetris pieces
    piece_colors = [
        (255, 255, 255),  # white
        (54, 175, 144),   # teal
        (255, 255, 0),    # yellow
        (255, 0, 0),      # red
        (102, 217, 238),  # cyan
        (147, 88, 254),   # purple
        (0, 0, 255),      # blue
        (254, 151, 32)    # orange
    ]

    # define the shapes of all available tetris pieces
    pieces = [
        [[1, 1],
         [1, 1]],  # O piece

        [[0, 2, 0],
         [2, 2, 2]],  # T piece

        [[0, 3, 3],
         [3, 3, 0]],  # S piece

        [[4, 4, 0],
         [0, 4, 4]],  # Z piece

        [[5, 5, 5, 5]],  # I piece

        [[0, 0, 6],
         [6, 6, 6]],  # J piece

        [[7, 0, 0],
         [7, 7, 7]]   # L piece
    ]

    def __init__(self, height=20, width=10, block_size=20):
        # initialize board dimensions and block size
        self.height = height
        self.width = width
        self.block_size = block_size

        # create an additional board panel for information display and set text color
        self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([96, 96, 96], dtype=np.uint8)
        self.text_color = (255, 255, 255)  # white for text

        # initialize the game by resetting the board
        self.reset()

    def reset(self):
        # initialize an empty board and reset score counters
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0

        # set up a new shuffled bag of pieces and pick the first one
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]

        # position the current piece at the top center of the board
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False
        return self.get_state_properties(self.board)

    def rotate(self, piece):
        # rotate a piece 90 degrees clockwise
        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array

    def get_state_properties(self, board):
        # calculate key game state properties like lines cleared, holes, and bumpiness
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def get_holes(self, board):
        # count the number of holes (empty cells beneath filled ones)
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board):
        # calculate the bumpiness (column height differences) and total board height
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_next_states(self):
        # determine all possible next states for the current piece
        states = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]
        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_state_properties(board)
            curr_piece = self.rotate(curr_piece)
        return states

    def get_current_board_state(self):
        # return the board state including the current active piece
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        return board

    def new_piece(self):
        # select a new piece from the bag and position it at the top center
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        # check if the new piece collides with existing blocks, ending the game
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        # check for collisions between the piece and existing blocks or the board's boundaries
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    def truncate(self, piece, pos):
        # remove rows of the piece that overflow the top boundary of the board
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    def store(self, piece, pos):
        # store the current piece on the board at its final position
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        # find and mark fully filled rows for deletion
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        # remove fully filled rows from the board and add empty rows at the top
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

    def step(self, action, render=True, video=None):
        # perform an action by moving and/or rotating the current piece
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        # move the piece downward until a collision occurs
        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            if render:
                self.render(video)

        # adjust the piece if it exceeds the top boundary
        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True

        # store the piece at its final position and update the board
        self.board = self.store(self.piece, self.current_pos)

        # count cleared rows, update scores, and decide whether to continue or end
        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        self.tetrominoes += 1
        self.cleared_lines += lines_cleared
        if not self.gameover:
            self.new_piece()
        if self.gameover:
            self.score -= 2

        return score, self.gameover

    def render(self, video=None):
        # larger block size to expand the main tetris board
        large_block_size = self.block_size * 2

        # create the main tetris board image
        if not self.gameover:
            img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        else:
            img = [self.piece_colors[p] for row in self.board for p in row]
        img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        # resize the main board with larger blocks
        img = img.resize((self.width * large_block_size, self.height * large_block_size), 0)
        img = np.array(img)

        # draw larger grid lines on the resized main board
        img[[i * large_block_size for i in range(self.height)], :, :] = 0
        img[:, [i * large_block_size for i in range(self.width)], :] = 0

        # create a wider bottom panel to hold the text data
        wider_panel_width = img.shape[1]
        bottom_panel_height = 5 * large_block_size
        bottom_panel = np.ones((bottom_panel_height, wider_panel_width, 3), dtype=np.uint8) * np.array([96, 96, 96], dtype=np.uint8)

        # position the text properly on the bottom panel
        text_offset_x = int(large_block_size / 2)
        cv2.putText(bottom_panel, "score:", (text_offset_x, int(1.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)
        cv2.putText(bottom_panel, str(self.score),
                    (text_offset_x, int(2.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)

        cv2.putText(bottom_panel, "tetrominos:", (text_offset_x, int(3.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)
        cv2.putText(bottom_panel, str(self.tetrominoes),
                    (text_offset_x, int(4.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)

        cv2.putText(bottom_panel, "lines:", (text_offset_x + 350, int(1.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)
        cv2.putText(bottom_panel, str(self.cleared_lines),
                    (text_offset_x + 350, int(2.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)

        # pad the tetris board to have the same width as the bottom panel
        pad_width = wider_panel_width - img.shape[1]
        img_padded = np.pad(img, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

        # concatenate the wider bottom panel with the padded main board
        full_img = np.concatenate((img_padded, bottom_panel), axis=0)

        # record the game video if the option is specified
        if video:
            video.write(full_img)

        # display the final image with text and updated scores
        cv2.imshow("421 final tetris", full_img)
        cv2.waitKey(1)
