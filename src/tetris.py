import torch
import random
import cv2
import numpy as np
from PIL import Image
from matplotlib import style

style.use("ggplot")


class Tetris:
    piece_colors = [
        (255, 255, 255),
        (54, 175, 144),
        (255, 255, 0),
        (255, 0, 0),
        (102, 217, 238),
        (147, 88, 254),
        (0, 0, 255),
        (254, 151, 32),
    ]

    pieces = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5, 5, 5, 5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]]
    ]

    def __init__(self, height=20, width=10, block_size=20):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([96, 96, 96], dtype=np.uint8)
        self.text_color = (255, 255, 255)
        self.reset()

    def reset(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False
        return self.get_state_properties(self.board)

    def rotate(self, piece):
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
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def get_holes(self, board):
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board):
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
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        return board

    def new_piece(self):
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                            "y": 0
                            }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    def truncate(self, piece, pos):
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
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

    def step(self, action, render=True, video=None):
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            if render:
                self.render(video)

        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True

        self.board = self.store(self.piece, self.current_pos)

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
        # Larger block size to expand the main Tetris board
        large_block_size = self.block_size * 2

        # Create the main Tetris board image
        if not self.gameover:
            img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        else:
            img = [self.piece_colors[p] for row in self.board for p in row]
        img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        # Resize the main board with larger blocks
        img = img.resize((self.width * large_block_size, self.height * large_block_size), 0)
        img = np.array(img)

        # Draw larger grid lines on the resized main board
        img[[i * large_block_size for i in range(self.height)], :, :] = 0
        img[:, [i * large_block_size for i in range(self.width)], :] = 0

        # Create a wider bottom panel to hold the text data
        wider_panel_width = img.shape[1] 
        bottom_panel_height = 5 * large_block_size
        bottom_panel = np.ones((bottom_panel_height, wider_panel_width, 3), dtype=np.uint8) * np.array([96, 96, 96], dtype=np.uint8)

        # Position the text properly on the bottom panel
        text_offset_x = int(large_block_size / 2)
        cv2.putText(bottom_panel, "Score:", (text_offset_x, int(1.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)
        cv2.putText(bottom_panel, str(self.score),
                    (text_offset_x, int(2.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)

        cv2.putText(bottom_panel, "Tetrominos:", (text_offset_x, int(3.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)
        cv2.putText(bottom_panel, str(self.tetrominoes),
                    (text_offset_x, int(4.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)

        cv2.putText(bottom_panel, "Lines:", (text_offset_x + 350, int(1.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)
        cv2.putText(bottom_panel, str(self.cleared_lines),
                    (text_offset_x + 350, int(2.5 * large_block_size)),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=2.0, color=self.text_color)

        # Pad the Tetris board to have the same width as the bottom panel
        pad_width = wider_panel_width - img.shape[1]
        img_padded = np.pad(img, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

        # Concatenate the wider bottom panel with the padded main board
        full_img = np.concatenate((img_padded, bottom_panel), axis=0)

        if video:
            video.write(full_img)

        cv2.imshow("421 Final Tetris", full_img)
        cv2.waitKey(1)

