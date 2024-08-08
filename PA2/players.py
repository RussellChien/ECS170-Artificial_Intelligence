from copy import deepcopy
import random
import time
import pygame
import math
from connect4 import connect4
import numpy as np

class connect4Player(object):
    def __init__(self, position, seed=0, CVDMode=False):
        self.position = position
        self.opponent = None
        self.seed = seed
        random.seed(seed)
        if CVDMode:
            global P1COLOR
            global P2COLOR
            P1COLOR = (227, 60, 239)
            P2COLOR = (0, 255, 0)

    def play(self, env: connect4, move: list) -> None:
        move = [-1]

class human(connect4Player):

    def play(self, env: connect4, move: list) -> None:
        move[:] = [int(input('Select next move: '))]
        while True:
            if int(move[0]) >= 0 and int(move[0]) <= 6 and env.topPosition[int(move[0])] >= 0:
                break
            move[:] = [int(input('Index invalid. Select next move: '))]

class human2(connect4Player):

    def play(self, env: connect4, move: list) -> None:
        done = False
        while(not done):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
                    posx = event.pos[0]
                    if self.position == 1:
                        pygame.draw.circle(screen, P1COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
                    else: 
                        pygame.draw.circle(screen, P2COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
                pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    posx = event.pos[0]
                    col = int(math.floor(posx/SQUARESIZE))
                    move[:] = [col]
                    done = True

class randomAI(connect4Player):

    def play(self, env: connect4, move: list) -> None:
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)
        move[:] = [random.choice(indices)]

class stupidAI(connect4Player):

    def play(self, env: connect4, move: list) -> None:
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)
        if 3 in indices:
            move[:] = [3]
        elif 2 in indices:
            move[:] = [2]
        elif 1 in indices:
            move[:] = [1]
        elif 5 in indices:
            move[:] = [5]
        elif 6 in indices:
            move[:] = [6]
        else:
            move[:] = [0]
            

class minimaxAI(connect4Player):
    
    def minimax(self, valid_moves, best_move, env, depth, position):
        if depth == 0:
            return self.eval_func(env.board), None
        
        maximizingPlayer = True if position == 1 else False
        
        if maximizingPlayer:
            max_eval = float('-inf')
            for curr_move in valid_moves:
                temp_env = deepcopy(env)
                self.simulate_move(temp_env, curr_move, self.position)
                if temp_env.gameOver(curr_move, self.position):
                    return self.eval_func(temp_env.board), curr_move
                eval = self.minimax(valid_moves, curr_move, temp_env, depth-1, 2)[0]
                if eval > max_eval:
                    max_eval = eval
                    best_move = curr_move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for curr_move in valid_moves:
                temp_env = deepcopy(env)
                self.simulate_move(temp_env, curr_move, self.opponent.position)
                if temp_env.gameOver(curr_move, self.opponent.position):
                    return self.eval_func(temp_env.board), curr_move
                eval = self.minimax(valid_moves, curr_move, temp_env, depth-1, 1)[0]
                if eval < min_eval:
                    min_eval = eval
                    best_move = curr_move
            return min_eval, best_move
        
    def simulate_move(self, env: connect4, move: int, player: int):
        env.board[env.topPosition[move]][move] = player
        env.topPosition[move] -= 1
        env.history[0].append(move)
    
    def play(self, env: connect4, move: list) -> None:
        random.seed(self.seed)
        valid_moves = [i for i, p in enumerate(env.topPosition) if p >= 0]
        first_move = 4 # random.choice(valid_moves)
        i, best_move = self.minimax(valid_moves, first_move, deepcopy(env), 2, self.position)  
        move[:] = [best_move]
        print('play minimax', i, best_move)
    
    def eval_func(self, board):  
        
        weight_matrix = np.array([
            [1, 2, 3, 4, 3, 2, 1],
            [1, 2, 3, 4, 3, 2, 1],
            [1, 2, 3, 4, 3, 2, 1],
            [1, 2, 3, 4, 3, 2, 1],
            [1, 2, 3, 4, 3, 2, 1],
            [1, 2, 3, 4, 3, 2, 1]
        ])
        
        score = 0
        
        # Horizontal check
        for row in range(ROW_COUNT):
            row_array = [int(i) for i in list(board[row,:])]
            for col in range(COLUMN_COUNT - 3):
                window = row_array[col:col+4]
                score += self.evaluate(window) * np.sum(weight_matrix[row, col:col+4])

        # Vertical check
        for col in range(COLUMN_COUNT):
            col_array = [int(i) for i in list(board[:, col])]
            for row in range(ROW_COUNT - 3):
                window = col_array[row:row+4]
                score += self.evaluate(window) * np.sum(weight_matrix[row:row+4, col])
                
        # Positive diagonal check
        for row in range(ROW_COUNT - 3):
            for col in range(COLUMN_COUNT - 3):
                window = [board[row+i][col+i] for i in range(4)]
                score += self.evaluate(window) * sum(weight_matrix[row+i, col+i] for i in range(4))

        # Negative diagonal check
        for row in range(3, ROW_COUNT):
            for col in range(COLUMN_COUNT - 3):
                window = [board[row-i][col+i] for i in range(4)]
                score += self.evaluate(window) * sum(weight_matrix[row-i, col+i] for i in range(4))

        return score
        
    def evaluate(self, window):
        score = 0
        position = self.position
        opponent = self.opponent.position

        if window.count(position) == 4:
            score += 100
        elif window.count(position) == 3 and window.count(0) == 1:
            if position == 1:
                score += 10
        elif window.count(position) == 2 and window.count(0) == 2:
            score += 4
        elif window.count(position) == 1 and window.count(0) == 3:
            score += 1
        elif window.count(opponent) == 4:
            score -= 100
        elif window.count(opponent) == 3 and window.count(0) == 1:
            score -= 8
        elif window.count(opponent) == 2 and window.count(0) == 2:
            score -= 4
        elif window.count(opponent) == 1 and window.count(0) == 3:
            score -= 1
         
        return score
        
    
class alphaBetaAI(minimaxAI):
    
    def minimax(self, valid_moves, best_move, env, depth, position, alpha=float('-inf'), beta=float('inf')):            
        if depth == 0:
            return self.eval_func(env.board), None
        
        maximizingPlayer = True if position == 1 else False
        
        if maximizingPlayer:
            max_eval = float('-inf')
            for curr_move in valid_moves:
                temp_env = deepcopy(env)
                self.simulate_move(temp_env, curr_move, self.position)
                if temp_env.gameOver(curr_move, self.position):
                    return self.eval_func(temp_env.board), curr_move
                eval, _ = self.minimax(valid_moves, curr_move, temp_env, depth-1, 2, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = curr_move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    # print('prune')
                    break  # prune
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for curr_move in valid_moves:
                temp_env = deepcopy(env)
                self.simulate_move(temp_env, curr_move, self.opponent.position)  
                if temp_env.gameOver(curr_move, self.opponent.position):
                    return self.eval_func(temp_env.board), curr_move
                eval, _ = self.minimax(valid_moves, curr_move, temp_env, depth-1, 1, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = curr_move
                beta = min(beta, eval)
                if beta <= alpha:
                    # print('prune')
                    break  # prune
            return min_eval, best_move

    def play(self, env: connect4, move: list) -> None:
        random.seed(self.seed)
        valid_moves = [i for i, p in enumerate(env.topPosition) if p >= 0]
        first_move = 4 # random.choice(valid_moves)
        # using depth of 2, more consistent than 3 and 4 sometimes times out late game
        i, best_move = self.minimax(valid_moves, first_move, deepcopy(env), 2, self.position)  
        move[:] = [best_move]
        print('play alphabeta', i, best_move)
    

SQUARESIZE = 100
BLUE = (0,0,255)
BLACK = (0,0,0)
P1COLOR = (255,0,0)
P2COLOR = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)




