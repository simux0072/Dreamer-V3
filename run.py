import numpy
import numba
from environment import Environment

game_dims: list[int] = [10, 10]
num_games: int = 1

environment = Environment(game_dims, num_games)

while True:
    move: list[int] = [int(input("Your move: "))]
    gameEndMask, snakeHitFoodMask, gameEnd = environment.update(move)
    if gameEnd:
        break
    print(gameEndMask)
    print(snakeHitFoodMask)
    print(environment.stateSpace)
