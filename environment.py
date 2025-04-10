import numpy
from snake import Snake


class Environment:
    def __init__(self, gameDimensions: list[int], numberOfGames: int) -> None:
        self.gameDimensions: list[int] = gameDimensions
        self.numberOfGames: int = numberOfGames
        self.snake: Snake = Snake(self.gameDimensions, self.numberOfGames)
        self.reset()

    def reset(self) -> None:
        self.snake.resetGameState()
        self.stateSpace: numpy.ndarray = numpy.ones(
            (self.numberOfGames, self.gameDimensions[1], self.gameDimensions[0]),
            dtype=int,
        )
        self.stateSpace[:, 1:-1, 1:-1] = 0

    def update(self, moves: list[int]) -> tuple[numpy.ndarray, numpy.ndarray, bool]:
        oldSnakeEndIndicies: numpy.ndarray = self.snake.currentBodyEndIndex.copy()
        gameEndMask, snakeHitFoodMask = self.snake.moveSnakeBody(moves)
        self.removeFromStateSpace(oldSnakeEndIndicies, snakeHitFoodMask)
        self.updateStateSpace()

        if gameEndMask.all():
            return gameEndMask, snakeHitFoodMask, True

        self.removeEndedGames(gameEndMask)
        return gameEndMask, snakeHitFoodMask, False

    def removeFromStateSpace(
        self, oldSnakeEndIndicies: numpy.ndarray, snakeHitFoodMask
    ) -> None:
        snakeHitNothingMask: numpy.ndarray = ~snakeHitFoodMask
        indiciesOfHitNothingMask: numpy.ndarray = numpy.where(snakeHitNothingMask)[0]
        maskedOldSnakeEndIndicies: numpy.ndarray = oldSnakeEndIndicies[
            indiciesOfHitNothingMask
        ]

        self.stateSpace[
            indiciesOfHitNothingMask,
            self.snake.snakeBodyLocation[
                indiciesOfHitNothingMask, maskedOldSnakeEndIndicies, 0
            ],
            self.snake.snakeBodyLocation[
                indiciesOfHitNothingMask, maskedOldSnakeEndIndicies, 1
            ],
        ] = 0

    def updateStateSpace(self) -> None:
        indiciesForSnake: numpy.ndarray = numpy.arange(self.stateSpace.shape[0])
        self.stateSpace[
            indiciesForSnake,
            self.snake.snakeBodyLocation[indiciesForSnake, 0, 0],
            self.snake.snakeBodyLocation[indiciesForSnake, 0, 1],
        ] = 2

    def removeEndedGames(self, gameEndMask: numpy.ndarray) -> None:
        self.stateSpace = self.stateSpace[~gameEndMask]
        self.snake.snakeBodyLocation = self.snake.snakeBodyLocation[~gameEndMask]
        self.snake.foodLocation = self.snake.foodLocation[~gameEndMask]
        self.snake.currentBodyEndIndex = self.snake.currentBodyEndIndex[~gameEndMask]
