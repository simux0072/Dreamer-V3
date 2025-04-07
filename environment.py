import numpy
from snake import Snake


class Environment:
    def __init__(self, gameDimensions: list[int], numberOfGames: int) -> None:
        self.gameDimensions: list[int] = gameDimensions
        self.numberOfGames: int = numberOfGames
        self.snake: Snake = Snake(self.gameDimensions, self.numberOfGames)

    def reset(self) -> None:
        self.snake.resetGameState()
        self.stateSpace: numpy.ndarray = numpy.ones(
            (self.numberOfGames, self.gameDimensions[1], self.gameDimensions[0]),
            dtype=int,
        )
        self.stateSpace[
            :, 1 : (self.gameDimensions[1] - 2), 1 : (self.gameDimensions[0] - 2)
        ] = 0

    def update(self, moves: list[int]) -> None:
        coordinatesToErase: numpy.ndarray = self.snake.snakeBodyLocation[
            numpy.arange(self.snake.snakeBodyLocation.shape[0]),
            self.snake.currentBodyEndIndex[:],
        ].copy()

        gameEndMask, snakeHitFood = self.snake.moveSnakeBody(moves)

    def updateStateSpace(
        self,
        coordinatesToErase: numpy.ndarray,
        snakeHitFoodMask: numpy.ndarray,
        gameEndMask: numpy.ndarray,
    ) -> numpy.ndarray:

        snakeHitNothingMask: numpy.ndarray = ~numpy.bitwise_or(
            gameEndMask, snakeHitFoodMask
        )

        maskedIndicies: numpy.ndarray = numpy.where(snakeHitNothingMask)[0]
        maskedCoordinates: numpy.ndarray = coordinatesToErase[maskedIndicies]

        self.stateSpace[
            maskedIndicies, maskedCoordinates[:, 0], maskedCoordinates[:, 1]
        ] = 0

        return numpy.array((0))
