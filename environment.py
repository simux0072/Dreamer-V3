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
        gameEndMask, snakeHitFoodMask, snakeHitNothingMask = self.snake.moveSnakeBody(
            moves
        )
        # TODO: check if all games have ended
        # TODO: update the gameState element
        # TODO: remove the last body part for evey snakeHitNOthingMask element.
        # TODO: remove ended games

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
