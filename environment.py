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
        self.drawInitialSnake()
        self.updateFoodLocation()

    def drawInitialSnake(self) -> None:
        indicies: numpy.ndarray = numpy.arange(self.stateSpace.shape[0])
        self.stateSpace[
            indicies,
            self.snake.snakeBodyLocation[indicies, 0, 0],
            self.snake.snakeBodyLocation[indicies, 0, 1],
        ] = 2

    def update(self, moves: list[int]) -> tuple[numpy.ndarray, numpy.ndarray, bool]:
        nextSnakePositions: numpy.ndarray = self.snake.generateNextSnakePosition(moves)
        gameEndMask, snakeHitFoodMask = self.snake.generateMasks(nextSnakePositions)

        self.removeFromStateSpace(snakeHitFoodMask)

        self.snake.updateSnakeBodyCoordinates(nextSnakePositions, snakeHitFoodMask)

        if snakeHitFoodMask.any():
            self.snake.generateCoordinatesFromMask(snakeHitFoodMask)
            self.updateFoodLocation(snakeHitFoodMask)
        self.updateStateSpace()

        if gameEndMask.all():
            return gameEndMask, snakeHitFoodMask, True

        self.removeEndedGames(gameEndMask)
        return gameEndMask, snakeHitFoodMask, False

    def updateFoodLocation(self, snakeHitFoodMask: numpy.ndarray | None = None) -> None:
        if type(snakeHitFoodMask) == numpy.ndarray:
            updateFoodIndicies: numpy.ndarray = numpy.where(snakeHitFoodMask)[0]
        else:
            updateFoodIndicies: numpy.ndarray = numpy.arange(self.stateSpace.shape[0])
        self.stateSpace[
            updateFoodIndicies,
            self.snake.foodLocation[updateFoodIndicies, 0],
            self.snake.foodLocation[updateFoodIndicies, 1],
        ] = 3

    def removeFromStateSpace(self, snakeHitFoodMask: numpy.ndarray) -> None:
        snakeHitNothingMask: numpy.ndarray = ~snakeHitFoodMask
        indiciesOfHitNothingMask: numpy.ndarray = numpy.where(snakeHitNothingMask)[0]
        maskedOldSnakeEndIndicies: numpy.ndarray = self.snake.currentBodyEndIndex[
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
