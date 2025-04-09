import numpy


# Up, Right, Down, Left
DIRECTIONS = numpy.array(
    [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
    ],
    dtype=numpy.int8,
)


def generateAllPossibleCoordinates(dimensions: list[int]):
    xCoordinates: numpy.ndarray = numpy.linspace(
        1, dimensions[0] - 2, dimensions[0] - 2
    )
    yCoordinates: numpy.ndarray = numpy.linspace(
        1, dimensions[1] - 2, dimensions[1] - 2
    )

    possibleCoordinates: numpy.ndarray = numpy.array(
        numpy.meshgrid(yCoordinates, xCoordinates)
    ).T.reshape(-1, 2)

    return possibleCoordinates


class Snake:
    def __init__(self, gameDimensions: list[int], numberOfGames: int) -> None:
        self.gameDimensions: list[int] = gameDimensions
        self.numberOfGames: int = numberOfGames
        self.possibleCoordinates: numpy.ndarray = generateAllPossibleCoordinates(
            self.gameDimensions
        )

        self.resetGameState()

    def resetGameState(self) -> None:
        self.snakeBodyLocation: numpy.ndarray = numpy.zeros(
            (
                self.numberOfGames,
                self.gameDimensions[1] * self.gameDimensions[0],
                2,
            ),
            dtype=numpy.int16,
        )

        self.foodLocation: numpy.ndarray = numpy.zeros(
            (
                self.numberOfGames,
                2,
            ),
            dtype=numpy.int16,
        )

        self.currentBodyEndIndex = numpy.zeros((self.numberOfGames), dtype=int)

        self.generateRandomLocations(resetGame=True)

    def generateRandomLocations(
        self, resetGame: bool = False, snakeHitFood: numpy.ndarray | None = None
    ) -> None:
        if resetGame:
            if snakeHitFood is not None:
                raise Exception(
                    f"Trying to use resetGame logic for random location generation, when snakeHitFood is of Type: {type(snakeHitFood)}!"
                )

            self.generateCoordinatesOnReset()

        else:
            if snakeHitFood is None:
                raise Exception(
                    f"Tried to generate coordinates when resetGame is {resetGame}, but snakeHitFood is of Type: {type(snakeHitFood)}!"
                )

            self.generateCoordinatesFromMask(snakeHitFood)

    def generateCoordinatesFromMask(self, snakeHitFood: numpy.ndarray):
        for gameIndex in numpy.where(snakeHitFood)[0]:
            freeCoordinateMask: numpy.ndarray = (
                ~(
                    self.possibleCoordinates[:, None]
                    == self.snakeBodyLocation[gameIndex, :]
                )
                .all(-1)
                .any(-1)
            )

            freeCoordinateIndicies: numpy.ndarray = numpy.where(freeCoordinateMask)[0]
            selectedCoordiante: numpy.ndarray = numpy.random.choice(
                freeCoordinateIndicies
            )
            self.foodLocation[gameIndex] = self.possibleCoordinates[selectedCoordiante]

    def generateCoordinatesOnReset(self):
        for gameIndex in range(self.numberOfGames):
            chosenCoordinates: numpy.ndarray = numpy.random.choice(
                (self.gameDimensions[0] - 2) * (self.gameDimensions[1] - 2),
                size=2,
                replace=False,
            )
            self.foodLocation[gameIndex] = self.possibleCoordinates[
                chosenCoordinates[0]
            ]
            self.snakeBodyLocation[gameIndex, 0] = self.possibleCoordinates[
                chosenCoordinates[1]
            ]

    def findSnakeHitSelf(self, nextSnakePosition: numpy.ndarray) -> numpy.ndarray:
        firstElementExpanded: numpy.ndarray = nextSnakePosition[:, 0].reshape(
            self.snakeBodyLocation.shape[0], 1, self.snakeBodyLocation.shape[2]
        )
        bodyComparison: numpy.ndarray = (
            firstElementExpanded == self.snakeBodyLocation[:, 1:]
        )
        snakeHitSelf: numpy.ndarray = bodyComparison.all(-1).any(-1)
        return snakeHitSelf

    def removeEndedGames(self, gameEnded: numpy.ndarray) -> None:
        self.snakeBodyLocation = self.snakeBodyLocation[~gameEnded]
        self.foodLocation = self.foodLocation[~gameEnded]
        self.currentBodyEndIndex = self.currentBodyEndIndex[~gameEnded]

    def moveSnakeBody(
        self, moveDirection: list[int]
    ) -> tuple[numpy.ndarray, numpy.ndarray]:

        nextSnakePosition: numpy.ndarray = (
            self.snakeBodyLocation[:, 0] + DIRECTIONS[moveDirection]
        ).reshape(self.numberOfGames, 1, 2)

        gameEndMask: numpy.ndarray = self.generateGameEndMask(nextSnakePosition)
        snakeHitFoodMask: numpy.ndarray = (
            nextSnakePosition[:, 0] == self.foodLocation[:]
        ).all(-1)

        self.updateSnakeBodyCoordinates(nextSnakePosition, snakeHitFoodMask)

        return gameEndMask, snakeHitFoodMask

    def updateSnakeBodyCoordinates(
        self,
        nextSnakePosition: numpy.ndarray,
        snakeHitFoodMask: numpy.ndarray,
    ) -> None:

        hitNothingMaskedIndicies: numpy.ndarray = numpy.where(~snakeHitFoodMask)[0]
        maskedCurrentBodyEndIndicies: numpy.ndarray = self.currentBodyEndIndex[
            ~snakeHitFoodMask
        ]

        self.snakeBodyLocation[
            hitNothingMaskedIndicies, maskedCurrentBodyEndIndicies
        ] = [0, 0]

        self.currentBodyEndIndex += snakeHitFoodMask
        self.snakeBodyLocation = numpy.hstack(
            (nextSnakePosition, self.snakeBodyLocation[:, :-1, :])
        )

    def generateGameEndMask(self, nextSnakePosition: numpy.ndarray) -> numpy.ndarray:
        updatedGameDimensions: list[int] = [
            self.gameDimensions[0] - 1,
            self.gameDimensions[1] - 1,
        ]
        snakeHitWallEquality: numpy.ndarray = (
            nextSnakePosition[:, 0] % updatedGameDimensions == 0
        )

        snakeHitWallMask: numpy.ndarray = snakeHitWallEquality.any(-1)

        snakeHitSelfMask: numpy.ndarray = self.findSnakeHitSelf(nextSnakePosition)

        gameEndMask: numpy.ndarray = numpy.bitwise_or(
            snakeHitWallMask, snakeHitSelfMask
        )

        return gameEndMask
