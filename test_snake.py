import unittest

import numpy

from snake import Snake


def makeFoodSameAsHead(snake: Snake):
    for gameIndex in range(snake.numberOfGames):
        snake.foodLocation[gameIndex] = snake.snakeBodyLocation[gameIndex, 0]


def makeBodySameAsHead(snake: Snake):
    for gameIndex in range(snake.numberOfGames):
        snake.snakeBodyLocation[gameIndex, 1] = snake.snakeBodyLocation[gameIndex, 0]


def equalNumpyArrays(array_1: numpy.ndarray, array_2: numpy.ndarray) -> bool:
    equalInPlace: numpy.ndarray = array_1 == array_2
    equalInRow: numpy.ndarray = equalInPlace.all(-1)
    equal: bool = equalInRow.any(-1)
    return equal


class TestSnake(unittest.TestCase):
    def setUp(self) -> None:
        gameDimensions_1: list[int] = [5, 4]
        numberOfGames_1: int = 2

        gameDimensions_2: list[int] = [6, 6]
        numberOfGames_2: int = 3

        self.snake_1 = Snake(gameDimensions_1, numberOfGames_1)
        self.snake_2 = Snake(gameDimensions_2, numberOfGames_2)

    def test_initBodyShape(self) -> None:
        self.assertTrue(self.snake_1.snakeBodyLocation.shape == (2, 5 * 4, 2))
        self.assertTrue(self.snake_2.snakeBodyLocation.shape == (3, 6 * 6, 2))

    def test_initFoodShape(self) -> None:
        self.assertTrue(self.snake_1.foodLocation.shape == (2, 2))
        self.assertTrue(self.snake_2.foodLocation.shape == (3, 2))

    # TODO: test moveSnakeBody
    # TODO: test generateRandomLocation
    def test_generateCoordinatesOnReset(self) -> None:
        makeFoodSameAsHead(self.snake_1)
        makeFoodSameAsHead(self.snake_2)

        makeBodySameAsHead(self.snake_1)
        makeBodySameAsHead(self.snake_2)

        self.snake_1.generateCoordinatesOnReset()
        self.snake_2.generateCoordinatesOnReset()

        for gameIndex in range(self.snake_1.numberOfGames):
            self.assertTrue(
                equalNumpyArrays(
                    self.snake_1.foodLocation[gameIndex],
                    self.snake_1.snakeBodyLocation[gameIndex, :],
                )
                == False,
            )

        for gameIndex in range(self.snake_2.numberOfGames):
            self.assertTrue(
                equalNumpyArrays(
                    self.snake_2.foodLocation[gameIndex],
                    self.snake_2.snakeBodyLocation[gameIndex, :],
                )
                == False,
            )

    def test_generateCoordiantesFromMask(self) -> None:
        self.snake_1.foodLocation[0] = self.snake_1.snakeBodyLocation[0, 0]
        self.snake_2.foodLocation[0] = self.snake_2.snakeBodyLocation[0, 0]
        self.snake_2.foodLocation[1] = self.snake_2.snakeBodyLocation[1, 0]

        snakeHitFoodMask_1: numpy.ndarray = numpy.array([True, False])
        snakeHitFoodMask_2: numpy.ndarray = numpy.array([True, True, False])

        self.snake_1.generateCoordinatesFromMask(snakeHitFoodMask_1)
        self.snake_2.generateCoordinatesFromMask(snakeHitFoodMask_2)

        self.assertTrue(
            equalNumpyArrays(
                self.snake_1.snakeBodyLocation[0, 0], self.snake_1.foodLocation[0]
            )
        )
        self.assertTrue(
            equalNumpyArrays(
                self.snake_1.snakeBodyLocation[1, 0], self.snake_1.foodLocation[1]
            )
            == False
        )

        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.snakeBodyLocation[0, 0], self.snake_2.foodLocation[0]
            )
        )
        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.snakeBodyLocation[1, 0], self.snake_2.foodLocation[1]
            )
        )
        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.snakeBodyLocation[2, 0], self.snake_2.foodLocation[2]
            )
            == False
        )

    def test_findSnakeHitSelf(self) -> None:
        self.snake_1.snakeBodyLocation[0, 1] = self.snake_1.snakeBodyLocation[0, 0]
        self.snake_2.snakeBodyLocation[1, 1] = self.snake_2.snakeBodyLocation[1, 0]
        self.snake_2.snakeBodyLocation[2, 1] = self.snake_2.snakeBodyLocation[2, 0]

        snakeHitSelfMask_1: numpy.ndarray = self.snake_1.findSnakeHitSelf()
        snakeHitSelfMask_2: numpy.ndarray = self.snake_2.findSnakeHitSelf()

        self.assertTrue((snakeHitSelfMask_1 == numpy.array([True, False])).all(-1))
        self.assertTrue(
            (snakeHitSelfMask_2 == numpy.array([False, True, True])).all(-1)
        )

    def test_removeEndedGames(self) -> None:
        gameEnded_1: numpy.ndarray = numpy.array([True, False], dtype=bool)
        gameEnded_2: numpy.ndarray = numpy.array([False, False, True])

        changeSnake_1: numpy.ndarray = (
            self.snake_1.snakeBodyLocation[1].copy().reshape(1, 20, 2)
        )
        changeSnake_2: numpy.ndarray = (
            self.snake_2.snakeBodyLocation[:2].copy().reshape(2, 36, 2)
        )

        self.snake_1.removeEndedGames(gameEnded_1)
        self.assertTrue(self.snake_1.snakeBodyLocation.shape == changeSnake_1.shape)
        self.assertTrue(
            (self.snake_1.snakeBodyLocation[:, 0] == changeSnake_1[:, 0])
            .all(-1)
            .any(-1),
            print(
                f"Snake 1 head: {self.snake_1.snakeBodyLocation[:, 0]}\nChanged Snake 1 head: {changeSnake_1[:, 0]}"
            ),
        )

        self.snake_2.removeEndedGames(gameEnded_2)
        self.assertTrue(self.snake_2.snakeBodyLocation.shape == changeSnake_2.shape)
        self.assertTrue(
            (self.snake_2.snakeBodyLocation[:, 0] == changeSnake_2[:, 0])
            .all(-1)
            .any(-1),
            print(
                f"Snake 2 head: {self.snake_2.snakeBodyLocation[:, 0]}\nChanged Snake 2 head: {changeSnake_2[:, 0]}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
