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
        self.gameDimensions_1: list[int] = [5, 4]
        self.numberOfGames_1: int = 2

        self.gameDimensions_2: list[int] = [6, 6]
        self.numberOfGames_2: int = 3

    def test_initBodyShape(self) -> None:
        self.snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

        self.assertTrue(self.snake_1.snakeBodyLocation.shape == (2, 5 * 4, 2))
        self.assertTrue(self.snake_2.snakeBodyLocation.shape == (3, 6 * 6, 2))

    def test_initFoodShape(self) -> None:
        self.snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

        self.assertTrue(self.snake_1.foodLocation.shape == (2, 2))
        self.assertTrue(self.snake_2.foodLocation.shape == (3, 2))

    def test_generateRandomLocation_coordinatesFromMask(self) -> None:
        self.snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

        self.snake_1.foodLocation[0] = self.snake_1.snakeBodyLocation[0, 0]
        self.snake_2.foodLocation[0] = self.snake_2.snakeBodyLocation[0, 0]
        self.snake_2.foodLocation[1] = self.snake_2.snakeBodyLocation[1, 0]

        snakeHitFoodMask_1: numpy.ndarray = numpy.array([True, False])
        snakeHitFoodMask_2: numpy.ndarray = numpy.array([True, True, False])

        self.snake_1.generateRandomLocations(snakeHitFood=snakeHitFoodMask_1)
        self.snake_2.generateRandomLocations(snakeHitFood=snakeHitFoodMask_2)

        self.assertTrue(
            equalNumpyArrays(
                self.snake_1.snakeBodyLocation[0, 0], self.snake_1.foodLocation[0]
            )
            == False
        )
        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.snakeBodyLocation[0, 0], self.snake_2.foodLocation[0]
            )
            == False
        )
        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.snakeBodyLocation[1, 0], self.snake_2.foodLocation[1]
            )
            == False
        )
    def test_generateRandomLocation_exceptions(self) -> None:
        self.snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

        with self.assertRaises(Exception):
            self.snake_1.generateRandomLocations(
                resetGame=True, snakeHitFood=numpy.array([True, False])
            )
        with self.assertRaises(Exception):
            self.snake_2.generateRandomLocations(
                resetGame=True, snakeHitFood=numpy.array([True, True, False])
            )

        with self.assertRaises(Exception):
            self.snake_1.generateRandomLocations()

        with self.assertRaises(Exception):
            self.snake_2.generateRandomLocations()
    
    def test_generateRandomLocation_coordinatesOnReset(self) -> None:
        self.snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

        makeFoodSameAsHead(self.snake_1)
        makeFoodSameAsHead(self.snake_2)

        makeBodySameAsHead(self.snake_1)
        makeBodySameAsHead(self.snake_2)

        self.snake_1.generateRandomLocations(resetGame=True)
        self.snake_2.generateRandomLocations(resetGame=True)

        for gameIndex in range(self.snake_1.numberOfGames):
            self.assertTrue(
                equalNumpyArrays(
                    self.snake_1.foodLocation[gameIndex],
                    self.snake_1.snakeBodyLocation[gameIndex, 0],
                )
                == False,
            )

        for gameIndex in range(self.snake_2.numberOfGames):
            self.assertTrue(
                equalNumpyArrays(
                    self.snake_2.foodLocation[gameIndex],
                    self.snake_2.snakeBodyLocation[gameIndex, 0],
                )
                == False,
            )

    def test_generateCoordinatesOnReset(self) -> None:
        self.snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

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
                    self.snake_1.snakeBodyLocation[gameIndex, 0],
                )
                == False,
            )

        for gameIndex in range(self.snake_2.numberOfGames):
            self.assertTrue(
                equalNumpyArrays(
                    self.snake_2.foodLocation[gameIndex],
                    self.snake_2.snakeBodyLocation[gameIndex, 0],
                )
                == False,
            )

    def test_generateCoordiantesFromMask(self) -> None:
        self.snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

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
            == False
        )
        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.snakeBodyLocation[0, 0], self.snake_2.foodLocation[0]
            )
            == False
        )
        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.snakeBodyLocation[1, 0], self.snake_2.foodLocation[1]
            )
            == False
        )

    def test_findSnakeHitSelf(self) -> None:
        self.snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        self.snake_1.snakeBodyLocation[0, 1] = self.snake_1.snakeBodyLocation[0, 0]
        self.snake_2.snakeBodyLocation[1, 1] = self.snake_2.snakeBodyLocation[1, 0]
        self.snake_2.snakeBodyLocation[2, 1] = self.snake_2.snakeBodyLocation[2, 0]

        snakeHitSelfMask_1: numpy.ndarray = self.snake_1.findSnakeHitSelf(
            self.snake_1.snakeBodyLocation[:, 0, :].reshape(self.snake_1.numberOfGames, 1, -1)
        )
        snakeHitSelfMask_2: numpy.ndarray = self.snake_2.findSnakeHitSelf(
            self.snake_2.snakeBodyLocation[:, 0, :].reshape(self.snake_2.numberOfGames, 1, -1)
        )

        self.assertTrue((snakeHitSelfMask_1 == numpy.array([True, False])).all(-1))
        self.assertTrue(
            (snakeHitSelfMask_2 == numpy.array([False, True, True])).all(-1)
        )

    def test_removeEndedGames(self) -> None:
        self.snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

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
        )

        self.snake_2.removeEndedGames(gameEnded_2)
        self.assertTrue(self.snake_2.snakeBodyLocation.shape == changeSnake_2.shape)
        self.assertTrue(
            (self.snake_2.snakeBodyLocation[:, 0] == changeSnake_2[:, 0])
            .all(-1)
            .any(-1),
        )

    def test_moveSnakeBody_hitWall(self) -> None:
        self.snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snakeMove_1: list[int] = [0, 1]
        snakeMove_2: list[int] = [2, 0, 3]

        self.snake_1.snakeBodyLocation[0, 0] = [1, 2]  # Hit wall up
        self.snake_1.snakeBodyLocation[1, 0] = [1, 2]  # Hit wall right

        self.snake_2.snakeBodyLocation[0, 0] = [4, 2]  # Hit wall down
        self.snake_2.snakeBodyLocation[1, 0] = [2, 1]
        self.snake_2.snakeBodyLocation[2, 0] = [3, 1]  # Hit wall left

        snake_1_gameEnded, _ = self.snake_1.moveSnakeBody(
            snakeMove_1
        )
        snake_2_gameEnded, _ = self.snake_2.moveSnakeBody(
            snakeMove_2
        )

        self.assertTrue(
            equalNumpyArrays(snake_1_gameEnded, numpy.array([True, True])),
            print(f"Snake 1 game end: {snake_1_gameEnded}"),
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_gameEnded, numpy.array([True, False, True])),
            print(f"Snake 2 game end: {snake_2_gameEnded}"),
        )

    def test_moveSnakeBody_hitFood(self) -> None:
        
        self.snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snakeMove_1: list[int] = [0, 1]
        snakeMove_2: list[int] = [2, 0, 3]

        self.snake_1.snakeBodyLocation[0, 0] = [2, 2]
        self.snake_1.snakeBodyLocation[1, 0] = [2, 1]
        self.snake_2.snakeBodyLocation[0, 0] = [1, 2]
        self.snake_2.snakeBodyLocation[1, 0] = [3, 2]
        self.snake_2.snakeBodyLocation[2, 0] = [2, 2]

        self.snake_1.foodLocation[:] = [[1, 2], [2, 1]]
        self.snake_2.foodLocation[:] = [[1, 2], [2, 2], [2, 1]]

        _, snake_1_snakeHitFood = self.snake_1.moveSnakeBody(
            snakeMove_1
        )
        _, snake_2_snakeHitFood = self.snake_2.moveSnakeBody(
            snakeMove_2
        )

        self.assertTrue(
            equalNumpyArrays(snake_1_snakeHitFood, numpy.array([True, False])),
            print(f"Snake 1 HitFood: {snake_1_snakeHitFood}"),
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_snakeHitFood, numpy.array([False, True, True])),
            print(f"Snake 2 HitFood: {snake_2_snakeHitFood}"),
        )

    def test_moveSnakeBody_hitSelf(self) -> None:
        self.snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snakeMove_1: list[int] = [0, 0]
        snakeMove_2: list[int] = [3, 2, 0]

        self.snake_1.snakeBodyLocation[0, 0] = [2, 2]
        self.snake_1.snakeBodyLocation[1, 0] = [2, 2]

        self.snake_2.snakeBodyLocation[0, 0] = [3, 1]
        self.snake_2.snakeBodyLocation[1, 0] = [3, 2]
        self.snake_2.snakeBodyLocation[2, 0] = [3, 2]

        self.snake_1.snakeBodyLocation[0, 1] = self.snake_1.snakeBodyLocation[0, 0]
        self.snake_2.snakeBodyLocation[0, 1] = self.snake_2.snakeBodyLocation[0, 0]
        self.snake_2.snakeBodyLocation[1, 1] = self.snake_2.snakeBodyLocation[1, 0]

        self.snake_1.snakeBodyLocation[0, 0] = [3, 2]
        self.snake_2.snakeBodyLocation[0, 0] = [3, 2]
        self.snake_2.snakeBodyLocation[1, 0] = [2, 2]

        snake_1_gameEnded, _ = self.snake_1.moveSnakeBody(snakeMove_1)
        snake_2_gameEnded, _ = self.snake_2.moveSnakeBody(snakeMove_2)

        self.assertTrue(
            equalNumpyArrays(snake_1_gameEnded, numpy.array([True, False])),
            print(f"Snake 1: {self.snake_1.snakeBodyLocation}"),
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_gameEnded, numpy.array([True, True, False])),
            print(f"Snake 2: {self.snake_2.snakeBodyLocation}"),
        )


if __name__ == "__main__":
    unittest.main()
