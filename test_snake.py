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
    equal: bool = bool(equalInPlace.all())
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
            self.snake_1.snakeBodyLocation[:, 0, :].reshape(
                self.snake_1.numberOfGames, 1, -1
            )
        )
        snakeHitSelfMask_2: numpy.ndarray = self.snake_2.findSnakeHitSelf(
            self.snake_2.snakeBodyLocation[:, 0, :].reshape(
                self.snake_2.numberOfGames, 1, -1
            )
        )

        self.assertTrue(
            equalNumpyArrays(snakeHitSelfMask_1, numpy.array([True, False])),
            f"Snake 1 snakeHitSelfMask_1: {snakeHitSelfMask_1}",
        )
        self.assertTrue(
            equalNumpyArrays(snakeHitSelfMask_2, numpy.array([False, True, True])),
            f"Snake 2 snakeHitSelfMask_2: {snakeHitSelfMask_2}",
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

        snake_1_gameEnded, _ = self.snake_1.moveSnakeBody(snakeMove_1)
        snake_2_gameEnded, _ = self.snake_2.moveSnakeBody(snakeMove_2)

        self.assertTrue(
            equalNumpyArrays(snake_1_gameEnded, numpy.array([True, True])),
            f"Snake 1 game end: {snake_1_gameEnded}",
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_gameEnded, numpy.array([True, False, True])),
            f"Snake 2 game end: {snake_2_gameEnded}",
        )

        snake_1_snakeBodyLocation_shape: numpy.ndarray = numpy.array(
            [2, self.snake_1.gameDimensions[0] * self.snake_1.gameDimensions[1], 2]
        )
        snake_2_snakeBodyLocation_shape: numpy.ndarray = numpy.array(
            [3, self.snake_2.gameDimensions[0] * self.snake_2.gameDimensions[1], 2]
        )

        self.assertTrue(
            equalNumpyArrays(
                self.snake_1.snakeBodyLocation.shape, snake_1_snakeBodyLocation_shape
            ),
            f"Snake 1 snakeBodyLocation shape: {self.snake_1.snakeBodyLocation.shape}",
        )

        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.snakeBodyLocation.shape, snake_2_snakeBodyLocation_shape
            ),
            f"Snake 2 snakeBodyLocation shape: {self.snake_2.snakeBodyLocation.shape}",
        )

        self.assertTrue(
            self.snake_1.currentBodyEndIndex.shape == (2,),
            f"Snake 1 currentBodyEndIndex shape: {self.snake_1.currentBodyEndIndex.shape}",
        )
        self.assertTrue(
            self.snake_2.currentBodyEndIndex.shape == (3,),
            f"Snake 2 currentBodyEndIndex shape: {self.snake_2.currentBodyEndIndex.shape}",
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

        _, snake_1_snakeHitFood = self.snake_1.moveSnakeBody(snakeMove_1)
        _, snake_2_snakeHitFood = self.snake_2.moveSnakeBody(snakeMove_2)

        self.assertTrue(
            equalNumpyArrays(snake_1_snakeHitFood, numpy.array([True, False])),
            f"Snake 1 HitFood: {snake_1_snakeHitFood}",
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_snakeHitFood, numpy.array([False, True, True])),
            f"Snake 2 HitFood: {snake_2_snakeHitFood}",
        )

        snake_1_snakeBodyLocation_shape: numpy.ndarray = numpy.array(
            [2, self.snake_1.gameDimensions[0] * self.snake_1.gameDimensions[1], 2]
        )

        snake_2_snakeBodyLocation_shape: numpy.ndarray = numpy.array(
            [3, self.snake_2.gameDimensions[0] * self.snake_2.gameDimensions[1], 2]
        )

        self.assertTrue(
            equalNumpyArrays(
                self.snake_1.snakeBodyLocation.shape, snake_1_snakeBodyLocation_shape
            ),
            f"Snake 1 snakeBodyLocation shape: {self.snake_1.snakeBodyLocation.shape}",
        )

        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.snakeBodyLocation.shape, snake_2_snakeBodyLocation_shape
            ),
            f"Snake 2 snakeBodyLocation shape: {self.snake_2.snakeBodyLocation.shape}",
        )

        snake_1_foodLocation_shape: numpy.ndarray = numpy.array([2, 2])
        snake_2_foodLocation_shape: numpy.ndarray = numpy.array([3, 2])

        self.assertTrue(
            equalNumpyArrays(
                self.snake_1.foodLocation.shape, snake_1_foodLocation_shape
            ),
            f"Snake 1 foodLocation shape: {self.snake_1.foodLocation.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.foodLocation.shape, snake_2_foodLocation_shape
            ),
            f"Snake 2 foodLocation shape: {self.snake_2.foodLocation.shape}",
        )

        self.assertTrue(
            self.snake_1.currentBodyEndIndex.shape == (2,),
            f"Snake 1 currentBodyEndIndex shape: {self.snake_1.currentBodyEndIndex.shape}",
        )
        self.assertTrue(
            self.snake_2.currentBodyEndIndex.shape == (3,),
            f"Snake 2 currentBodyEndIndex shape: {self.snake_2.currentBodyEndIndex.shape}",
        )

    def test_moveSnakeBody_hitSelf(
        self,
    ) -> None:
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
            f"Snake 1 gameEnded: {snake_1_gameEnded}",
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_gameEnded, numpy.array([True, True, False])),
            f"Snake 2 gameEnded: {snake_2_gameEnded}",
        )

        snake_1_snakeBodyLocation_shape: numpy.ndarray = numpy.array(
            [2, self.snake_1.gameDimensions[0] * self.snake_1.gameDimensions[1], 2]
        )

        snake_2_snakeBodyLocation_shape: numpy.ndarray = numpy.array(
            [3, self.snake_2.gameDimensions[0] * self.snake_2.gameDimensions[1], 2]
        )

        self.assertTrue(
            equalNumpyArrays(
                self.snake_1.snakeBodyLocation.shape, snake_1_snakeBodyLocation_shape
            ),
            f"Snake 1 snakeBodyLocation shape: {self.snake_1.snakeBodyLocation.shape}",
        )

        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.snakeBodyLocation.shape, snake_2_snakeBodyLocation_shape
            ),
            f"Snake 2 snakeBodyLocation shape: {self.snake_2.snakeBodyLocation.shape}",
        )

        snake_1_foodLocation_shape: numpy.ndarray = numpy.array([2, 2])
        snake_2_foodLocation_shape: numpy.ndarray = numpy.array([3, 2])

        self.assertTrue(
            equalNumpyArrays(
                self.snake_1.foodLocation.shape, snake_1_foodLocation_shape
            ),
            f"Snake 1 foodLocation shape: {self.snake_1.foodLocation.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.foodLocation.shape, snake_2_foodLocation_shape
            ),
            f"Snake 2 foodLocation shape: {self.snake_2.foodLocation.shape}",
        )

        self.assertTrue(
            self.snake_1.currentBodyEndIndex.shape == (2,),
            f"Snake 1 currentBodyEndIndex shape: {self.snake_1.currentBodyEndIndex.shape}",
        )
        self.assertTrue(
            self.snake_2.currentBodyEndIndex.shape == (3,),
            f"Snake 2 currentBodyEndIndex shape: {self.snake_2.currentBodyEndIndex.shape}",
        )

    def test_moveSnakeBody_hitMix(self) -> None:

        self.snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snakeMove_1: list[int] = [0, 0]
        snakeMove_2: list[int] = [3, 2, 1]

        self.snake_1.snakeBodyLocation[0, 0] = [2, 2]
        self.snake_1.snakeBodyLocation[1, 0] = [1, 2]

        self.snake_2.snakeBodyLocation[0, 0] = [2, 2]
        self.snake_2.snakeBodyLocation[1, 0] = [3, 2]
        self.snake_2.snakeBodyLocation[2, 0] = [3, 4]

        self.snake_1.snakeBodyLocation[0, 1] = self.snake_1.snakeBodyLocation[0, 0]
        self.snake_2.snakeBodyLocation[1, 1] = self.snake_2.snakeBodyLocation[1, 0]

        self.snake_1.snakeBodyLocation[0, 0] = [3, 2]
        self.snake_2.snakeBodyLocation[1, 0] = [2, 2]

        self.snake_1.foodLocation[:] = [0, 0]
        self.snake_2.foodLocation[:] = [0, 0]
        self.snake_2.foodLocation[0] = [2, 1]

        snake_1_gameEnded, snake_1_snakeHitFood = self.snake_1.moveSnakeBody(
            snakeMove_1
        )
        snake_2_gameEnded, snake_2_snakeHitFood = self.snake_2.moveSnakeBody(
            snakeMove_2
        )

        self.assertTrue(
            equalNumpyArrays(snake_1_gameEnded, numpy.array([True, True])),
            f"Snake 1 gameEnded: {snake_1_gameEnded}",
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_gameEnded, numpy.array([False, True, True])),
            f"Snake 2 gameEnded: { snake_2_gameEnded }",
        )

        self.assertTrue(
            equalNumpyArrays(snake_1_snakeHitFood, numpy.array([False, False])),
            f"Snake 1 HitFood: {snake_1_snakeHitFood}",
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_snakeHitFood, numpy.array([True, False, False])),
            f"Snake 2 HitFood: {snake_2_snakeHitFood}",
        )

        snake_1_snakeBodyLocation_shape: numpy.ndarray = numpy.array(
            [2, self.snake_1.gameDimensions[0] * self.snake_1.gameDimensions[1], 2]
        )

        snake_2_snakeBodyLocation_shape: numpy.ndarray = numpy.array(
            [3, self.snake_2.gameDimensions[0] * self.snake_2.gameDimensions[1], 2]
        )

        self.assertTrue(
            equalNumpyArrays(
                self.snake_1.snakeBodyLocation.shape, snake_1_snakeBodyLocation_shape
            ),
            f"Snake 1 snakeBodyLocation shape: {self.snake_1.snakeBodyLocation.shape}",
        )

        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.snakeBodyLocation.shape, snake_2_snakeBodyLocation_shape
            ),
            f"Snake 2 snakeBodyLocation shape: {self.snake_2.snakeBodyLocation.shape}",
        )

        snake_1_foodLocation_shape: numpy.ndarray = numpy.array([2, 2])
        snake_2_foodLocation_shape: numpy.ndarray = numpy.array([3, 2])

        self.assertTrue(
            equalNumpyArrays(
                self.snake_1.foodLocation.shape, snake_1_foodLocation_shape
            ),
            f"Snake 1 foodLocation shape: {self.snake_1.foodLocation.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.foodLocation.shape, snake_2_foodLocation_shape
            ),
            f"Snake 2 foodLocation shape: {self.snake_2.foodLocation.shape}",
        )

        self.assertTrue(
            self.snake_1.currentBodyEndIndex.shape == (2,),
            f"Snake 1 currentBodyEndIndex shape: {self.snake_1.currentBodyEndIndex.shape}",
        )
        self.assertTrue(
            self.snake_2.currentBodyEndIndex.shape == (3,),
            f"Snake 2 currentBodyEndIndex shape: {self.snake_2.currentBodyEndIndex.shape}",
        )

    def test_generateGameEndMask_hitWall(self) -> None:
        self.snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1_nextSnakePosition: numpy.ndarray = numpy.array([[[1, 0]], [[1, 1]]])
        snake_2_nextSnakePosition: numpy.ndarray = numpy.array(
            [[[5, 2]], [[1, 1]], [[2, 5]]]
        )

        snake_1_gameEndMask: numpy.ndarray = self.snake_1.generateGameEndMask(
            snake_1_nextSnakePosition
        )
        snake_2_gameEndMask: numpy.ndarray = self.snake_2.generateGameEndMask(
            snake_2_nextSnakePosition
        )

        self.assertTrue(
            equalNumpyArrays(snake_1_gameEndMask, numpy.array([True, False])),
            f"Snake 1 game end: {snake_1_gameEndMask}",
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_gameEndMask, numpy.array([True, False, True])),
            f"Snake 2 game end: {snake_2_gameEndMask}",
        )

    def test_generateGameEndMask_hitSelf(self) -> None:

        self.snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1_nextSnakePosition: numpy.ndarray = numpy.array([[[1, 1]], [[1, 1]]])
        snake_2_nextSnakePosition: numpy.ndarray = numpy.array(
            [[[4, 2]], [[1, 1]], [[2, 4]]]
        )
        self.snake_1.snakeBodyLocation[0, 1] = [3, 2]
        self.snake_1.snakeBodyLocation[1, 1] = [1, 1]

        self.snake_2.snakeBodyLocation[0, 1] = [3, 2]
        self.snake_2.snakeBodyLocation[1, 1] = [1, 1]
        self.snake_2.snakeBodyLocation[2, 1] = [2, 3]

        snake_1_gameEndMask: numpy.ndarray = self.snake_1.generateGameEndMask(
            snake_1_nextSnakePosition
        )
        snake_2_gameEndMask: numpy.ndarray = self.snake_2.generateGameEndMask(
            snake_2_nextSnakePosition
        )

        self.assertTrue(
            equalNumpyArrays(snake_1_gameEndMask, numpy.array([False, True])),
            f"Snake 1 game end: {snake_1_gameEndMask}",
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_gameEndMask, numpy.array([False, True, False])),
            f"Snake 2 game end: {snake_2_gameEndMask}",
        )

    def test_generateGameEndMask_hitMix(self) -> None:

        self.snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1_nextSnakePosition: numpy.ndarray = numpy.array([[[1, 2]], [[1, 0]]])
        snake_2_nextSnakePosition: numpy.ndarray = numpy.array(
            [[[4, 2]], [[1, 1]], [[2, 5]]]
        )
        self.snake_1.snakeBodyLocation[0, 1] = [1, 2]
        self.snake_1.snakeBodyLocation[1, 1] = [1, 1]

        self.snake_2.snakeBodyLocation[0, 1] = [3, 2]
        self.snake_2.snakeBodyLocation[1, 1] = [1, 1]
        self.snake_2.snakeBodyLocation[2, 1] = [2, 3]

        snake_1_gameEndMask: numpy.ndarray = self.snake_1.generateGameEndMask(
            snake_1_nextSnakePosition
        )
        snake_2_gameEndMask: numpy.ndarray = self.snake_2.generateGameEndMask(
            snake_2_nextSnakePosition
        )

        self.assertTrue(
            equalNumpyArrays(snake_1_gameEndMask, numpy.array([True, True])),
            f"Snake 1 game end: {snake_1_gameEndMask}",
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_gameEndMask, numpy.array([False, True, True])),
            f"Snake 2 game end: {snake_2_gameEndMask}",
        )

    def test_updateSnakeBodyCoordinates(self) -> None:

        self.snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        self.snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1_nextSnakePosition: numpy.ndarray = numpy.array([[[2, 2]], [[3, 1]]])
        snake_2_nextSnakePosition: numpy.ndarray = numpy.array(
            [[[4, 2]], [[1, 1]], [[2, 4]]]
        )

        self.snake_1.snakeBodyLocation[0, 0] = [1, 2]  # Hit Nothing
        self.snake_1.snakeBodyLocation[0, 1] = [1, 1]
        self.snake_1.snakeBodyLocation[1, 0] = [2, 1]  # Hit Food
        self.snake_1.snakeBodyLocation[1, 1] = [1, 1]

        self.snake_2.snakeBodyLocation[0, 0] = [3, 2]  # Hit Food
        self.snake_2.snakeBodyLocation[0, 1] = [2, 2]
        self.snake_2.snakeBodyLocation[1, 0] = [1, 2]  # Hit Nothing
        self.snake_2.snakeBodyLocation[1, 1] = [1, 3]
        self.snake_2.snakeBodyLocation[2, 0] = [2, 3]  # Hit Food
        self.snake_2.snakeBodyLocation[2, 1] = [2, 2]

        self.snake_1.currentBodyEndIndex[:] = [1, 1]
        self.snake_2.currentBodyEndIndex[:] = [1, 1, 1]

        snake_1_snakeBodyLocation: numpy.ndarray = numpy.zeros((2, 20, 2))
        snake_2_snakeBodyLocation: numpy.ndarray = numpy.zeros((3, 36, 2))

        snake_1_snakeBodyLocation[0, 0] = [2, 2]
        snake_1_snakeBodyLocation[0, 1] = [1, 2]

        snake_1_snakeBodyLocation[1, 0] = [3, 1]
        snake_1_snakeBodyLocation[1, 1] = [2, 1]
        snake_1_snakeBodyLocation[1, 2] = [1, 1]

        snake_2_snakeBodyLocation[0, 0] = [4, 2]
        snake_2_snakeBodyLocation[0, 1] = [3, 2]
        snake_2_snakeBodyLocation[0, 2] = [2, 2]

        snake_2_snakeBodyLocation[1, 0] = [1, 1]
        snake_2_snakeBodyLocation[1, 1] = [1, 2]

        snake_2_snakeBodyLocation[2, 0] = [2, 4]
        snake_2_snakeBodyLocation[2, 1] = [2, 3]
        snake_2_snakeBodyLocation[2, 2] = [2, 2]

        snake_1_SnakeHitFoodMask: numpy.ndarray = numpy.array([False, True])
        snake_2_SnakeHitFoodMask: numpy.ndarray = numpy.array([True, False, True])

        self.snake_1.updateSnakeBodyCoordinates(
            snake_1_nextSnakePosition, snake_1_SnakeHitFoodMask
        )
        self.snake_2.updateSnakeBodyCoordinates(
            snake_2_nextSnakePosition, snake_2_SnakeHitFoodMask
        )

        self.assertTrue(
            equalNumpyArrays(self.snake_1.snakeBodyLocation, snake_1_snakeBodyLocation),
            f"First 3 elements of Snake 1 snakeBodyLocation: {self.snake_1.snakeBodyLocation[:, :3]}\nFirst 3 elements of Snake 1 test snakeBodyLocation: {snake_1_snakeBodyLocation[:, :3]}",
        )

        self.assertTrue(
            equalNumpyArrays(self.snake_2.snakeBodyLocation, snake_2_snakeBodyLocation),
            f"First 3 elements of Snake 2 snakeBodyLocation: {self.snake_2.snakeBodyLocation[:, :3]}\nFirst 3 elements of Snake 2 test snakeBodyLocation: {snake_2_snakeBodyLocation[:, :3]}",
        )

        snake_1_currentBodyEndIndex: numpy.ndarray = numpy.array([1, 2], dtype=int)
        snake_2_currentBodyEndIndex: numpy.ndarray = numpy.array([2, 1, 2], dtype=int)

        self.assertTrue(
            equalNumpyArrays(
                self.snake_1.currentBodyEndIndex, snake_1_currentBodyEndIndex
            ),
            f"Snake 1 true currentBodyEndIndex: {self.snake_1.currentBodyEndIndex}\nSnake 1 test currentBodyEndIndex: {snake_1_currentBodyEndIndex}",
        )

        self.assertTrue(
            equalNumpyArrays(
                self.snake_2.currentBodyEndIndex, snake_2_currentBodyEndIndex
            ),
            f"Snake 2 true currentBodyEndIndex: {self.snake_2.currentBodyEndIndex}\nSnake 2 test currentBodyEndIndex: {snake_2_currentBodyEndIndex}",
        )


if __name__ == "__main__":
    unittest.main()
