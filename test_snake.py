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
        snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

        self.assertTrue(snake_1.snakeBodyLocation.shape == (2, 5 * 4, 2))
        self.assertTrue(snake_2.snakeBodyLocation.shape == (3, 6 * 6, 2))

    def test_initFoodShape(self) -> None:
        snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

        self.assertTrue(snake_1.foodLocation.shape == (2, 2))
        self.assertTrue(snake_2.foodLocation.shape == (3, 2))

    def test_generateRandomLocation_exceptions(self) -> None:
        snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

        with self.assertRaises(Exception):
            snake_1.generateRandomLocations(
                resetGame=True, snakeHitFoodMask=numpy.array([True, False])
            )
        with self.assertRaises(Exception):
            snake_2.generateRandomLocations(
                resetGame=True, snakeHitFoodMask=numpy.array([True, True, False])
            )

        with self.assertRaises(Exception):
            snake_1.generateRandomLocations()

        with self.assertRaises(Exception):
            snake_2.generateRandomLocations()

    def test_generateRandomLocation_coordinatesOnReset(self) -> None:
        snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

        makeFoodSameAsHead(snake_1)
        makeFoodSameAsHead(snake_2)

        makeBodySameAsHead(snake_1)
        makeBodySameAsHead(snake_2)

        snake_1.generateRandomLocations(resetGame=True)
        snake_2.generateRandomLocations(resetGame=True)

        for gameIndex in range(snake_1.numberOfGames):
            self.assertTrue(
                equalNumpyArrays(
                    snake_1.foodLocation[gameIndex],
                    snake_1.snakeBodyLocation[gameIndex, 0],
                )
                == False,
            )

        for gameIndex in range(snake_2.numberOfGames):
            self.assertTrue(
                equalNumpyArrays(
                    snake_2.foodLocation[gameIndex],
                    snake_2.snakeBodyLocation[gameIndex, 0],
                )
                == False,
            )

    def test_generateCoordinatesOnReset(self) -> None:
        snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

        makeFoodSameAsHead(snake_1)
        makeFoodSameAsHead(snake_2)

        makeBodySameAsHead(snake_1)
        makeBodySameAsHead(snake_2)

        snake_1.generateCoordinatesOnReset()
        snake_2.generateCoordinatesOnReset()

        for gameIndex in range(snake_1.numberOfGames):
            self.assertTrue(
                equalNumpyArrays(
                    snake_1.foodLocation[gameIndex],
                    snake_1.snakeBodyLocation[gameIndex, 0],
                )
                == False,
            )

        for gameIndex in range(snake_2.numberOfGames):
            self.assertTrue(
                equalNumpyArrays(
                    snake_2.foodLocation[gameIndex],
                    snake_2.snakeBodyLocation[gameIndex, 0],
                )
                == False,
            )

    def test_generateRandomLocation_coordinatesFromMask(self) -> None:
        snake_1 = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2 = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1.foodLocation[0] = snake_1.snakeBodyLocation[0, 0]
        snake_2.foodLocation[0] = snake_2.snakeBodyLocation[0, 0]
        snake_2.foodLocation[1] = snake_2.snakeBodyLocation[1, 0]

        snakeHitFoodMask_1: numpy.ndarray = numpy.array([True, False])
        snakeHitFoodMask_2: numpy.ndarray = numpy.array([True, True, False])

        snake_1.generateRandomLocations(snakeHitFoodMask=snakeHitFoodMask_1)
        snake_2.generateRandomLocations(snakeHitFoodMask=snakeHitFoodMask_2)

        self.assertTrue(
            equalNumpyArrays(snake_1.snakeBodyLocation[0, 0], snake_1.foodLocation[0])
            == False
        )
        self.assertTrue(
            equalNumpyArrays(snake_2.snakeBodyLocation[0, 0], snake_2.foodLocation[0])
            == False
        )
        self.assertTrue(
            equalNumpyArrays(snake_2.snakeBodyLocation[1, 0], snake_2.foodLocation[1])
            == False
        )

    def test_generateCoordiantesFromMask(self) -> None:
        snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1.foodLocation[0] = snake_1.snakeBodyLocation[0, 0]
        snake_2.foodLocation[0] = snake_2.snakeBodyLocation[0, 0]
        snake_2.foodLocation[1] = snake_2.snakeBodyLocation[1, 0]

        snakeHitFoodMaskMask_1: numpy.ndarray = numpy.array([True, False])
        snakeHitFoodMaskMask_2: numpy.ndarray = numpy.array([True, True, False])

        snake_1.generateCoordinatesFromMask(snakeHitFoodMaskMask_1)
        snake_2.generateCoordinatesFromMask(snakeHitFoodMaskMask_2)

        self.assertTrue(
            equalNumpyArrays(snake_1.snakeBodyLocation[0, 0], snake_1.foodLocation[0])
            == False
        )
        self.assertTrue(
            equalNumpyArrays(snake_2.snakeBodyLocation[0, 0], snake_2.foodLocation[0])
            == False
        )
        self.assertTrue(
            equalNumpyArrays(snake_2.snakeBodyLocation[1, 0], snake_2.foodLocation[1])
            == False
        )

    def test_findSnakeHitSelf(self) -> None:
        snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1.snakeBodyLocation[0, 1] = snake_1.snakeBodyLocation[0, 0]
        snake_2.snakeBodyLocation[1, 1] = snake_2.snakeBodyLocation[1, 0]
        snake_2.snakeBodyLocation[2, 1] = snake_2.snakeBodyLocation[2, 0]

        snakeHitSelfMask_1: numpy.ndarray = snake_1.findSnakeHitSelf(
            snake_1.snakeBodyLocation[:, 0, :].reshape(snake_1.numberOfGames, 1, -1)
        )
        snakeHitSelfMask_2: numpy.ndarray = snake_2.findSnakeHitSelf(
            snake_2.snakeBodyLocation[:, 0, :].reshape(snake_2.numberOfGames, 1, -1)
        )

        self.assertTrue(
            equalNumpyArrays(snakeHitSelfMask_1, numpy.array([True, False])),
            f"Snake 1 snakeHitSelfMask_1: {snakeHitSelfMask_1}",
        )
        self.assertTrue(
            equalNumpyArrays(snakeHitSelfMask_2, numpy.array([False, True, True])),
            f"Snake 2 snakeHitSelfMask_2: {snakeHitSelfMask_2}",
        )

    def test_generateMasks_hitFood(self) -> None:

        snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1_nextSnakePosition: numpy.ndarray = numpy.array([[[1, 2]], [[2, 2]]])
        snake_2_nextSnakePosition: numpy.ndarray = numpy.array(
            [[[1, 3]], [[2, 2]], [[2, 2]]]
        )

        snake_1.foodLocation[:] = [[1, 2], [2, 1]]
        snake_2.foodLocation[:] = [[1, 2], [2, 2], [2, 1]]

        _, snake_1_snakeHitFoodMask = snake_1.generateMasks(snake_1_nextSnakePosition)

        _, snake_2_snakeHitFoodMask = snake_2.generateMasks(snake_2_nextSnakePosition)

        self.assertTrue(
            equalNumpyArrays(
                snake_1_snakeHitFoodMask, numpy.array([True, False], dtype=bool)
            ),
            f"Snake 1 HitFood: {snake_1_snakeHitFoodMask}",
        )
        self.assertTrue(
            equalNumpyArrays(
                snake_2_snakeHitFoodMask, numpy.array([False, True, False], dtype=bool)
            ),
            f"Snake 2 HitFood: {snake_2_snakeHitFoodMask}",
        )

    def test_generateNextSnakePosition(self) -> None:
        snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1_moveDirections: list[int] = [0, 1]
        snake_2_moveDirections: list[int] = [1, 2, 3]

        snake_1.snakeBodyLocation[0, 0] = [2, 2]
        snake_1.snakeBodyLocation[1, 0] = [3, 2]

        snake_2.snakeBodyLocation[0, 0] = [2, 2]
        snake_2.snakeBodyLocation[1, 0] = [3, 2]
        snake_2.snakeBodyLocation[2, 0] = [2, 2]

        snake_1_nextSnakePosition: numpy.ndarray = snake_1.generateNextSnakePosition(
            snake_1_moveDirections
        )

        snake_2_nextSnakePosition: numpy.ndarray = snake_2.generateNextSnakePosition(
            snake_2_moveDirections
        )

        snake_1_nextSnakePosition_test: numpy.ndarray = numpy.array(
            [[[1, 2]], [[3, 3]]]
        )
        snake_2_nextSnakePosition_test: numpy.ndarray = numpy.array(
            [[[2, 3]], [[4, 2]], [[2, 1]]]
        )

        self.assertTrue(
            equalNumpyArrays(snake_1_nextSnakePosition, snake_1_nextSnakePosition_test),
            f"Snake 1 Next Snake Position: {snake_1_nextSnakePosition}",
        )

        self.assertTrue(
            equalNumpyArrays(snake_2_nextSnakePosition, snake_2_nextSnakePosition_test),
            f"Snake 2 Next Snake Position: {snake_2_nextSnakePosition}",
        )

    def test_generateMasks_hitWall(self) -> None:
        snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1_nextSnakePosition: numpy.ndarray = numpy.array([[[1, 0]], [[1, 1]]])
        snake_2_nextSnakePosition: numpy.ndarray = numpy.array(
            [[[5, 2]], [[1, 1]], [[2, 5]]]
        )

        snake_1.foodLocation[:] = [0, 0]
        snake_2.foodLocation[:] = [0, 0]

        snake_1_gameEndMask, snake_1_hitFoodMask = snake_1.generateMasks(
            snake_1_nextSnakePosition
        )
        snake_2_gameEndMask, snake_2_hitFoodMask = snake_2.generateMasks(
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

        self.assertTrue(
            equalNumpyArrays(snake_1_hitFoodMask, numpy.array([False, False])),
            f"Snake 1 Hit Food Mask: {snake_1_hitFoodMask}",
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_hitFoodMask, numpy.array([False, False, False])),
            f"Snake 2 Hit Food Mask: {snake_2_hitFoodMask}",
        )

    def test_generateMasks_hitSelf(self) -> None:

        snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1_nextSnakePosition: numpy.ndarray = numpy.array([[[1, 1]], [[1, 1]]])
        snake_2_nextSnakePosition: numpy.ndarray = numpy.array(
            [[[4, 2]], [[1, 1]], [[2, 4]]]
        )
        snake_1.snakeBodyLocation[0, 1] = [3, 2]
        snake_1.snakeBodyLocation[1, 1] = [1, 1]

        snake_2.snakeBodyLocation[0, 1] = [3, 2]
        snake_2.snakeBodyLocation[1, 1] = [1, 1]
        snake_2.snakeBodyLocation[2, 1] = [2, 3]

        snake_1.foodLocation[:] = [0, 0]
        snake_2.foodLocation[:] = [0, 0]

        snake_1_gameEndMask, snake_1_hitFoodMask = snake_1.generateMasks(
            snake_1_nextSnakePosition
        )
        snake_2_gameEndMask, snake_2_hitFoodMask = snake_2.generateMasks(
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

        self.assertTrue(
            equalNumpyArrays(snake_1_hitFoodMask, numpy.array([False, False])),
            f"Snake 1 Hit Food Mask: {snake_1_hitFoodMask}",
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_hitFoodMask, numpy.array([False, False, False])),
            f"Snake 2 Hit Food Mask: {snake_2_hitFoodMask}",
        )

    def test_generateMasks_hitMix(self) -> None:

        snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1_nextSnakePosition: numpy.ndarray = numpy.array([[[1, 2]], [[1, 0]]])
        snake_2_nextSnakePosition: numpy.ndarray = numpy.array(
            [[[4, 2]], [[1, 1]], [[2, 5]]]
        )
        snake_1.snakeBodyLocation[0, 1] = [1, 2]
        snake_1.snakeBodyLocation[1, 1] = [1, 1]

        snake_2.snakeBodyLocation[0, 1] = [3, 2]
        snake_2.snakeBodyLocation[1, 1] = [1, 1]
        snake_2.snakeBodyLocation[2, 1] = [2, 3]

        snake_1.foodLocation[:] = [0, 0]
        snake_2.foodLocation[:] = [0, 0]

        snake_1_gameEndMask, snake_1_hitFoodMask = snake_1.generateMasks(
            snake_1_nextSnakePosition
        )
        snake_2_gameEndMask, snake_2_hitFoodMask = snake_2.generateMasks(
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

        self.assertTrue(
            equalNumpyArrays(snake_1_hitFoodMask, numpy.array([False, False])),
            f"Snake 1 Hit Food Mask: {snake_1_hitFoodMask}",
        )
        self.assertTrue(
            equalNumpyArrays(snake_2_hitFoodMask, numpy.array([False, False, False])),
            f"Snake 2 Hit Food Mask: {snake_2_hitFoodMask}",
        )

    def test_updateSnakeBodyCoordinates(self) -> None:

        snake_1: Snake = Snake(self.gameDimensions_1, self.numberOfGames_1)
        snake_2: Snake = Snake(self.gameDimensions_2, self.numberOfGames_2)

        snake_1_nextSnakePosition: numpy.ndarray = numpy.array([[[2, 2]], [[3, 1]]])
        snake_2_nextSnakePosition: numpy.ndarray = numpy.array(
            [[[4, 2]], [[1, 1]], [[2, 4]]]
        )

        snake_1.snakeBodyLocation[0, 0] = [1, 2]
        snake_1.snakeBodyLocation[0, 1] = [1, 1]
        snake_1.snakeBodyLocation[1, 0] = [2, 1]
        snake_1.snakeBodyLocation[1, 1] = [1, 1]

        snake_2.snakeBodyLocation[0, 0] = [3, 2]
        snake_2.snakeBodyLocation[0, 1] = [2, 2]
        snake_2.snakeBodyLocation[1, 0] = [1, 2]
        snake_2.snakeBodyLocation[1, 1] = [1, 3]
        snake_2.snakeBodyLocation[2, 0] = [2, 3]
        snake_2.snakeBodyLocation[2, 1] = [2, 2]

        snake_1.currentBodyEndIndex[:] = [1, 1]
        snake_2.currentBodyEndIndex[:] = [1, 1, 1]

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

        snake_1.updateSnakeBodyCoordinates(
            snake_1_nextSnakePosition, snake_1_SnakeHitFoodMask
        )
        snake_2.updateSnakeBodyCoordinates(
            snake_2_nextSnakePosition, snake_2_SnakeHitFoodMask
        )

        self.assertTrue(
            equalNumpyArrays(snake_1.snakeBodyLocation, snake_1_snakeBodyLocation),
            f"First 3 elements of Snake 1 snakeBodyLocation: {snake_1.snakeBodyLocation[:, :3]}\nFirst 3 elements of Snake 1 test snakeBodyLocation: {snake_1_snakeBodyLocation[:, :3]}",
        )

        self.assertTrue(
            equalNumpyArrays(snake_2.snakeBodyLocation, snake_2_snakeBodyLocation),
            f"First 3 elements of Snake 2 snakeBodyLocation: {snake_2.snakeBodyLocation[:, :3]}\nFirst 3 elements of Snake 2 test snakeBodyLocation: {snake_2_snakeBodyLocation[:, :3]}",
        )

        snake_1_currentBodyEndIndex: numpy.ndarray = numpy.array([1, 2], dtype=int)
        snake_2_currentBodyEndIndex: numpy.ndarray = numpy.array([2, 1, 2], dtype=int)

        self.assertTrue(
            equalNumpyArrays(snake_1.currentBodyEndIndex, snake_1_currentBodyEndIndex),
            f"Snake 1 true currentBodyEndIndex: {snake_1.currentBodyEndIndex}\nSnake 1 test currentBodyEndIndex: {snake_1_currentBodyEndIndex}",
        )

        self.assertTrue(
            equalNumpyArrays(snake_2.currentBodyEndIndex, snake_2_currentBodyEndIndex),
            f"Snake 2 true currentBodyEndIndex: {snake_2.currentBodyEndIndex}\nSnake 2 test currentBodyEndIndex: {snake_2_currentBodyEndIndex}",
        )


if __name__ == "__main__":
    unittest.main()
