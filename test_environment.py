import unittest
import numpy
from environment import Environment


def equalNumpyArrays(array_1: numpy.ndarray, array_2: numpy.ndarray) -> bool:
    equalInPlace: numpy.ndarray = array_1 == array_2
    equal: bool = bool(equalInPlace.all())
    return equal


class TestEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self.gameDimensions_1: list[int] = [5, 4]
        self.numberOfGames_1: int = 2

        self.gameDimensions_2: list[int] = [6, 6]
        self.numberOfGames_2: int = 3

    def test_reset(self) -> None:
        environment_1: Environment = Environment(
            self.gameDimensions_1, self.numberOfGames_1
        )
        environment_2: Environment = Environment(
            self.gameDimensions_2, self.numberOfGames_2
        )

        environment_1.reset()
        environment_2.reset()

        environment_1_stateSpaceShape: numpy.ndarray = numpy.array(
            (self.numberOfGames_1, self.gameDimensions_1[1], self.gameDimensions_1[0])
        )
        environment_2_stateSpaceShape: numpy.ndarray = numpy.array(
            (self.numberOfGames_2, self.gameDimensions_2[1], self.gameDimensions_2[0])
        )

        self.assertTrue(
            equalNumpyArrays(
                environment_1.stateSpace.shape, environment_1_stateSpaceShape
            ),
            f"Environment 1 stateSpace shape: {environment_1.stateSpace.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(
                environment_2.stateSpace.shape, environment_2_stateSpaceShape
            ),
            f"Environment 2 stateSpace shape: {environment_2.stateSpace.shape}",
        )

        environment_1_stateSpace_numberOfOnes: numpy.ndarray = numpy.sum(
            environment_1.stateSpace == 1
        )
        environment_2_stateSpace_numberOfOnes: numpy.ndarray = numpy.sum(
            environment_2.stateSpace == 1
        )

        self.assertTrue(
            equalNumpyArrays(environment_1_stateSpace_numberOfOnes, numpy.array((28))),
            f"Environment 1 number of 0s in state space: {environment_1_stateSpace_numberOfOnes}",
        )
        self.assertTrue(
            equalNumpyArrays(environment_2_stateSpace_numberOfOnes, numpy.array((60))),
            f"Environment 2 number of 0s in state space: {environment_2_stateSpace_numberOfOnes}",
        )

    def test_drawInitialSnake(self) -> None:
        environment_1: Environment = Environment(
            self.gameDimensions_1, self.numberOfGames_1
        )
        environment_2: Environment = Environment(
            self.gameDimensions_2, self.numberOfGames_2
        )

        environment_1.snake.snakeBodyLocation[0, 0] = [2, 2]
        environment_1.snake.snakeBodyLocation[1, 0] = [3, 1]

        environment_2.snake.snakeBodyLocation[0, 0] = [1, 1]
        environment_2.snake.snakeBodyLocation[1, 0] = [2, 3]
        environment_2.snake.snakeBodyLocation[2, 0] = [2, 1]

        environment_1.drawInitialSnake()
        environment_2.drawInitialSnake()

        self.assertTrue(
            environment_1.stateSpace[0, 2, 2] == 2,
            f"Environment 1 State Space value at location (0, 2, 2): {environment_1.stateSpace[0, 2, 2]}",
        )
        self.assertTrue(
            environment_1.stateSpace[1, 3, 1] == 2,
            f"Environment 1 State Space value at location (1, 3, 1): {environment_1.stateSpace[1, 3, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[0, 1, 1] == 2,
            f"Environment 2 State Space value at location (0, 1, 1): {environment_2.stateSpace[0, 1, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[1, 2, 3] == 2,
            f"Environment 2 State Space value at location (1, 2, 3): {environment_2.stateSpace[1, 2, 3]}",
        )
        self.assertTrue(
            environment_2.stateSpace[2, 2, 1] == 2,
            f"Environment 2 State Space value at location (2, 2, 1): {environment_2.stateSpace[2, 2, 1]}",
        )

    def test_update(self) -> None:
        environment_1: Environment = Environment(
            self.gameDimensions_1, self.numberOfGames_1
        )
        environment_2: Environment = Environment(
            self.gameDimensions_2, self.numberOfGames_2
        )

        environment_1_moves: list[int] = [0, 3]
        environment_2_moves: list[int] = [2, 1, 2]

        environment_1.snake.foodLocation[0] = [3, 3]
        environment_1.snake.foodLocation[1] = [1, 1]
        environment_2.snake.foodLocation[:] = [1, 1]

        environment_1.snake.currentBodyEndIndex = numpy.array([2, 1], dtype=int)
        environment_2.snake.currentBodyEndIndex = numpy.array([0, 1, 2], dtype=int)

        environment_1.snake.snakeBodyLocation[0, 0] = [2, 1]
        environment_1.snake.snakeBodyLocation[0, 1] = [2, 2]
        environment_1.snake.snakeBodyLocation[0, 2] = [2, 3]

        environment_1.snake.snakeBodyLocation[1, 0] = [1, 2]
        environment_1.snake.snakeBodyLocation[1, 1] = [1, 3]

        environment_1.stateSpace[0, 2, 1] = 2
        environment_1.stateSpace[0, 2, 2] = 2
        environment_1.stateSpace[0, 2, 3] = 2
        environment_1.stateSpace[1, 1, 2] = 2
        environment_1.stateSpace[1, 1, 3] = 2

        environment_2.snake.snakeBodyLocation[0, 0] = [4, 1]

        environment_2.snake.snakeBodyLocation[1, 0] = [2, 2]
        environment_2.snake.snakeBodyLocation[1, 1] = [2, 3]

        environment_2.snake.snakeBodyLocation[2, 0] = [4, 2]
        environment_2.snake.snakeBodyLocation[2, 1] = [3, 2]
        environment_2.snake.snakeBodyLocation[2, 2] = [2, 2]

        environment_2.stateSpace[0, 4, 1] = 2
        environment_2.stateSpace[1, 2, 2] = 2
        environment_2.stateSpace[1, 2, 3] = 2
        environment_2.stateSpace[2, 4, 2] = 2
        environment_2.stateSpace[2, 3, 2] = 2
        environment_2.stateSpace[2, 2, 2] = 2

        (
            environment_1_gameEndMask,
            environment_1_snakeHitFoodMask,
            environment_1_allGamesEnded,
        ) = environment_1.update(environment_1_moves)

        (
            environment_2_gameEndMask,
            environment_2_snakeHitFoodMask,
            environment_2_allGamesEnded,
        ) = environment_2.update(environment_2_moves)

        environment_1_stateSpaceShape: numpy.ndarray = numpy.array([2, 4, 5])
        environment_1_gameEndMask_test: numpy.ndarray = numpy.array([False, False])
        environment_1_snakeHitFoodMask_test: numpy.ndarray = numpy.array([False, True])
        environment_1_allGamesEnded_test: bool = False

        environment_2_stateSpaceShape: numpy.ndarray = numpy.array([3, 6, 6])
        environment_2_gameEndMask_test: numpy.ndarray = numpy.array([True, True, True])
        environment_2_snakeHitFoodMask_test: numpy.ndarray = numpy.array(
            [False, False, False]
        )
        environment_2_allGamesEnded_test: bool = True

        self.assertTrue(
            equalNumpyArrays(
                environment_1.stateSpace.shape, environment_1_stateSpaceShape
            ),
            f"Environment 1 State space shape: {environment_1.stateSpace.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(environment_1_gameEndMask, environment_1_gameEndMask_test),
            f"Environment 1 Game End Mask: {environment_1_gameEndMask}",
        )
        self.assertTrue(
            equalNumpyArrays(
                environment_1_snakeHitFoodMask, environment_1_snakeHitFoodMask_test
            ),
            f"Environment 1 Snake Hit Food Mask: {environment_1_snakeHitFoodMask}",
        )
        self.assertTrue(
            environment_1_allGamesEnded == environment_1_allGamesEnded_test,
            f"Environment 1 All Games Ended value: {environment_1_allGamesEnded}",
        )

        self.assertTrue(
            equalNumpyArrays(
                environment_2.stateSpace.shape, environment_2_stateSpaceShape
            ),
            f"Environment 2 State space shape: {environment_2.stateSpace.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(environment_2_gameEndMask, environment_2_gameEndMask_test),
            f"Environment 2 Game End Mask: {environment_2_gameEndMask}",
        )
        self.assertTrue(
            equalNumpyArrays(
                environment_2_snakeHitFoodMask, environment_2_snakeHitFoodMask_test
            ),
            f"Environment 2 Snake Hit Food Mask: {environment_2_snakeHitFoodMask}",
        )
        self.assertTrue(
            environment_2_allGamesEnded == environment_2_allGamesEnded_test,
            f"Environment 2 All Games Ended value: {environment_2_allGamesEnded}",
        )
        self.assertTrue(
            environment_1.stateSpace[0, 2, 3] == 0,
            f"Environment 1 State Space value at coordinates (0, 2, 3): {environment_1.stateSpace[0, 2, 3]}",
        )
        self.assertTrue(
            environment_1.stateSpace[0, 1, 1] == 2,
            f"Environment 1 State Space value at coordinates (0, 1, 1): {environment_1.stateSpace[0, 1, 1]}",
        )
        self.assertTrue(
            environment_1.stateSpace[1, 1, 1] == 2,
            f"Environment 1 State Space value at coordinates (1, 1, 1): {environment_1.stateSpace[1, 1, 1]}",
        )
        self.assertTrue(
            environment_1.stateSpace[1, 1, 2] == 2,
            f"Environment 1 State Space value at coordinates (1, 1, 2): {environment_1.stateSpace[1, 1, 2]}",
        )
        self.assertTrue(
            environment_1.stateSpace[1, 1, 3] == 2,
            f"Environment 1 State Space value at coordinates (1, 1, 3): {environment_1.stateSpace[1, 1, 3]}",
        )

        self.assertTrue(
            environment_2.stateSpace[0, 4, 1] == 0,
            f"Environment 2 State Space value at coordinates (0, 4, 1): {environment_2.stateSpace[0, 4, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[0, 5, 1] == 2,
            f"Environment 2 State Space value at coordinates (0, 5, 1): {environment_2.stateSpace[0, 5, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[1, 2, 2] == 2,
            f"Environment 2 State Space value at coordinates (1, 2, 2): {environment_2.stateSpace[1, 2, 2]}",
        )
        self.assertTrue(
            environment_2.stateSpace[1, 2, 3] == 2,
            f"Environment 2 State Space value at coordinates (1, 2, 3): {environment_2.stateSpace[1, 2, 3]}",
        )
        self.assertTrue(
            environment_2.stateSpace[2, 5, 2] == 2,
            f"Environment 2 State Space value at coordinates (2, 5, 2): {environment_2.stateSpace[2, 5, 2]}",
        )
        self.assertTrue(
            environment_2.stateSpace[2, 2, 2] == 0,
            f"Environment 2 State Space value at coordinates (2, 2, 2): {environment_2.stateSpace[2, 2, 2]}",
        )

    def test_updateFoodLocation_Array(self) -> None:
        environment_1: Environment = Environment(
            self.gameDimensions_1, self.numberOfGames_1
        )
        environment_2: Environment = Environment(
            self.gameDimensions_2, self.numberOfGames_2
        )

        environment_1.snake.foodLocation[0] = [2, 2]
        environment_1.snake.foodLocation[1] = [3, 1]

        environment_2.snake.foodLocation[0] = [1, 1]
        environment_2.snake.foodLocation[1] = [2, 3]
        environment_2.snake.foodLocation[2] = [2, 1]

        environment_1.stateSpace[1, 3, 1] = 0
        environment_2.stateSpace[2, 2, 1] = 0

        environment_1_snakeHitFoodMask: numpy.ndarray = numpy.array([True, False])
        environment_2_snakeHitFoodMask: numpy.ndarray = numpy.array([True, True, False])

        environment_1.updateFoodLocation(environment_1_snakeHitFoodMask)
        environment_2.updateFoodLocation(environment_2_snakeHitFoodMask)

        self.assertTrue(
            environment_1.stateSpace[0, 2, 2] == 3,
            f"Environment 1 State Space value at location (0, 2, 2): {environment_1.stateSpace[0, 2, 2]}",
        )
        self.assertTrue(
            environment_1.stateSpace[1, 3, 1] == 0,
            f"Environment 1 State Space value at location (1, 3, 1): {environment_1.stateSpace[1, 3, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[0, 1, 1] == 3,
            f"Environment 2 State Space value at location (0, 1, 1): {environment_2.stateSpace[0, 1, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[1, 2, 3] == 3,
            f"Environment 2 State Space value at location (1, 2, 3): {environment_2.stateSpace[1, 2, 3]}",
        )
        self.assertTrue(
            environment_2.stateSpace[2, 2, 1] == 0,
            f"Environment 2 State Space value at location (2, 2, 1): {environment_2.stateSpace[2, 2, 1]}",
        )

    def test_upateFoodLocation_None(self) -> None:
        environment_1: Environment = Environment(
            self.gameDimensions_1, self.numberOfGames_1
        )
        environment_2: Environment = Environment(
            self.gameDimensions_2, self.numberOfGames_2
        )

        environment_1.snake.foodLocation[0] = [2, 2]
        environment_1.snake.foodLocation[1] = [3, 1]

        environment_2.snake.foodLocation[0] = [1, 1]
        environment_2.snake.foodLocation[1] = [2, 3]
        environment_2.snake.foodLocation[2] = [2, 1]

        environment_1.updateFoodLocation()
        environment_2.updateFoodLocation()

        self.assertTrue(
            environment_1.stateSpace[0, 2, 2] == 3,
            f"Environment 1 State Space value at location (0, 2, 2): {environment_1.stateSpace[0, 2, 2]}",
        )
        self.assertTrue(
            environment_1.stateSpace[1, 3, 1] == 3,
            f"Environment 1 State Space value at location (1, 3, 1): {environment_1.stateSpace[1, 3, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[0, 1, 1] == 3,
            f"Environment 2 State Space value at location (0, 1, 1): {environment_2.stateSpace[0, 1, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[1, 2, 3] == 3,
            f"Environment 2 State Space value at location (1, 2, 3): {environment_2.stateSpace[1, 2, 3]}",
        )
        self.assertTrue(
            environment_2.stateSpace[2, 2, 1] == 3,
            f"Environment 2 State Space value at location (2, 2, 1): {environment_2.stateSpace[2, 2, 1]}",
        )

    def test_removeFromStateSpace(self) -> None:
        environment_1: Environment = Environment(
            self.gameDimensions_1, self.numberOfGames_1
        )
        environment_2: Environment = Environment(
            self.gameDimensions_2, self.numberOfGames_2
        )

        environment_1.snake.currentBodyEndIndex = numpy.array([1, 2], dtype=int)
        environment_2.snake.currentBodyEndIndex = numpy.array([2, 0, 0], dtype=int)
        environment_1_snakeHitFoodMask: numpy.ndarray = numpy.array([True, False])
        environment_2_snakeHitFoodMask: numpy.ndarray = numpy.array(
            [False, True, False]
        )

        environment_1.stateSpace[1, 2, 2] = 0
        environment_2.stateSpace[0, 2, 4] = 0
        environment_2.stateSpace[2, 2, 4] = 0

        environment_1.stateSpace[0, 3, 3] = 2
        environment_1.stateSpace[1, 2, 2] = 2
        environment_1.snake.snakeBodyLocation[0, 1] = [3, 3]
        environment_1.snake.snakeBodyLocation[1, 2] = [2, 2]

        environment_2.stateSpace[0, 2, 3] = 2
        environment_2.stateSpace[1, 4, 1] = 2
        environment_2.stateSpace[2, 2, 4] = 2
        environment_2.snake.snakeBodyLocation[0, 2] = [2, 3]
        environment_2.snake.snakeBodyLocation[1, 0] = [4, 1]
        environment_2.snake.snakeBodyLocation[2, 0] = [2, 4]

        environment_1.removeFromStateSpace(environment_1_snakeHitFoodMask)

        environment_2.removeFromStateSpace(environment_2_snakeHitFoodMask)

        self.assertTrue(
            environment_1.stateSpace[0, 3, 3] == 2,
            f"Environment 1 state space value at (0, 3, 3): {environment_1.stateSpace[0, 3, 3]}",
        )
        self.assertTrue(
            environment_1.stateSpace[1, 2, 2] == 0,
            f"Environment 1 state space value at (1, 2, 2): {environment_1.stateSpace[1, 2, 2]}",
        )
        self.assertTrue(
            environment_2.stateSpace[0, 2, 4] == 0,
            f"Environment 2 state space at (0, 2, 4): {environment_2.stateSpace[0, 2, 4]}",
        )
        self.assertTrue(
            environment_2.stateSpace[1, 4, 1] == 2,
            f"Environment 2 state space at (1, 4, 1): {environment_2.stateSpace[1, 4, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[2, 2, 4] == 0,
            f"Environment 2 state space at (2, 2, 4): {environment_2.stateSpace[2, 2, 4]}",
        )

    def test_updateStateSpace(self) -> None:
        environment_1: Environment = Environment(
            self.gameDimensions_1, self.numberOfGames_1
        )
        environment_2: Environment = Environment(
            self.gameDimensions_2, self.numberOfGames_2
        )

        environment_1.snake.snakeBodyLocation[0, 0] = [3, 2]
        environment_1.snake.snakeBodyLocation[1, 0] = [2, 1]
        environment_2.snake.snakeBodyLocation[0, 0] = [4, 4]
        environment_2.snake.snakeBodyLocation[1, 0] = [1, 1]
        environment_2.snake.snakeBodyLocation[2, 0] = [2, 2]

        environment_1.updateStateSpace()
        environment_2.updateStateSpace()

        self.assertTrue(
            environment_1.stateSpace[0, 3, 2] == 2,
            f"Environment 1 state space value at (0, 3, 2): {environment_1.stateSpace[0, 3, 2]}",
        )
        self.assertTrue(
            environment_1.stateSpace[1, 2, 1] == 2,
            f"Environment 1 state space value at (1, 2, 1): {environment_1.stateSpace[1, 2, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[0, 4, 4] == 2,
            f"Environment 2 state space at (0, 4, 4): {environment_2.stateSpace[0, 4, 4]}",
        )
        self.assertTrue(
            environment_2.stateSpace[1, 1, 1] == 2,
            f"Environment 2 state space at (1, 1, 1): {environment_2.stateSpace[1, 1, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[2, 2, 2] == 2,
            f"Environment 2 state space at (2, 2, 2): {environment_2.stateSpace[2, 2, 2]}",
        )

    def test_removeEndedGames(self) -> None:
        environment_1: Environment = Environment(
            self.gameDimensions_1, self.numberOfGames_1
        )
        environment_2: Environment = Environment(
            self.gameDimensions_2, self.numberOfGames_2
        )

        environment_1_gameEndMask: numpy.ndarray = numpy.array([False, True])
        environment_2_gameEndMask: numpy.ndarray = numpy.array([True, False, False])

        environment_1_stateSpace_shape: numpy.ndarray = numpy.array([1, 4, 5])
        environment_1_snakeBodyLocation_shape: numpy.ndarray = numpy.array([1, 20, 2])
        environment_1_foodLocation_shape: numpy.ndarray = numpy.array([1, 2])
        environment_1_currentBodyEndIndex_shape: numpy.ndarray = numpy.array([1])

        environment_2_stateSpace_shape: numpy.ndarray = numpy.array([2, 6, 6])
        environment_2_snakeBodyLocation_shape: numpy.ndarray = numpy.array([2, 36, 2])
        environment_2_foodLocation_shape: numpy.ndarray = numpy.array([2, 2])
        environment_2_currentBodyEndIndex_shape: numpy.ndarray = numpy.array([2])

        environment_1.stateSpace[0, 3, 3] = 5
        environment_1.stateSpace[1, 2, 2] = 5

        environment_2.stateSpace[0, 2, 3] = 5
        environment_2.stateSpace[1, 4, 1] = 5
        environment_2.stateSpace[2, 2, 4] = 5

        environment_1.removeEndedGames(environment_1_gameEndMask)
        environment_2.removeEndedGames(environment_2_gameEndMask)

        self.assertTrue(
            equalNumpyArrays(
                environment_1.stateSpace.shape, environment_1_stateSpace_shape
            ),
            f"Environment 1 State Space shape: {environment_1.stateSpace.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(
                environment_1.snake.snakeBodyLocation.shape,
                environment_1_snakeBodyLocation_shape,
            ),
            f"Environment 1 Snake Body Location shape: {environment_1.snake.snakeBodyLocation.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(
                environment_1.snake.foodLocation.shape, environment_1_foodLocation_shape
            ),
            f"Environment 1 Food Locatino shape: {environment_1.snake.foodLocation.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(
                environment_1.snake.currentBodyEndIndex.shape,
                environment_1_currentBodyEndIndex_shape,
            ),
            f"Environment 1 Current Body End Index shape: {environment_1.snake.currentBodyEndIndex.shape}",
        )

        self.assertTrue(
            equalNumpyArrays(
                environment_2.stateSpace.shape, environment_2_stateSpace_shape
            ),
            f"Environment 2 State Space shape: {environment_2.stateSpace.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(
                environment_2.snake.snakeBodyLocation.shape,
                environment_2_snakeBodyLocation_shape,
            ),
            f"Environment 2 Snake Body Location shape: {environment_2.snake.snakeBodyLocation.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(
                environment_2.snake.foodLocation.shape, environment_2_foodLocation_shape
            ),
            f"Environment 2 Food Locatino shape: {environment_2.snake.foodLocation.shape}",
        )
        self.assertTrue(
            equalNumpyArrays(
                environment_2.snake.currentBodyEndIndex.shape,
                environment_2_currentBodyEndIndex_shape,
            ),
            f"Environment 2 Current Body End Index shape: {environment_2.snake.currentBodyEndIndex.shape}",
        )

        self.assertTrue(
            environment_1.stateSpace[0, 3, 3] == 5,
            f"Environment 1 state space value at (0, 3, 3): {environment_1.stateSpace[0, 3, 3]}",
        )
        self.assertTrue(
            environment_1.stateSpace[0, 2, 2] != 5,
            f"Environment 1 state space value at (0, 2, 2): {environment_1.stateSpace[0, 2, 2]}",
        )
        self.assertTrue(
            environment_2.stateSpace[0, 4, 1] == 5,
            f"Environment 2 state space at (0, 4, 1): {environment_2.stateSpace[0, 4, 1]}",
        )
        self.assertTrue(
            environment_2.stateSpace[1, 2, 4] == 5,
            f"Environment 2 state space at (1, 2, 4): {environment_2.stateSpace[1, 2, 4]}",
        )
        self.assertTrue(
            environment_2.stateSpace[0, 2, 3] != 5,
            f"Environment 2 state space at (0, 2, 3): {environment_2.stateSpace[0, 2, 3]}",
        )
        self.assertTrue(
            environment_2.stateSpace[1, 2, 3] != 5,
            f"Environment 2 state space at (1, 2, 3): {environment_2.stateSpace[1, 2, 3]}",
        )
