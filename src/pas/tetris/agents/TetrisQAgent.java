package src.pas.tetris.agents;

// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Arrays;
import java.util.ArrayList;

// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU; // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;
import edu.bu.tetris.linalg.Shape;
import edu.bu.tetris.utils.Coordinate;

public class TetrisQAgent extends QAgent {

    private Random random;

    public TetrisQAgent(String name) {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() {
        return this.random;
    }

    @Override
    public Model initQFunction() {
        // The input to the neural network is the vector representation of the board
        // state
        final int numFeatures = Board.NUM_COLS * 3;
        final int hiddenDim1 = 3 * numFeatures; // Biggest hidden layer
        final int hiddenDim2 = 2 * numFeatures; // Smaller hidden layer
        final int hiddenDim3 = numFeatures; // Even smaller hidden layer
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(numFeatures, hiddenDim1)); // Biggest hidden layer
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim1, hiddenDim2)); // Smaller hidden layer
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim2, hiddenDim3)); // Even smaller hidden layer
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim3, outDim)); // Output layer

        return qFunction;
    }

    @Override
    public Matrix getQFunctionInput(final GameView game, final Mino potentialAction) {
        Matrix tetrisBoardMatrix = null;
        try {
            tetrisBoardMatrix = game.getGrayscaleImage(potentialAction);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }

        Shape shape = tetrisBoardMatrix.getShape();
        int numRows = shape.getNumRows();
        int numCols = shape.getNumCols();

        // Create a new matrix representing the row vector
        Matrix featureVector = Matrix.zeros(1, numCols * 3);
        int index = 0;

        // Iterate over each column
        for (int col = 0; col < numCols; col++) {
            // Initialize min and max position variables
            int firstMinoPosition = -1; // Initialize to -1, indicating no mino found yet. Topmost mino
            int lastMinoPosition = -1; // Initialize to -1, indicating no mino found yet. Bottommost mino

            // Iterate over each row from top to bottom
            for (int row = 0; row < numRows; row++) {
                // Check if the current cell is non-zero, indicating a mino
                if (tetrisBoardMatrix.get(row, col) > 0) {
                    // If this is the first mino found in the column, record its position
                    if (firstMinoPosition == -1) {
                        firstMinoPosition = row;
                    }
                    // Update the last mino position on each iteration
                    lastMinoPosition = row;
                }
            }
            int numOpenBlocks = firstMinoPosition == -1 ? (numRows - 1) : (firstMinoPosition - 1);

            // Add the features to the row vector
            featureVector.set(0, index++, firstMinoPosition); // Add the position of the first mino
            featureVector.set(0, index++, lastMinoPosition); // Add the position of the last mino
            featureVector.set(0, index++, numOpenBlocks);
        }

        return featureVector;
    }

    private int explorationStep = 0;
    private int explorationDecaySteps = 1000; // Number of steps over which exploration probability decays
    private double initialExplorationProb = 0.5; // Initial exploration probability

    @Override
    public boolean shouldExplore(final GameView game, final GameCounter gameCounter) {
        // Increment the step count
        explorationStep++;

        // Calculate the current exploration probability based on decay
        double currentExplorationProb = initialExplorationProb
                * Math.exp(-explorationStep / (double) explorationDecaySteps);

        // Generate a random number between 0 and 1
        double randomValue = this.getRandom().nextDouble();

        // Check if the random value is less than or equal to the current exploration
        // probability
        return randomValue <= currentExplorationProb;
    }

    @Override
    public Mino getExplorationMove(final GameView game) {
        List<Mino> availableActions = game.getFinalMinoPositions();
        int[] columnHeights = new int[game.getBoard().NUM_COLS];

        // Calculate the current height of each column on the board
        for (int col = 0; col < game.getBoard().NUM_COLS; col++) {
            for (int row = game.getBoard().NUM_ROWS - 1; row >= 0; row--) {
                if (game.getBoard().isCoordinateOccupied(col, row)) {
                    columnHeights[col] = game.getBoard().NUM_ROWS - row;
                    break;
                }
            }
        }

        // Find the minimum height among all columns
        int minHeight = Arrays.stream(columnHeights).min().orElse(0);

        // Find the actions that result in placing a piece in columns with the minimum
        // height
        List<Mino> minHeightActions = new ArrayList<>();
        for (Mino action : availableActions) {
            Coordinate pivot = action.getPivotBlockCoordinate();
            int pivotColumn = pivot.getXCoordinate(); // Get the column of the pivot block

            if (columnHeights[pivotColumn] == minHeight) {
                minHeightActions.add(action);
            }
        }

        // If there are actions available at the minimum height columns, choose one
        // randomly
        if (!minHeightActions.isEmpty()) {
            int randIdx = this.getRandom().nextInt(minHeightActions.size());
            return minHeightActions.get(randIdx);
        }

        // If no actions are available at the minimum height columns, choose randomly
        // from all available actions
        int randIdx = this.getRandom().nextInt(availableActions.size());
        return availableActions.get(randIdx);
    }

    @Override
    public double getReward(final GameView game) {
        double reward = game.getScoreThisTurn() * 1000; // Highly reward getting a score

        Board board = game.getBoard();
        int maxAllowedHeight = 18;
        int[] columnHeights = new int[board.NUM_COLS];

        // Calculate the column heights
        for (int col = 0; col < board.NUM_COLS; col++) {
            for (int row = board.NUM_ROWS - 1; row >= 0; row--) {
                if (board.isCoordinateOccupied(col, row)) {
                    columnHeights[col] = board.NUM_ROWS - row;
                    break;
                }
            }
        }

        // Reinforcement 1: Penalize for higher stack of minos
        for (int height : columnHeights) {
            reward -= 0.1 * height; // Penalize for each block height in a column
        }

        // Reinforcement 2: Penalize for height variance across columns to encourage a
        // flatter board
        int maxHeight = Arrays.stream(columnHeights).max().orElse(-1);
        int minHeight = Arrays.stream(columnHeights).min().orElse(-1);
        int heightVariance = maxHeight - minHeight;
        reward -= 0.5 * heightVariance; // Can tweak the weight as needed

        // Reinforcement 3: Penalty for exceeding the maximum stack height
        int maxColumnHeight = Arrays.stream(columnHeights).max().orElse(0);
        if (maxColumnHeight > maxAllowedHeight) {
            reward -= 100.0;
        }

        // Reinforcement 4: Encourage line clears since they earn points
        int linesCleared = board.clearFullLines().size();
        if (linesCleared == 4) { // Tetris clear
            reward += 1000;
        } else {
            reward += linesCleared * 100;
        }

        // Reinforcement 5: Penalize for enclosed spaces (holes)
        for (int col = 0; col < board.NUM_COLS; col++) {
            boolean enclosedSpace = false;
            int topRowWithBlock = board.NUM_ROWS;
            int bottomRowWithBlock = -1;

            for (int row = 0; row < board.NUM_ROWS; row++) {
                if (board.isCoordinateOccupied(col, row)) {
                    topRowWithBlock = Math.min(topRowWithBlock, row);
                    bottomRowWithBlock = Math.max(bottomRowWithBlock, row);
                }
            }

            if (topRowWithBlock < board.NUM_ROWS && bottomRowWithBlock > -1) {
                for (int row = topRowWithBlock + 1; row < bottomRowWithBlock; row++) {
                    if (!board.isCoordinateOccupied(col, row)) {
                        enclosedSpace = true;
                        break;
                    }
                }
            }

            if (enclosedSpace) {
                reward -= 50.0; // Penalize for enclosed spaces
            }
        }

        return reward;
    }

    @Override
    public void trainQFunction(Dataset dataset, LossFunction lossFunction, Optimizer optimizer, long numUpdates) {
        for (int epochIdx = 0; epochIdx < numUpdates; ++epochIdx) {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix>> batchIterator = dataset.iterator();

            while (batchIterator.hasNext()) {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(), lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }
}