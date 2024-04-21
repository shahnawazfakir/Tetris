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
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;
import edu.bu.tetris.linalg.Shape;
import edu.bu.tetris.utils.Coordinate;


//line clear, tetris, perfect clear, t spin, double t spin get u scores/points
// q function hiddem layers: can do big->small->big, or 3x->2x->1x->output
//csa secify >1300 buffer size
//100 train games, 50 eval games, 5000 cycles.
public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        // bigger to smaller vector space - means you are compressing it. vice versa - means more room to play around
        // can use relu but it prevent neg vals

        // //final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        // final int numFeatures = Board.NUM_COLS * 3; 
        // final int hiddenDim = 2 * numFeatures; 
        // //final int hiddenDim = 2 * numPixelsInImage;
        // final int outDim = 1;

        // Sequential qFunction = new Sequential();
        // //qFunction.add(new Dense(numPixelsInImage, hiddenDim));
        // qFunction.add(new Dense(numFeatures, hiddenDim));
        // qFunction.add(new Tanh());
        // qFunction.add(new Dense(hiddenDim, outDim));

        // return qFunction;
        final int numFeatures = Board.NUM_COLS * 3; 
        final int hiddenDim1 = 3 * numFeatures; // Biggest hidden layer
        final int hiddenDim2 = 2 * numFeatures; // Smaller hidden layer
        final int hiddenDim3 = numFeatures;     // Even smaller hidden layer
        final int outDim = 1;

        Sequential qFunction = new Sequential();
        qFunction.add(new Dense(numFeatures, hiddenDim1)); // Biggest hidden layer
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim1, hiddenDim2));  // Smaller hidden layer
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim2, hiddenDim3));  // Even smaller hidden layer
        qFunction.add(new Tanh());
        qFunction.add(new Dense(hiddenDim3, outDim));      // Output layer

        return qFunction;
    }

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for? e.g height of each col, argmin, agrmax, number of open spaces/holes for each col
     */
    //board has 10 cols, 22 rows
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        Matrix flattenedImage = null;
        Matrix tetrisBoardMatrix = null;
        try
        {
            flattenedImage = game.getGrayscaleImage(potentialAction).flatten();
            tetrisBoardMatrix = game.getGrayscaleImage(potentialAction);
            //System.out.println("Flattened Image Vector: " + flattenedImage);
            //System.out.println("potentialAction " + potentialAction); //gives you the current mino type
            ////System.out.println("next 3 mino types " + game.getNextThreeMinoTypes()); 

            ////System.out.println("tetrisBoardMatrix: "+ tetrisBoardMatrix); //top to bottom: row 0 to 21, left to right, cols 0 to 9
            //System.out.println("numRows " + tetrisBoardMatrix.getShape().getNumRows());
            //System.out.println("numCols " + tetrisBoardMatrix.getShape().getNumCols());
        } catch(Exception e)
        {
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
            int lastMinoPosition = -1;  // Initialize to -1, indicating no mino found yet. Bottommost mino

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
            int numOpenBlocks =  firstMinoPosition == -1 ? (numRows-1) : (firstMinoPosition-1);//num of open blocks in this column, excluding the top row. Can modify this to encode open space for T spins etc (e.g. even when a column appears "blocked")

            // Add the features to the row vector
            featureVector.set(0, index++, firstMinoPosition); // Add the position of the first mino
            featureVector.set(0, index++, lastMinoPosition);  // Add the position of the last mino
            featureVector.set(0, index++, numOpenBlocks);
        }

        ////System.out.println("featureVector " +featureVector); 
        
        return featureVector;// flattenedImage; //can change dimension of this row vector but need to update dimension of vector above as well
    }

    
    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    // the probability of exploration exponentially decays with time as the agent learns more info and the need to explore decreases
    private int explorationStep = 0;
    private int explorationDecaySteps = 1000; // Number of steps over which exploration probability decays
    private double initialExplorationProb = 0.5; // Initial exploration probability

    @Override
    public boolean shouldExplore(final GameView game, final GameCounter gameCounter) {
        // Increment the step count
        explorationStep++;

        // Calculate the current exploration probability based on decay
        double currentExplorationProb = initialExplorationProb * Math.exp(-explorationStep / (double) explorationDecaySteps);

        // Generate a random number between 0 and 1
        double randomValue = this.getRandom().nextDouble();

        // Check if the random value is less than or equal to the current exploration probability
        if (randomValue <= currentExplorationProb) {
            // Explore: return true to indicate exploration
            return true;
        } else {
            // Exploit: return false to indicate exploitation
            return false;
        }
    }


    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        
        List<Mino> availableActions = game.getFinalMinoPositions();
    
        // Calculate the current height of each column on the board
        int[] columnHeights = new int[game.getBoard().NUM_COLS];
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

        // System.out.println("minHeight "+ minHeight);
        
        // Find the actions that result in placing a piece in columns with the minimum height
        List<Mino> minHeightActions = new ArrayList<>();
        for (Mino action : availableActions) {
            Coordinate pivot = action.getPivotBlockCoordinate();
            // System.out.println("pivot "+ pivot);
            int pivotColumn = pivot.getXCoordinate(); // Get the column of the pivot block
            
            if (columnHeights[pivotColumn] == minHeight) {
                minHeightActions.add(action);
                // System.out.println("mino available at min height"+ pivot);
            }
        }

        // If there are actions available at the minimum height columns, choose one (mino) randomly
        if (!minHeightActions.isEmpty()) {
            int randIdx = this.getRandom().nextInt(minHeightActions.size());
            // System.out.println("chosen action "+ minHeightActions.get(randIdx).getPivotBlockCoordinate());
            return minHeightActions.get(randIdx);
        }
        
        // If no actions are available at the minimum height columns, choose randomly from all available actions
        int randIdx = this.getRandom().nextInt(availableActions.size());
        return availableActions.get(randIdx);
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    // this method is called *after* a turning i.e after placing a mino

    // Strategy to implement: flatter boards are better. Avoid long towers. Special case: an 'I' hole
    @Override
    public double getReward(final GameView game)
    {
        // System.out.println("-----------------------------------------------------------------------------------");
        // System.out.println("Score this turn: " + game.getScoreThisTurn());
        // System.out.println("-----------------------------------------------------------------------------------");

        // initially assing score of this turn to reward
        double reward = game.getScoreThisTurn() * 1000; //times 1000 becasue we want to highly reward getting a score, since achieving a score is rare
        Board board = game.getBoard();
        int maxAllowedHeight = 18;

        int[] columnHeights = new int[board.NUM_COLS]; // Array to store column heights
    
        // Iterate over each column
        for (int col = 0; col < board.NUM_COLS; col++) {
            int height = 0;
            
            // Iterate over each row in the column from bottom to top
            for (int row = board.NUM_ROWS - 1; row >= 0; row--) {
                // Check if the coordinate is occupied by a block
                if (board.isCoordinateOccupied(col, row)) {
                    height = board.NUM_ROWS - row; // Calculate the height of the column
                    break; // Exit the loop once the first occupied block is found
                }
            }
            columnHeights[col] = height;
        }

        // Reinforcement 1: Penalize for higher stack of minos
        for (int height : columnHeights) {
            reward -= 0.1 * height; // Penalize for each block height in a column
        }

        // Reinforcement 2: Penalize for height variance across columns to encourage a flatter board
        // This can help in preventing situations where a few columns are significantly higher than others
        int maxHeight = Arrays.stream(columnHeights).max().orElse(-1);
        int minHeight = Arrays.stream(columnHeights).min().orElse(-1);
        int heightVariance = maxHeight - minHeight;
        reward -= 0.5 * heightVariance; // can tweak the weight as needed. 

        // Reinforcement 3: Penalty for exceeding the maximum stack height
        // Calculate the overall stack height using columnHeights
        int maxColumnHeight = 0;
        for (int height : columnHeights) {
            maxColumnHeight = Math.max(maxColumnHeight, height);
        }
        if (maxColumnHeight > maxAllowedHeight){
            //System.out.println("MAX HEIGHT EXCEEDED");
            reward -= 100.0;
        }

        // Reinforcement 4: encourage line clears since they earn points
        int linesCleared = board.clearFullLines().size(); // Get the number of lines cleared
        ////System.out.println("linesCleared "+linesCleared);
        if (linesCleared == 4){ //if its a tetris clear
            reward += 1000;
        }
        else {
            reward += linesCleared * 100; 
        }

        // penalize hollow enclosed spaces
        ////System.out.println("REWARD: "+ reward);


        return reward;
    }

}



// System.out.println("numRows " + featureVector.getShape().getNumRows());
// System.out.println("numCols " + featureVector.getShape().getNumCols());
// try 
// {
//     Matrix featureVectorFlattened = featureVector.flatten();
//     System.out.println("featureVectorFlattened " +featureVectorFlattened); 
//     System.out.println("numRows " + featureVectorFlattened.getShape().getNumRows());
//     System.out.println("numCols " + featureVectorFlattened.getShape().getNumCols());
// }
// catch(Exception e) {
//     e.printStackTrace();
//     System.exit(-1);
// }