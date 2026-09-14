package neuralnetwork;

import java.util.List;
import java.util.ArrayList;
import java.util.Scanner;

public class Agent {

    private State currentState;
    private ExperienceReplay experienceReplay;
    private double totalReward;
    private EpsilonSoft epsilonSoft;
    private double[] qValues;
    private MazeApp mazeApp;
    private QLearningNetwork qLearningNetwork;
    private Scanner input;

    private int batchSize = 32; //amount of experiences to accumulate before beginning training
    private int bufferLimit = 1000;  //max amount of experiences kept in the replay buffer to be sampled
    
    int inputSize = 100;    //amount of neurons in input layer
    int outputSize = 4;     //amount of neurons in output layer
    int[] hiddenSizes = {10, 5};    //amount of neurons in each hidden layer
    double learningRate = 0.1;  //weight that new information has on known information
    double discountFactor = 0.9;    //reduces reward after each time step
    double epsilon = 1.0;   //exporation rate
    double minEpsilon = 0.1;    //minimum exploration rate
    double decayRate = 0.999;   //decay rate of epsilon after certain amount of moves
    double tau = 0.5;   //temperature parameter for softmax

    //amount of valid moves before epsilon is adjusted
    int movesBeforeDecay = 20;
    
    //initial state of move counter
    int moveCounter = 0;

    public Agent(State initialState, MazeApp mazeApp) {
        this.currentState = initialState;
        this.experienceReplay = new ExperienceReplay(bufferLimit);
        this.epsilonSoft = new EpsilonSoft(epsilon, tau);
        this.qLearningNetwork = new QLearningNetwork(inputSize, outputSize, hiddenSizes, learningRate, discountFactor);
        this.totalReward = 0.0;
        this.mazeApp = mazeApp;
    }

    public State getCurrentState() {
        return currentState;
    }

    public void move(List<Action> actions) throws InterruptedException {

        //create a new array to store q values for available actions
        qValues = new double[actions.size()];

        //use epsilon soft policy to select an action
        Action action = epsilonSoft.selectAction(qValues, actions);

        int[][] maze = mazeApp.getMaze();

        //flag for checking if a certain move is valid
        boolean isNextActionValid = currentState.isPath(action);

        //check if the next state is valid
        if (isNextActionValid) {

            //update the current state to the new valid state
            currentState = getCurrentState().getNextState(action);

            //also update the surroundings for the updated current state
            currentState.updateSurroundings(mazeApp.getMaze(), currentState.getX(), currentState.getY());


            //calculate the reward based on the move
            double reward = calculateReward(action);

            //create a new experience
            Experience experience = new Experience(currentState, action, reward, currentState);
            experienceReplay.addExperience(experience);

            //update the total reward
            totalReward += reward;

            //train the neural network with a batch of experiences
            trainWithBatch();

            //increment move counter and check if it's time to decay epsilon
            moveCounter++;
            if (moveCounter >= movesBeforeDecay) {
                if (this.epsilonSoft.getEpsilon() > minEpsilon) {
                    this.epsilonSoft.setEpsilon(this.epsilonSoft.getEpsilon() * decayRate);
                }
                moveCounter = 0; //reset the counter
            }
        } else {
            System.out.println("Invalid move, " + action.toString() + " is blocked");
        }
    }

    private void trainWithBatch() {
        //check if there are enough experiences in the replay buffer
        if (experienceReplay.getBufferSize() < batchSize) {
            return; //not enough experiences to train
        }

        List<Experience> batch = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            Experience experience = experienceReplay.sampleExperience();
            if (experience != null && !batch.contains(experience)) {
                batch.add(experience);
            }
        }

        //train the q learning network with the sampled batch of experiences
        int i = 1;
        for (Experience experience : batch) {
            qLearningNetwork.train(experience);
            i++;
        }
    }

    public double calculateReward(Action action) {
        int[] goalPos = mazeApp.getEndPosition();
        int currentX = currentState.getX();
        int currentY = currentState.getY();

        //calculate current Manhattan distance to the goal
        int currentDistance = Math.abs(currentX - goalPos[0]) + Math.abs(currentY - goalPos[1]);

        //high reward for reaching the goal
        if (currentX == goalPos[0] && currentY == goalPos[1]) {
            return 5000; // High reward for reaching the goal
        }

        //if the agent is at a dead end then deduct points
        if (currentState.countBlockedDirections() == 3) {
            return 1.0 / (currentDistance + 1) * 2;
        }

        //intermediate reward based on distance
        double distanceReward = 1.0 / (currentDistance + 1); //reward increases as distance decreases

        //small penalty for each action taken
        double stepPenalty = -10;

        return distanceReward + stepPenalty;
    }

    public void setTotalReward(int totalReward){
        this.totalReward = totalReward;
    }
}
