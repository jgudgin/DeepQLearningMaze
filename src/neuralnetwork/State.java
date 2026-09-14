package neuralnetwork;

import java.util.Map;
import java.util.HashMap;

//state object holding information about the agents current co-ordinates and surroundings
//includes method for getting the next state after an action
public class State {

    private int x;
    private int y;
    private Map<Action, Surrounding> surroundings; //maps each action to either WALL or PATH

    public static final int NORTH = 0;
    public static final int SOUTH = 1;
    public static final int EAST = 2;
    public static final int WEST = 3;

    //constructor for the coordinates and the surroundings
    public State(int x, int y, int[][] maze) {
        this.x = x;
        this.y = y;
        this.surroundings = getSurroundings(x, y, maze);
    }

    public enum Surrounding {
        WALL, PATH
    }

    //getters
    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public Map<Action, Surrounding> getSurroundings() {
        return surroundings;
    }

    //checks if a certain direction is a path or not
    public boolean isPath(Action action) {
        return surroundings.get(action) == Surrounding.PATH;
    }

    //get the next state based on the action
    public State getNextState(Action action, int[][] maze) {
        if (!isPath(action)) {
            return null;    //return null is the directions is a wall
        }

        //calculate new position
        int newX = x + action.getDeltaRow();
        int newY = y + action.getDeltaCol();

        //create new state with updated surroundings
        return new State(newX, newY, maze);

    }

    //method for encoding State coordinates and Action direction values into an array
    public double[] convertToInput(Action action) {
        double[] stateInput = new double[]{this.getX(), this.getY()};
        double[] actionInput = action.convertToInput();

        double[] combinedInput = new double[stateInput.length + actionInput.length];
        System.arraycopy(stateInput, 0, combinedInput, 0, stateInput.length);
        System.arraycopy(actionInput, 0, combinedInput, stateInput.length, actionInput.length);

        return combinedInput;
    }

    public static Map<Action, Surrounding> getSurroundings(int x, int y, int[][] maze) {
        Map<Action, Surrounding> surroundings = new HashMap<>();

        // Check each direction
        if (x > 0) { // North
            surroundings.put(Action.NORTH, maze[x - 1][y] == 1 ? Surrounding.WALL : Surrounding.PATH);
        }
        if (x < maze.length - 1) { // South
            surroundings.put(Action.SOUTH, maze[x + 1][y] == 1 ? Surrounding.WALL : Surrounding.PATH);
        }
        if (y > 0) { // West
            surroundings.put(Action.WEST, maze[x][y - 1] == 1 ? Surrounding.WALL : Surrounding.PATH);
        }
        if (y < maze[0].length - 1) { // East
            surroundings.put(Action.EAST, maze[x][y + 1] == 1 ? Surrounding.WALL : Surrounding.PATH);
        }

        return surroundings;
    }
}
