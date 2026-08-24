package neuralnetwork;

//action class holding information about the types of actions available for the game
public enum Action {
    NORTH(-1, 0),
    SOUTH(1,0),
    EAST(0,1),
    WEST(0,-1);

    //how many actions there are / size of the network output layer
    public static final int COUNT = values().length;

    private final int deltaRow;
    private final int deltaCol;

    Action(int deltaRow, int deltaCol){
        this.deltaRow = deltaRow;
        this.deltaCol = deltaCol;
    }

    public int getDeltaRow() {
        return deltaRow;
    }

    public int getDeltaCol() {
        return deltaCol;
    }

    //the position of this action in the networks output array
    //ensures that the moves are correctly mapped to what the agent choses
    public int index() {
        return ordinal();
    }
}