package neuralnetwork;

import javax.swing.*;
import java.awt.*;
import java.util.Random;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

// Class for generating a maze
public class MazeApp {

    private static final int GRID_SIZE = 16; // grid size
    private static final int CELL_SIZE = 30; //pixel size for each cell
    private int[][] maze; //maze is 2D array
    private JPanel gridPanel; //panel containing the grid
    private Agent agent;
    private List<Action> availableMoves = new ArrayList<>();
    private Timer gameTimer;
    long start = System.nanoTime();

    private int episodeAmount = 100; //amount of episodes to play through
    private int episodeNum = 0;

    private int moveCounter;

    private final int[] startPosition = {1, 1};
    private final int[] endPosition = {GRID_SIZE - 3, GRID_SIZE - 3};

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new MazeApp().createAndShowGUI());
    }

    private void createAndShowGUI() {
        JFrame frame = new JFrame("Maze Game");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        gridPanel = new JPanel(new GridLayout(GRID_SIZE, GRID_SIZE));
        gridPanel.setPreferredSize(new Dimension(GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE));

        //initialize the maze with walls
        maze = generateMaze();

        //create cells based on the maze array
        for (int row = 0; row < GRID_SIZE; row++) {
            for (int col = 0; col < GRID_SIZE; col++) {
                JPanel cell = new JPanel();
                cell.setPreferredSize(new Dimension(CELL_SIZE, CELL_SIZE));

                //determine wall or path based on the maze array
                if (maze[row][col] == 1) {
                    cell.setBackground(Color.BLACK); //wall
                } else {
                    cell.setBackground(Color.WHITE); //path
                }

                //set start and end point colors
                if (row == startPosition[0] && col == startPosition[1]) {
                    cell.setBackground(Color.GREEN); //start
                } else if (row == endPosition[0] && col == endPosition[1]) {
                    cell.setBackground(Color.RED); //end
                }

                gridPanel.add(cell);
            }
        }

        State initialState = new State(startPosition[0], startPosition[1], this);
        agent = new Agent(initialState, this);

        //add the grid panel to the frame
        frame.add(gridPanel);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        gameTimer = new Timer(0, e -> {
            try {
                startGameLoop();
            } catch (InterruptedException ex) {
                Logger.getLogger(MazeApp.class.getName()).log(Level.SEVERE, null, ex);
            }
        });
        gameTimer.start();
    }

    private int[][] generateMaze() {
        int[][] maze = new int[GRID_SIZE][GRID_SIZE];

        //initialize all the cells as wall
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                maze[i][j] = 1; //wall
            }
        }

        //start carving paths from the first cell
        carvePath(maze, 1, 1);

        //set start and end points
        maze[startPosition[0]][startPosition[1]] = 0; //start coordinates
        maze[endPosition[0]][endPosition[1]] = 0; //goal coordinates

        return maze;
    }

    //depth-first search maze generation
    private void carvePath(int[][] maze, int x, int y) {
        //define directions: (dx, dy)
        int[][] directions = {
            {2, 0}, //east
            {-2, 0}, //west
            {0, 2}, //south
            {0, -2} //north
        };

        //shuffle directions to ensure random carving
        Random random = new Random();
        for (int i = 0; i < directions.length; i++) {
            int j = random.nextInt(directions.length);
            //swap
            int[] temp = directions[i];
            directions[i] = directions[j];
            directions[j] = temp;
        }

        //carve paths in each direction
        for (int[] dir : directions) {
            int newX = x + dir[0];
            int newY = y + dir[1];

            //check if the new coordinates are within bounds and if the cell has not been visited
            if (newX > 0 && newX < GRID_SIZE - 1 && newY > 0 && newY < GRID_SIZE - 1 && maze[newX][newY] == 1) {
                //carve a path between the current cell and the new cell
                maze[x + dir[0] / 2][y + dir[1] / 2] = 0; //remove wall between
                maze[newX][newY] = 0; //carve the new cell

                //recursively carve from the new cell
                carvePath(maze, newX, newY);
            }
        }
    }

    //getters for start and end position (for agent reward calculation)
    public int[] getStartPosition() {
        return startPosition;
    }

    public int[] getEndPosition() {
        return endPosition;
    }

    //getter for entire 2D maze array
    public int[][] getMaze() {
        return maze;
    }

    public void startGameLoop() throws InterruptedException {

        agent.getCurrentState().updateSurroundings(maze, agent.getCurrentState().getX(), agent.getCurrentState().getY());
        //check if the agent has reached the end position
        int[] currentPosition = {agent.getCurrentState().getX(), agent.getCurrentState().getY()};
        if (currentPosition[0] == endPosition[0] && currentPosition[1] == endPosition[1]) {
            if (episodeNum == episodeAmount) {
                //stop the game loop
                gameTimer.stop();
                JOptionPane.showMessageDialog(gridPanel, "Agent has finished training");
                return;
            }
            long end = System.nanoTime();

            double elapsedTime = (end - start) / 1_000_000_000.0;
            System.out.printf("Episode %d complete after %.3f seconds with %d moves%n", episodeNum, elapsedTime, moveCounter);

            episodeNum++;
            moveCounter = 0;
            start = System.nanoTime();
            agent.getCurrentState().setCurrentState(startPosition[0], startPosition[1]);
            agent.getCurrentState().updateSurroundings(maze, startPosition[0], startPosition[1]);
            agent.setTotalReward(0);    //reset total reward

        }

        availableMoves.clear();
        //find all available moves based on surroundings
        for (Map.Entry<Action, State.Surrounding> entry : agent.getCurrentState().getSurroundings().entrySet()) {
            Action action = entry.getKey();
            State.Surrounding surrounding = entry.getValue();

            if (surrounding == State.Surrounding.PATH) {
                availableMoves.add(action); //add the action to the list if its a path
            }
        }

        //store the current position before the agent moves
        int oldX = agent.getCurrentState().getX();
        int oldY = agent.getCurrentState().getY();

        //let agent choose a move based on its current knowledge
        agent.move(availableMoves);
        
        moveCounter++;

        //get the new position after the move
        int newX = agent.getCurrentState().getX();
        int newY = agent.getCurrentState().getY();

        //update the visual representation
        updateAgentPosition(oldX, oldY, newX, newY);

    }

    private void updateAgentPosition(int oldX, int oldY, int newX, int newY) {
        //reset the old position back to path color (white)
        JPanel oldCell = (JPanel) gridPanel.getComponent(oldY * GRID_SIZE + oldX);
        oldCell.setBackground(Color.WHITE); //reset to path color

        //set the new position color to indicate the agent's current position
        JPanel newCell = (JPanel) gridPanel.getComponent(newY * GRID_SIZE + newX);
        newCell.setBackground(Color.BLUE); //change color to indicate agent's position

        //repaint the grid to reflect changes
        gridPanel.repaint();
    }

}
