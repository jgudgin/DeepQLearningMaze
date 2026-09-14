import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import neuralnetwork.Action;
import neuralnetwork.Agent;
import neuralnetwork.Experience;
import neuralnetwork.ExperienceReplay;
import neuralnetwork.MazeApp;
import neuralnetwork.State;

// Checks that a move into the goal is stored as a terminal experience.
// Expectation: the episode ends at the goal, so the experience for the move into the goal must have
// no next state. QLearningNetwork.train treats a null next state as terminal and uses the reward
// alone as the target. An ordinary move must keep its real next state.
// Setup: an open grid with the goal at column 2 row 2. The agent moves SOUTH from column 1 row 1 to
// column 1 row 2, then EAST into the goal. The probe then resets the agent's state in place, as
// MazeApp.startGameLoop does at the end of an episode, and checks that no stored experience changed.
public class GoalTerminalProbe {

    private static int failures = 0;

    private static void check(String name, boolean passed) {
        System.out.println((passed ? "PASS " : "FAIL ") + name);
        if (!passed) {
            failures++;
        }
    }

    // Supplies a fixed grid and goal without opening the Swing window.
    static class SmallMaze extends MazeApp {
        private final int[][] grid;
        private final int[] goal;

        SmallMaze(int[][] grid, int[] goal) {
            this.grid = grid;
            this.goal = goal;
        }

        @Override
        public int[][] getMaze() {
            return grid;
        }

        @Override
        public int[] getEndPosition() {
            return goal;
        }
    }

    private static String position(State state) {
        if (state == null) {
            return "null";
        }
        return "column " + state.getX() + " row " + state.getY();
    }

    private static boolean at(State state, int column, int row) {
        return state != null && state.getX() == column && state.getY() == row;
    }

    private static void printExperiences(List<Experience> buffer) {
        for (int i = 0; i < buffer.size(); i++) {
            Experience experience = buffer.get(i);
            System.out.println("  experience " + i + ": current " + position(experience.getCurrentState())
                    + ", action " + experience.getAction()
                    + ", reward " + experience.getRewardReceived()
                    + ", next " + position(experience.getNextState()));
        }
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        // maze[row][col], with every cell inside the border open.
        int[][] grid = {
            {1, 1, 1, 1, 1},
            {1, 0, 0, 0, 1},
            {1, 0, 0, 0, 1},
            {1, 0, 0, 0, 1},
            {1, 1, 1, 1, 1}
        };
        SmallMaze maze = new SmallMaze(grid, new int[]{2, 2});

        State start = new State(1, 1, maze);
        start.updateSurroundings(grid, 1, 1);
        Agent agent = new Agent(start, maze);

        List<Action> south = new ArrayList<>();
        south.add(Action.SOUTH);
        agent.move(south);

        List<Action> east = new ArrayList<>();
        east.add(Action.EAST);
        agent.move(east);

        Field replayField = Agent.class.getDeclaredField("experienceReplay");
        replayField.setAccessible(true);
        ExperienceReplay replay = (ExperienceReplay) replayField.get(agent);
        Field bufferField = ExperienceReplay.class.getDeclaredField("replayBuffer");
        bufferField.setAccessible(true);
        List<Experience> buffer = (List<Experience>) bufferField.get(replay);

        System.out.println("after the two moves:");
        printExperiences(buffer);
        check("two experiences are stored", buffer.size() == 2);
        if (buffer.size() < 2) {
            System.out.println("failures: " + failures);
            System.exit(1);
        }

        Experience ordinary = buffer.get(0);
        Experience goal = buffer.get(1);

        check("the SOUTH move keeps its real next state, column 1 row 2", at(ordinary.getNextState(), 1, 2));
        check("the EAST move into the goal starts from column 1 row 2", at(goal.getCurrentState(), 1, 2));
        check("the EAST move into the goal has no next state", goal.getNextState() == null);
        check("the EAST move into the goal gets the goal reward of 5000", goal.getRewardReceived() == 5000.0);

        State agentState = agent.getCurrentState();
        agentState.setCurrentState(1, 1);
        agentState.updateSurroundings(grid, 1, 1);

        System.out.println("after resetting the agent's state in place:");
        printExperiences(buffer);
        check("after the reset, the SOUTH move still runs from column 1 row 1 to column 1 row 2",
                at(ordinary.getCurrentState(), 1, 1) && at(ordinary.getNextState(), 1, 2));
        check("after the reset, the goal move still runs from column 1 row 2 with no next state",
                at(goal.getCurrentState(), 1, 2) && goal.getNextState() == null);

        System.out.println("failures: " + failures);
        if (failures > 0) {
            System.exit(1);
        }
    }
}
