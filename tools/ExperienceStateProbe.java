import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import neuralnetwork.Action;
import neuralnetwork.Agent;
import neuralnetwork.Experience;
import neuralnetwork.ExperienceReplay;
import neuralnetwork.MazeApp;
import neuralnetwork.State;

// Checks the states stored in each experience after Agent.move.
// Expectation: Q-learning stores (s, a, r, s'), where s is the position the agent acted from and
// s' is the position the action led to. The agent starts at column 1 row 1 in an open grid, moves
// EAST to column 2 row 1, then moves SOUTH to column 2 row 2. Each move offers only one action, so
// the choice does not depend on the network.
public class ExperienceStateProbe {

    private static int failures = 0;

    private static void check(String name, boolean passed) {
        System.out.println((passed ? "PASS " : "FAIL ") + name);
        if (!passed) {
            failures++;
        }
    }

    // Supplies a fixed grid without opening the Swing window.
    static class OpenMaze extends MazeApp {
        private final int[][] grid;

        OpenMaze(int[][] grid) {
            this.grid = grid;
        }

        @Override
        public int[][] getMaze() {
            return grid;
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
        OpenMaze maze = new OpenMaze(grid);

        State start = new State(1, 1, maze);
        start.updateSurroundings(grid, 1, 1);
        Agent agent = new Agent(start, maze);

        List<Action> east = new ArrayList<>();
        east.add(Action.EAST);
        agent.move(east);

        List<Action> south = new ArrayList<>();
        south.add(Action.SOUTH);
        agent.move(south);

        Field replayField = Agent.class.getDeclaredField("experienceReplay");
        replayField.setAccessible(true);
        ExperienceReplay replay = (ExperienceReplay) replayField.get(agent);
        Field bufferField = ExperienceReplay.class.getDeclaredField("replayBuffer");
        bufferField.setAccessible(true);
        List<Experience> buffer = (List<Experience>) bufferField.get(replay);

        System.out.println("stored experiences: " + buffer.size());
        check("two experiences are stored", buffer.size() == 2);
        if (buffer.size() < 2) {
            System.out.println("failures: " + failures);
            System.exit(1);
        }

        Experience first = buffer.get(0);
        Experience second = buffer.get(1);
        System.out.println("first experience: current " + position(first.getCurrentState()) + ", action "
                + first.getAction() + ", next " + position(first.getNextState()));
        System.out.println("second experience: current " + position(second.getCurrentState()) + ", action "
                + second.getAction() + ", next " + position(second.getNextState()));

        check("first experience action is EAST", first.getAction() == Action.EAST);
        check("first experience current state is column 1 row 1, where the agent acted",
                at(first.getCurrentState(), 1, 1));
        check("first experience next state is column 2 row 1, where EAST led",
                at(first.getNextState(), 2, 1));

        check("second experience action is SOUTH", second.getAction() == Action.SOUTH);
        check("second experience current state is column 2 row 1, where the agent acted",
                at(second.getCurrentState(), 2, 1));
        check("second experience next state is column 2 row 2, where SOUTH led",
                at(second.getNextState(), 2, 2));

        System.out.println("failures: " + failures);
        if (failures > 0) {
            System.exit(1);
        }
    }
}
