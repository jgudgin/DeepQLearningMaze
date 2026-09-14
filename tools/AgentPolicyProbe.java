import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import neuralnetwork.Action;
import neuralnetwork.Agent;
import neuralnetwork.EpsilonSoft;
import neuralnetwork.Experience;
import neuralnetwork.MazeApp;
import neuralnetwork.QLearningNetwork;
import neuralnetwork.State;

// Checks that Agent.move chooses its action from the network's Q-values.
// Setup: the agent stands at column 1 row 1 with SOUTH and EAST open. A stand-in network
// returns Q = 10 for EAST and Q = 0 for SOUTH. Epsilon is 0, so every move uses softmax.
// Expectation: softmax at tau 0.5 gives EAST a probability of e^20 / (e^20 + 1), which is
// about 0.999999998, so EAST should win at least 199 of 200 moves. With no network input,
// the two actions are equally likely and EAST wins about 100 moves.
public class AgentPolicyProbe {

    private static int failures = 0;

    private static void check(String name, boolean passed) {
        System.out.println((passed ? "PASS " : "FAIL ") + name);
        if (!passed) {
            failures++;
        }
    }

    // Returns Q = 10 at EAST's output index when asked about EAST, and 0 everywhere else.
    // Records the state and action of every query.
    static class FixedNetwork extends QLearningNetwork {
        final List<String> queries = new ArrayList<>();

        FixedNetwork() {
            super(6, 4, new int[]{4}, 0.1, 0.9);
        }

        @Override
        public double[] predict(Experience experience) {
            State state = experience.getCurrentState();
            queries.add(state.getX() + "," + state.getY() + "," + experience.getAction());
            double[] output = new double[4];
            if (experience.getAction() == Action.EAST) {
                output[Action.EAST.index()] = 10.0;
            }
            return output;
        }

        @Override
        public void train(Experience experience) {
        }
    }

    // Supplies a fixed grid without opening the Swing window.
    static class FixedMaze extends MazeApp {
        private final int[][] grid;

        FixedMaze(int[][] grid) {
            this.grid = grid;
        }

        @Override
        public int[][] getMaze() {
            return grid;
        }
    }

    private static Object getField(Object target, String name) throws Exception {
        Field field = Agent.class.getDeclaredField(name);
        field.setAccessible(true);
        return field.get(target);
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field field = Agent.class.getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }

    public static void main(String[] args) throws Exception {
        // maze[row][col]. Column 1 row 1 is open, with EAST (row 1 column 2) and
        // SOUTH (row 2 column 1) open, and NORTH and WEST walls.
        int[][] grid = {
            {1, 1, 1, 1},
            {1, 0, 0, 1},
            {1, 0, 1, 1},
            {1, 1, 1, 1}
        };
        FixedMaze maze = new FixedMaze(grid);

        Agent agent = new Agent(new State(1, 1, maze), maze);
        FixedNetwork network = new FixedNetwork();
        setField(agent, "qLearningNetwork", network);
        EpsilonSoft policy = (EpsilonSoft) getField(agent, "epsilonSoft");
        policy.setEpsilon(0.0);

        // SOUTH comes first, so an agent that ignored the network index would favor SOUTH.
        List<Action> actions = new ArrayList<>();
        actions.add(Action.SOUTH);
        actions.add(Action.EAST);

        int moves = 200;
        int east = 0;
        int south = 0;
        int other = 0;
        for (int m = 0; m < moves; m++) {
            State start = new State(1, 1, maze);
            start.updateSurroundings(grid, 1, 1);
            setField(agent, "currentState", start);

            agent.move(actions);

            State after = agent.getCurrentState();
            if (after.getX() == 2 && after.getY() == 1) {
                east++;
            } else if (after.getX() == 1 && after.getY() == 2) {
                south++;
            } else {
                other++;
            }
        }

        System.out.println("moves: " + moves + " east=" + east + " south=" + south + " other=" + other);
        System.out.println("network queries: " + network.queries.size());

        boolean allFromStart = true;
        boolean askedEast = false;
        boolean askedSouth = false;
        for (String query : network.queries) {
            if (!query.startsWith("1,1,")) {
                allFromStart = false;
            }
            if (query.equals("1,1,EAST")) {
                askedEast = true;
            }
            if (query.equals("1,1,SOUTH")) {
                askedSouth = true;
            }
        }

        check("the agent queries the network", network.queries.size() > 0);
        check("every query uses the position before the move (column 1 row 1)",
                network.queries.size() > 0 && allFromStart);
        check("the agent asks about EAST", askedEast);
        check("the agent asks about SOUTH", askedSouth);
        check("every move lands on EAST or SOUTH", other == 0);
        check("EAST wins at least 199 of 200 moves", east >= 199);

        System.out.println("failures: " + failures);
        if (failures > 0) {
            System.exit(1);
        }
    }
}
