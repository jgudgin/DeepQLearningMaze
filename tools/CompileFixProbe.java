import neuralnetwork.Action;
import neuralnetwork.State;

// Checks the State movement and Action encoding against expectations that do not come
// from the code under test.
// Movement: State.getSurroundings checks maze[x + 1][y] for SOUTH (State.java:83),
// so a SOUTH move must land on the cell that check looked at.
// Encoding: the Action class at commit 29abb51 returned {0, 0, 1, 0} for EAST and
// {1, 0, 0, 0} for NORTH.
public class CompileFixProbe {

    private static int failures = 0;

    public static void main(String[] args) {
        int[][] maze = {
            {1, 1, 1, 1},
            {1, 0, 1, 1},
            {1, 0, 1, 1},
            {1, 1, 1, 1}
        };

        State start = new State(1, 1, maze);

        State south = start.getNextState(Action.SOUTH, maze);
        check("SOUTH from (1,1) is not null", south != null);
        if (south != null) {
            check("SOUTH from (1,1) lands on row 2", south.getX() == 2);
            check("SOUTH from (1,1) stays on column 1", south.getY() == 1);
        }

        check("NORTH from (1,1) is a wall", start.getNextState(Action.NORTH, maze) == null);
        check("EAST from (1,1) is a wall", start.getNextState(Action.EAST, maze) == null);

        double[] east = Action.EAST.convertToInput();
        check("EAST encoding has length 4", east.length == 4);
        check("EAST encoding is {0, 0, 1, 0}",
                east[0] == 0.0 && east[1] == 0.0 && east[2] == 1.0 && east[3] == 0.0);

        double[] north = Action.NORTH.convertToInput();
        check("NORTH encoding is {1, 0, 0, 0}",
                north[0] == 1.0 && north[1] == 0.0 && north[2] == 0.0 && north[3] == 0.0);

        double[] input = start.convertToInput(Action.EAST);
        check("State input for (1,1) EAST is {1, 1, 0, 0, 1, 0}",
                input.length == 6
                && input[0] == 1.0 && input[1] == 1.0
                && input[2] == 0.0 && input[3] == 0.0
                && input[4] == 1.0 && input[5] == 0.0);

        System.out.println("failures: " + failures);
        if (failures > 0) {
            System.exit(1);
        }
    }

    private static void check(String name, boolean passed) {
        if (passed) {
            System.out.println("PASS " + name);
        } else {
            System.out.println("FAIL " + name);
            failures++;
        }
    }
}
