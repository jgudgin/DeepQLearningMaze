import neuralnetwork.Action;
import neuralnetwork.State;

// Checks the State movement and Action encoding against expectations that do not come
// from the code under test.
// Movement: MazeApp.java:53 draws maze[row][col], so SOUTH must move down one row and
// land on an open cell.
// Encoding: the Action class at commit 29abb51 returned {0, 0, 1, 0} for EAST and
// {1, 0, 0, 0} for NORTH, and its values() returned NORTH, SOUTH, EAST, WEST.
public class CompileFixProbe {

    private static int failures = 0;

    public static void main(String[] args) {
        // Row 1 column 1 is open. Row 2 column 1 is open. Row 1 column 2 is a wall.
        int[][] maze = {
            {1, 1, 1, 1},
            {1, 0, 1, 1},
            {1, 0, 1, 1},
            {1, 1, 1, 1}
        };

        // x is the column and y is the row.
        State start = new State(1, 1, null);
        start.updateSurroundings(maze, 1, 1);

        check("SOUTH from row 1 column 1 is a path", start.isPath(Action.SOUTH));
        check("NORTH from row 1 column 1 is a wall", !start.isPath(Action.NORTH));
        check("EAST from row 1 column 1 is a wall", !start.isPath(Action.EAST));

        State south = start.getNextState(Action.SOUTH);
        check("SOUTH from row 1 column 1 is not null", south != null);
        if (south != null) {
            check("SOUTH moves to row 2", south.getY() == 2);
            check("SOUTH stays on column 1", south.getX() == 1);
            check("SOUTH lands on an open cell", maze[south.getY()][south.getX()] == 0);
        }

        check("EAST from row 1 column 1 returns null", start.getNextState(Action.EAST) == null);

        Action[] actions = Action.values();
        check("Action.values() has 4 actions", actions.length == 4);
        check("Action.values() order is NORTH, SOUTH, EAST, WEST",
                actions[0] == Action.NORTH && actions[1] == Action.SOUTH
                && actions[2] == Action.EAST && actions[3] == Action.WEST);

        double[] east = Action.EAST.convertToInput();
        check("EAST encoding has length 4", east.length == 4);
        check("EAST encoding is {0, 0, 1, 0}",
                east[0] == 0.0 && east[1] == 0.0 && east[2] == 1.0 && east[3] == 0.0);

        double[] north = Action.NORTH.convertToInput();
        check("NORTH encoding is {1, 0, 0, 0}",
                north[0] == 1.0 && north[1] == 0.0 && north[2] == 0.0 && north[3] == 0.0);

        double[] input = start.convertToInput(Action.EAST);
        check("State input for column 1 row 1 EAST is {1, 1, 0, 0, 1, 0}",
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
