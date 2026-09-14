import neuralnetwork.Action;
import neuralnetwork.Experience;
import neuralnetwork.QLearningNetwork;
import neuralnetwork.State;

// Checks that training moves a prediction toward its target.
// Setup: a terminal experience (no next state) with reward 10, so the Q-learning target is exactly 10.
// Network sizes match Agent: 100 inputs, hidden layers {10, 5}, 4 outputs, learning rate 0.1, discount 0.9.
// Expectation: gradient descent on squared error with small steps never moves a prediction away
// from a fixed target. The output layer still applies a ReLU derivative, which is a known bug, so a
// network whose starting prediction is 0 or below gets no update and stays where it is. Every network
// that starts above 0 should move closer.
public class TrainingDirectionProbe {

    private static int failures = 0;

    private static void check(String name, boolean passed) {
        System.out.println((passed ? "PASS " : "FAIL ") + name);
        if (!passed) {
            failures++;
        }
    }

    public static void main(String[] args) {
        State state = new State(3, 5, null);
        Experience terminal = new Experience(state, Action.EAST, 10.0, null);
        int eastIndex = Action.EAST.index();
        double target = 10.0;

        int trials = 200;
        int steps = 20;
        int closer = 0;
        int farther = 0;
        int unchanged = 0;
        int startedAboveZero = 0;
        int aboveZeroAndCloser = 0;

        for (int t = 0; t < trials; t++) {
            QLearningNetwork network = new QLearningNetwork(100, 4, new int[]{10, 5}, 0.1, 0.9);
            double start = network.predict(terminal)[eastIndex];
            double startDistance = Math.abs(target - start);

            for (int s = 0; s < steps; s++) {
                network.train(terminal);
            }

            double end = network.predict(terminal)[eastIndex];
            double endDistance = Math.abs(target - end);

            if (endDistance < startDistance) {
                closer++;
            } else if (endDistance > startDistance) {
                farther++;
            } else {
                unchanged++;
            }

            if (start > 0) {
                startedAboveZero++;
                if (endDistance < startDistance) {
                    aboveZeroAndCloser++;
                }
            }
        }

        System.out.println("networks: " + trials + ", steps each: " + steps);
        System.out.println("closer=" + closer + " farther=" + farther + " unchanged=" + unchanged);
        System.out.println("started above 0: " + startedAboveZero + ", of those closer: " + aboveZeroAndCloser);

        check("no network moves farther from the target", farther == 0);
        check("at least one network moves closer", closer > 0);
        check("every network that starts above 0 moves closer", aboveZeroAndCloser == startedAboveZero);

        System.out.println("failures: " + failures);
        if (failures > 0) {
            System.exit(1);
        }
    }
}
