import neuralnetwork.Action;
import neuralnetwork.Experience;
import neuralnetwork.QLearningNetwork;
import neuralnetwork.State;

// Checks that training moves a prediction toward its target.
// Setup: a terminal experience (no next state) with reward 10, so the Q-learning target is exactly 10.
// Network: one input per value from State.convertToInput, hidden layers {10, 5}, 4 outputs, discount 0.9.
// Expectation: gradient descent on squared error with a small enough step never moves a prediction
// away from a fixed target. The checks use learning rate 0.01. The output layer still applies a ReLU
// derivative, which is a known bug, so a network whose starting prediction is 0 or below gets no
// update and stays where it is. Every network that starts above 0 should move closer.
// The probe also reports, without checking, what happens at the agent's learning rate of 0.1, where
// steps can jump past the target.
public class TrainingDirectionProbe {

    private static int failures = 0;

    private static void check(String name, boolean passed) {
        System.out.println((passed ? "PASS " : "FAIL ") + name);
        if (!passed) {
            failures++;
        }
    }

    static class Result {
        int closer = 0;
        int farther = 0;
        int unchanged = 0;
        int startedAboveZero = 0;
        int aboveZeroAndCloser = 0;
        int fartherAndCrossedTarget = 0;
    }

    private static Result run(double learningRate) {
        State state = new State(3, 5, null);
        Experience terminal = new Experience(state, Action.EAST, 10.0, null);
        int eastIndex = Action.EAST.index();
        double target = 10.0;
        int inputLength = state.convertToInput(Action.EAST).length;

        int trials = 200;
        int steps = 20;
        Result result = new Result();

        for (int t = 0; t < trials; t++) {
            QLearningNetwork network = new QLearningNetwork(inputLength, 4, new int[]{10, 5}, learningRate, 0.9);
            double start = network.predict(terminal)[eastIndex];
            double startDistance = Math.abs(target - start);
            double startSide = Math.signum(start - target);
            boolean crossedTarget = false;

            for (int s = 0; s < steps; s++) {
                network.train(terminal);
                double side = Math.signum(network.predict(terminal)[eastIndex] - target);
                if (side != 0 && side != startSide) {
                    crossedTarget = true;
                }
            }

            double end = network.predict(terminal)[eastIndex];
            double endDistance = Math.abs(target - end);

            if (endDistance < startDistance) {
                result.closer++;
            } else if (endDistance > startDistance) {
                result.farther++;
                if (crossedTarget) {
                    result.fartherAndCrossedTarget++;
                }
            } else {
                result.unchanged++;
            }

            if (start > 0) {
                result.startedAboveZero++;
                if (endDistance < startDistance) {
                    result.aboveZeroAndCloser++;
                }
            }
        }
        return result;
    }

    private static void print(String label, Result result) {
        System.out.println(label + ": 200 networks, 20 steps each");
        System.out.println("  closer=" + result.closer + " farther=" + result.farther + " unchanged=" + result.unchanged);
        System.out.println("  started above 0: " + result.startedAboveZero + ", of those closer: " + result.aboveZeroAndCloser);
        System.out.println("  farther and crossed the target at some step: " + result.fartherAndCrossedTarget);
    }

    public static void main(String[] args) {
        Result small = run(0.01);
        print("learning rate 0.01", small);
        check("no network moves farther from the target", small.farther == 0);
        check("at least one network moves closer", small.closer > 0);
        check("every network that starts above 0 moves closer", small.aboveZeroAndCloser == small.startedAboveZero);

        Result agentRate = run(0.1);
        print("learning rate 0.1 (the agent's rate, reported only)", agentRate);

        System.out.println("failures: " + failures);
        if (failures > 0) {
            System.exit(1);
        }
    }
}
