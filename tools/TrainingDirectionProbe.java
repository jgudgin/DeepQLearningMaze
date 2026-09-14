import neuralnetwork.Action;
import neuralnetwork.Experience;
import neuralnetwork.QLearningNetwork;
import neuralnetwork.State;

// Checks that training moves a prediction toward its target without overshooting.
// Setup: a terminal experience (no next state) with reward 10, so the Q-learning target is exactly 10.
// Network: one input per value from State.convertToInput, hidden layers {10, 5}, 4 outputs, discount 0.9.
// Expectation: with coordinates scaled to between 0 and 1, gradient descent on squared error at the
// agent's learning rate of 0.01 never moves a prediction away from a fixed target of 10, whatever the
// starting prediction. A scratchpad comparison of scaled and raw coordinates at this rate found no
// overshoot with scaled coordinates and 10 of 200 networks overshooting with raw ones.
// The probe also reports, without checking, what happens at learning rate 0.1.
public class TrainingDirectionProbe {

    private static int failures = 0;

    private static void check(String name, boolean passed) {
        System.out.println((passed ? "PASS " : "FAIL ") + name);
        if (!passed) {
            failures++;
        }
    }

    static class Result {
        int trials = 0;
        int closer = 0;
        int farther = 0;
        int unchanged = 0;
        int startedAtOrBelowZero = 0;
        int atOrBelowZeroAndCloser = 0;
        int fartherAndCrossedTarget = 0;
    }

    private static Result run(double learningRate) {
        State state = new State(3, 5, null);
        Experience terminal = new Experience(state, Action.EAST, 10.0, null);
        int eastIndex = Action.EAST.index();
        double target = 10.0;
        int inputLength = state.convertToInput(Action.EAST).length;

        int steps = 20;
        Result result = new Result();
        result.trials = 200;

        for (int t = 0; t < result.trials; t++) {
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

            if (start <= 0) {
                result.startedAtOrBelowZero++;
                if (endDistance < startDistance) {
                    result.atOrBelowZeroAndCloser++;
                }
            }
        }
        return result;
    }

    private static void print(String label, Result result) {
        System.out.println(label + ": " + result.trials + " networks, 20 steps each");
        System.out.println("  closer=" + result.closer + " farther=" + result.farther + " unchanged=" + result.unchanged);
        System.out.println("  started at or below 0: " + result.startedAtOrBelowZero + ", of those closer: " + result.atOrBelowZeroAndCloser);
        System.out.println("  farther and crossed the target at some step: " + result.fartherAndCrossedTarget);
    }

    public static void main(String[] args) {
        Result agentRate = run(0.01);
        print("learning rate 0.01 (the agent's rate)", agentRate);
        check("no network moves farther from the target", agentRate.farther == 0);
        check("every network moves closer to the target", agentRate.closer == agentRate.trials);
        check("every network that starts at or below 0 moves closer", agentRate.atOrBelowZeroAndCloser == agentRate.startedAtOrBelowZero);

        Result large = run(0.1);
        print("learning rate 0.1 (reported only)", large);

        System.out.println("failures: " + failures);
        if (failures > 0) {
            System.exit(1);
        }
    }
}
