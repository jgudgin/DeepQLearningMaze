import java.util.ArrayList;
import java.util.List;

import neuralnetwork.Action;
import neuralnetwork.EpsilonSoft;

// Checks how often EpsilonSoft.selectAction picks the first of two actions when epsilon is 0.
// Expected shares come from softmax at tau 0.5, worked out by hand:
// P(first) = e^(q1 / 0.5) / (e^(q1 / 0.5) + e^(q2 / 0.5)) = 1 / (1 + e^(2 * (q2 - q1)))
//   q = {1, 0}:              1 / (1 + e^-2)    = 0.8808
//   q = {1000, 0}:           1 / (1 + e^-2000) = 1.0000
//   q = {1000.5, 1000}:      1 / (1 + e^-1)    = 0.7311
//   q = {-1000, -1000.5}:    1 / (1 + e^-1)    = 0.7311
//   q = {0, 1000}:           1 / (1 + e^2000)  = 0.0000
public class SoftmaxProbe {

    private static int failures = 0;

    private static void check(String name, boolean passed) {
        System.out.println((passed ? "PASS " : "FAIL ") + name);
        if (!passed) {
            failures++;
        }
    }

    private static double firstActionShare(double[] qValues, int trials) {
        EpsilonSoft policy = new EpsilonSoft(0.0, 0.5);
        List<Action> actions = new ArrayList<>();
        actions.add(Action.SOUTH);
        actions.add(Action.EAST);

        int first = 0;
        for (int t = 0; t < trials; t++) {
            Action chosen = policy.selectAction(qValues.clone(), actions);
            if (chosen == Action.SOUTH) {
                first++;
            }
        }
        return (double) first / trials;
    }

    private static void runCase(String label, double[] qValues, double expected) {
        int trials = 20000;
        double tolerance = 0.02;
        double share = firstActionShare(qValues, trials);
        System.out.printf("%s: first action share %.4f, expected %.4f%n", label, share, expected);
        check(label + " is within " + tolerance + " of " + expected, Math.abs(share - expected) <= tolerance);
    }

    public static void main(String[] args) {
        runCase("q = {1, 0}", new double[]{1.0, 0.0}, 0.8808);
        runCase("q = {1000, 0}", new double[]{1000.0, 0.0}, 1.0);
        runCase("q = {1000.5, 1000}", new double[]{1000.5, 1000.0}, 0.7311);
        runCase("q = {-1000, -1000.5}", new double[]{-1000.0, -1000.5}, 0.7311);
        runCase("q = {0, 1000}", new double[]{0.0, 1000.0}, 0.0);

        System.out.println("failures: " + failures);
        if (failures > 0) {
            System.exit(1);
        }
    }
}
