import java.lang.reflect.Field;

import neuralnetwork.Action;
import neuralnetwork.Agent;
import neuralnetwork.Hidden;
import neuralnetwork.Layer;
import neuralnetwork.Output;
import neuralnetwork.QLearningNetwork;
import neuralnetwork.State;

// Checks Layer.updateWeights coverage and the size of the agent's network input.
// Expectation for updateWeights: a gradient descent step changes every weight whose gradient is
// nonzero, for any layer shape, and never throws when the gradient matches the weight shape.
// Expectation for the input size: State.convertToInput returns x, y and a one-hot action, so the
// agent's first hidden layer needs exactly one weight row per value in that array.
public class UpdateWeightsProbe {

    private static int failures = 0;

    private static void check(String name, boolean passed) {
        System.out.println((passed ? "PASS " : "FAIL ") + name);
        if (!passed) {
            failures++;
        }
    }

    private static double[][] weightsOf(Layer layer) throws Exception {
        Field field = Layer.class.getDeclaredField("weights");
        field.setAccessible(true);
        return (double[][]) field.get(layer);
    }

    private static double[][] copy(double[][] source) {
        double[][] result = new double[source.length][];
        for (int i = 0; i < source.length; i++) {
            result[i] = source[i].clone();
        }
        return result;
    }

    private static double[][] ones(int rows, int cols) {
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = 1.0;
            }
        }
        return result;
    }

    private static int countChanged(double[][] before, double[][] after) {
        int changed = 0;
        for (int i = 0; i < before.length; i++) {
            for (int j = 0; j < before[i].length; j++) {
                if (before[i][j] != after[i][j]) {
                    changed++;
                }
            }
        }
        return changed;
    }

    public static void main(String[] args) throws Exception {
        Output wide = new Output(6, 4);
        double[][] wideBefore = copy(weightsOf(wide));
        wide.updateWeights(ones(6, 4), new double[4], 1.0);
        int wideChanged = countChanged(wideBefore, weightsOf(wide));
        System.out.println("6-input 4-output layer: " + wideChanged + " of 24 weights changed");
        check("updateWeights changes all 24 weights of a 6-input 4-output layer", wideChanged == 24);

        Hidden tall = new Hidden(3, 5);
        double[][] tallBefore = copy(weightsOf(tall));
        String thrown = "none";
        try {
            tall.updateWeights(ones(3, 5), new double[5], 1.0);
        } catch (ArrayIndexOutOfBoundsException e) {
            thrown = "ArrayIndexOutOfBoundsException: " + e.getMessage();
        }
        int tallChanged = countChanged(tallBefore, weightsOf(tall));
        System.out.println("3-input 5-output layer: exception " + thrown + ", " + tallChanged + " of 15 weights changed");
        check("updateWeights on a 3-input 5-output layer does not throw", thrown.equals("none"));
        check("updateWeights changes all 15 weights of a 3-input 5-output layer", tallChanged == 15);

        int inputLength = new State(1, 1, null).convertToInput(Action.NORTH).length;
        Agent agent = new Agent(new State(1, 1, null), null);
        Field networkField = Agent.class.getDeclaredField("qLearningNetwork");
        networkField.setAccessible(true);
        QLearningNetwork network = (QLearningNetwork) networkField.get(agent);
        Field hiddenField = QLearningNetwork.class.getDeclaredField("hidden");
        hiddenField.setAccessible(true);
        Layer[] hidden = (Layer[]) hiddenField.get(network);
        int firstLayerRows = weightsOf(hidden[0]).length;
        System.out.println("network input length: " + inputLength + ", agent's first hidden layer weight rows: " + firstLayerRows);
        check("the agent's first hidden layer has one weight row per input value", firstLayerRows == inputLength);

        System.out.println("failures: " + failures);
        if (failures > 0) {
            System.exit(1);
        }
    }
}
