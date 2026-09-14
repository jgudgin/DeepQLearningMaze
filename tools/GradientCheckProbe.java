import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import neuralnetwork.Action;
import neuralnetwork.Agent;
import neuralnetwork.Experience;
import neuralnetwork.Layer;
import neuralnetwork.MazeApp;
import neuralnetwork.QLearningNetwork;
import neuralnetwork.State;

// Part 1: numerical gradient check of one QLearningNetwork.train step.
// For a terminal experience, train sets the target for the chosen action to
// u = Q + alpha * (reward - Q) and leaves the other outputs at their predictions. One step should
// therefore follow the gradient of L = 1/2 * (Q - u)^2 for every weight and bias.
// The probe nudges each weight and bias by +/- 1e-6 and measures the change in L. That central
// difference does not depend on the backpropagation code. The probe then runs one train step and
// recovers the gradient the code used from each parameter's change: -(after - before) / alpha.
// It runs the check twice: once for a network whose prediction starts between 0 and 10, and once
// for a network whose prediction starts below 0. Both networks have active neurons in both hidden
// layers.
// Part 2: the agent trains its real network for 50 moves without an exception. Training starts
// once the replay buffer holds 32 experiences.
public class GradientCheckProbe {

    private static int failures = 0;

    private static void check(String name, boolean passed) {
        System.out.println((passed ? "PASS " : "FAIL ") + name);
        if (!passed) {
            failures++;
        }
    }

    private static Object read(Class<?> owner, Object target, String name) throws Exception {
        Field field = owner.getDeclaredField(name);
        field.setAccessible(true);
        return field.get(target);
    }

    private static double loss(QLearningNetwork network, Experience experience, int actionIndex, double target) {
        double q = network.predict(experience)[actionIndex];
        return 0.5 * (q - target) * (q - target);
    }

    private static int countActive(double[] outputs) {
        int active = 0;
        for (double value : outputs) {
            if (value > 0) {
                active++;
            }
        }
        return active;
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

    private static void checkGradient(String label, double lowestQ, double highestQ) throws Exception {
        System.out.println("-- " + label);
        double alpha = 0.1;
        double reward = 10.0;
        int actionIndex = Action.EAST.index();
        Experience terminal = new Experience(new State(3, 5, null), Action.EAST, reward, null);
        int inputLength = new State(3, 5, null).convertToInput(Action.EAST).length;

        QLearningNetwork network = null;
        for (int t = 0; t < 5000 && network == null; t++) {
            QLearningNetwork candidate = new QLearningNetwork(inputLength, 4, new int[]{10, 5}, alpha, 0.9);
            double q = candidate.predict(terminal)[actionIndex];
            Layer[] hidden = (Layer[]) read(QLearningNetwork.class, candidate, "hidden");
            if (q > lowestQ && q < highestQ && countActive(hidden[0].getOutputs()) > 0 && countActive(hidden[1].getOutputs()) > 0) {
                network = candidate;
            }
        }
        check(label + ": found a matching network with active hidden neurons", network != null);
        if (network == null) {
            return;
        }

        Layer[] hidden = (Layer[]) read(QLearningNetwork.class, network, "hidden");
        Layer output = (Layer) read(QLearningNetwork.class, network, "output");
        Layer[] layers = {hidden[0], hidden[1], output};
        String[] names = {"hidden layer 0", "hidden layer 1", "output layer"};

        double q0 = network.predict(terminal)[actionIndex];
        double target = q0 + alpha * (reward - q0);
        System.out.printf("Q before the step: %.6f, training target u: %.6f%n", q0, target);

        double epsilon = 1e-6;
        double[][][] numericalWeights = new double[3][][];
        double[][] numericalBiases = new double[3][];
        double[][][] weightsBefore = new double[3][][];
        double[][] biasesBefore = new double[3][];

        for (int l = 0; l < layers.length; l++) {
            double[][] weights = (double[][]) read(Layer.class, layers[l], "weights");
            double[] biases = (double[]) read(Layer.class, layers[l], "biases");

            numericalWeights[l] = new double[weights.length][];
            weightsBefore[l] = new double[weights.length][];
            for (int i = 0; i < weights.length; i++) {
                numericalWeights[l][i] = new double[weights[i].length];
                for (int j = 0; j < weights[i].length; j++) {
                    double original = weights[i][j];
                    weights[i][j] = original + epsilon;
                    double lossUp = loss(network, terminal, actionIndex, target);
                    weights[i][j] = original - epsilon;
                    double lossDown = loss(network, terminal, actionIndex, target);
                    weights[i][j] = original;
                    numericalWeights[l][i][j] = (lossUp - lossDown) / (2 * epsilon);
                }
                weightsBefore[l][i] = weights[i].clone();
            }

            numericalBiases[l] = new double[biases.length];
            for (int j = 0; j < biases.length; j++) {
                double original = biases[j];
                biases[j] = original + epsilon;
                double lossUp = loss(network, terminal, actionIndex, target);
                biases[j] = original - epsilon;
                double lossDown = loss(network, terminal, actionIndex, target);
                biases[j] = original;
                numericalBiases[l][j] = (lossUp - lossDown) / (2 * epsilon);
            }
            biasesBefore[l] = biases.clone();
        }

        boolean trained = true;
        try {
            network.train(terminal);
        } catch (RuntimeException e) {
            trained = false;
            System.out.println("train threw " + e);
        }
        check(label + ": one train step runs without an exception", trained);
        if (!trained) {
            return;
        }

        for (int l = 0; l < layers.length; l++) {
            double[][] weights = (double[][]) read(Layer.class, layers[l], "weights");
            double[] biases = (double[]) read(Layer.class, layers[l], "biases");

            int weightCount = 0;
            int weightNonzero = 0;
            int weightMismatches = 0;
            double weightLargest = 0.0;
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    double used = -(weights[i][j] - weightsBefore[l][i][j]) / alpha;
                    double numerical = numericalWeights[l][i][j];
                    double difference = Math.abs(used - numerical);
                    weightCount++;
                    if (Math.abs(numerical) > 1e-9) {
                        weightNonzero++;
                    }
                    if (difference > 1e-5 + 1e-3 * Math.abs(numerical)) {
                        weightMismatches++;
                    }
                    weightLargest = Math.max(weightLargest, difference);
                }
            }

            int biasNonzero = 0;
            int biasMismatches = 0;
            double biasLargest = 0.0;
            for (int j = 0; j < biases.length; j++) {
                double used = -(biases[j] - biasesBefore[l][j]) / alpha;
                double numerical = numericalBiases[l][j];
                double difference = Math.abs(used - numerical);
                if (Math.abs(numerical) > 1e-9) {
                    biasNonzero++;
                }
                if (difference > 1e-5 + 1e-3 * Math.abs(numerical)) {
                    biasMismatches++;
                }
                biasLargest = Math.max(biasLargest, difference);
            }

            System.out.printf("%s weights: %d total, %d with nonzero numerical gradient, %d mismatches, largest difference %.3e%n",
                    names[l], weightCount, weightNonzero, weightMismatches, weightLargest);
            System.out.printf("%s biases: %d total, %d with nonzero numerical gradient, %d mismatches, largest difference %.3e%n",
                    names[l], biases.length, biasNonzero, biasMismatches, biasLargest);
            check(label + ": " + names[l] + " weights follow the numerical gradient", weightMismatches == 0);
            check(label + ": " + names[l] + " biases follow the numerical gradient", biasMismatches == 0);
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("== Part 1: numerical gradient check");
        checkGradient("prediction between 0 and 10", 0.5, 9.5);
        checkGradient("prediction below 0", -9.5, -0.5);

        System.out.println("== Part 2: agent trains for 50 moves");
        int[][] grid = {
            {1, 1, 1, 1, 1},
            {1, 0, 0, 0, 1},
            {1, 0, 0, 0, 1},
            {1, 0, 0, 0, 1},
            {1, 1, 1, 1, 1}
        };
        OpenMaze maze = new OpenMaze(grid);
        Agent agent = new Agent(new State(1, 1, maze), maze);
        int completed = 0;
        try {
            for (int m = 0; m < 50; m++) {
                State state = agent.getCurrentState();
                state.updateSurroundings(grid, state.getX(), state.getY());
                List<Action> open = new ArrayList<>();
                for (Action action : Action.values()) {
                    if (state.isPath(action)) {
                        open.add(action);
                    }
                }
                agent.move(open);
                completed++;
            }
        } catch (Exception e) {
            System.out.println("move " + (completed + 1) + " threw " + e);
        }
        System.out.println("moves completed: " + completed + " of 50");
        check("the agent completes 50 moves, including training, without an exception", completed == 50);

        System.out.println("failures: " + failures);
        if (failures > 0) {
            System.exit(1);
        }
    }
}
