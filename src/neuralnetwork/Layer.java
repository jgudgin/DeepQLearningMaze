package neuralnetwork;

import java.util.Arrays;

import java.util.Random;

public abstract class Layer {

    protected double[][] weights;
    protected double[] biases;
    protected double[] outputs;
    protected double[] inputs;

    protected int inputSize;

    public Layer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        weights = new double[inputSize][outputSize];
        biases = new double[outputSize];
        outputs = new double[outputSize];
        initWeights();
    }

    //initialize weights using kaiming initialization
    //w ~ N(0,(2/n))
    //where w = weight
    //and N(0,(2/n)) is a gaussian distribution with mean = 0 and n is number of neurons in previous layer / inputs to the current layer
    private void initWeights() {
        Random random = new Random();

        //for every weight, a random number is gotten from a gaussian distribution
        //centered at 0 with a SD of sqrt(2/n)
        for (int j = 0; j < weights[0].length; j++) {  // Iterate over output neurons
            int fanIn = (weights.length == 0) ? inputSize : weights.length; // Use inputSize for the first layer
            for (int i = 0; i < weights.length; i++) { // Iterate over input neurons
                weights[i][j] = random.nextGaussian() * Math.sqrt(2.0 / fanIn);
            }
        }

    }

    //method for updating the weights and biases of each neuron
    public void updateWeights(double[][] weightGradients, double[] biasGradients, double learningRate) {
        //update weights based on the calculated weight gradients
        //using formula Wnew = Wold - alpha * (∂L / ∂W)
        //where W = weight, alpha = learning rate, L = loss
        //weight gradient (dL / dW) is already calculated before this method is called
        for (int i = 0; i < weights[i].length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= learningRate * weightGradients[i][j];
            }
        }

        //update biases using the calculated weight gradients
        //using formula b = b - alpha * (∂L / ∂b)
        //where b = bias, alpha = learning rate, L = loss
        //weight gradient (∂L / ∂W) is already calculated before this method is called
        for (int j = 0; j < biases.length; j++) {
            biases[j] -= learningRate * biasGradients[j];
        }
    }

    //abstract method for the forward pass
    public abstract double[] forward(double[] inputs);

    //return the outputs of current layer
    public double[] getOutputs() {
        return outputs;
    }

    //return the number of outputs for current layer
    public int getOutputSize() {
        return outputs.length;
    }

    //abstract method for calculating gradients for previous layer
    public abstract double[] calcNextGradients(double[] layerGradients);
}
