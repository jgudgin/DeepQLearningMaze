package neuralnetwork;

//the hidden layer appplies transformations to inputs using weights and biases
//it then applies and activation function (ReLU) to the result before it passing it into the next layer
public class Hidden extends Layer {

    public Hidden(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    //forward pass, calculating the output for each neuron in current layer
    @Override
    public double[] forward(double[] inputs) {
        this.inputs = inputs;
        
        //calculate the output for each neuron in the hidden layer
        for (int j = 0; j < outputs.length; j++) {
            outputs[j] = biases[j]; //start with the bias of the current neuron
            
            //add the product of each input and its weight for the current neuron
            for (int i = 0; i < inputs.length; i++) {
                outputs[j] += inputs[i] * weights[i][j];
            }
            
            //apply ReLU activation function: max(0, output[j]) to introduce non-linearity
            outputs[j] = Math.max(0, outputs[j]);
        }
        
        //returns the transformed and activated outputs of the next layer
        return outputs;
    }
    
    @Override
    //method for calculating the gradient of the next layer during backpropagation
    public double[] calcNextGradients(double[] layerGradients) {
        double[] nextGradients = new double[inputs.length];

        // TODO comment needed: why this method does not apply the ReLU derivative, since QLearningNetwork.backpropagate applies it once before calling this method
        for (int j = 0; j < outputs.length; j++) {
            //propagate the gradients back to the previous layer
            for (int i = 0; i < inputs.length; i++) {
                nextGradients[i] += layerGradients[j] * weights[i][j];
            }
        }

        return nextGradients;
    }
   
}
