package neuralnetwork;

import java.util.Random;
import java.util.List;
import java.lang.Math;

//epsilon soft action selection policy
public class EpsilonSoft {

    private double epsilon; //probability of choosing a random action
    private double tau; //temperature parameter that controls the level of exploration
    private Random random;

    public EpsilonSoft(double epsilon, double tau) {
        this.epsilon = epsilon;
        this.tau = tau;
        this.random = new Random();
    }

    public Action selectAction(double[] qValues, List<Action> actions) {
        //choose to explore
        if (random.nextDouble() < epsilon) {
            return actions.get(random.nextInt(actions.size())); //choose a random available action
        } else {
            //choose to exploit known actions
            //use softmax to choose an action based on Q-values
            double[] probabilities = softmax(qValues);
            return selectActionFromProbs(probabilities, actions);
        }
    }

    //softmax funciton for selecting action if not random
    private double[] softmax(double[] qValues) {
        double[] probs = new double[qValues.length];
        double sumExp = 0.0;

        //calculate the exponentials using e^(Q(a)/τ from softmax equation
        //for each Q-value, the e function is computed and accumulated into sumExp
        //sumExp = ∑j e^(Q(aj) / τ​
        for (double q : qValues) {
            sumExp += Math.exp(q / tau);
        }

        //calculate probabilites
        //each actions prob is calulated by diving its exponential value by the sum of total exponentials
        //P(a|s) = (e^(Q(a) / τ) / (∑j e^(Q(aj) / τ)
        for (int i = 0; i < qValues.length; i++) {
            probs[i] = Math.exp(qValues[i] / tau) / sumExp;
        }

        return probs;
    }

    private Action selectActionFromProbs(double[] probs, List<Action> actions) {
        double randomValue = Math.random();
        double cumulativeProb = 0.0;
        

        //loop through the probabilites of each action
        for (int i = 0; i < actions.size(); i++) {

            //accumulate the probability of the current action
            cumulativeProb += probs[i];
            


            //use probabilistic action selection by comparing a random value to the cumulative probability
            if (randomValue <= cumulativeProb) {
                return actions.get(i);
            }
        }

        //if no action is selected (which should not happen) return the last action as a default
        return actions.get(actions.size() - 1);
    }
    
    public void setEpsilon(double epsilon){
        this.epsilon = epsilon;
    }
    
    public double getEpsilon(){
        return epsilon;
    }

}
