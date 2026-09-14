package neuralnetwork;

//used to store each experience for calculating its Q-value

public class Experience {
    
    private State currentState, nextState; //the current given state and the next state, stored as coordinates in arrays
    private Action action, nextAction; //the current action about to be taken and the next action to be taken, stored as integers
    private double rewardReceived;  //the reward received from that action in that state
    
    public Experience(State currentState, Action action, double rewardReceived, State nextState){
        this.currentState = currentState;
        this.action = action;
        this.rewardReceived = rewardReceived;
        this.nextState = nextState;
    }
    
    
    //store getters and setters in here so they can be easily tracked and modified
    //setters
    public void setAction(Action action){
        this.action = action;
    }
    
    public void setNextAction(Action nextAction){
        this.nextAction = nextAction;
    }
    
    public void setRewardReceived(double rewardReceived){
        this.rewardReceived = rewardReceived;
    }
    
    public void setCurrentState(State currentState){
        this.currentState = currentState;
    }
    

    //getters
    public State getCurrentState(){
        return currentState;
    }
    
    public Action getAction(){
        return action;
    }
    
    public double getRewardReceived(){
        return rewardReceived;
    }
    
    public State getNextState(){
        return nextState;
    }
    
    public Action getNextAction(){
        return nextAction;
    }
}
