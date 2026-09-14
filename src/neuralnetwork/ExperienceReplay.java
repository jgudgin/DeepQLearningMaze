package neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

//store intial experiences before using the neural network and select one to start learning from
public class ExperienceReplay {

    private final List<Experience> replayBuffer;
    private final int bufferLimit;
    private final Random random;

    public ExperienceReplay(int bufferLimit) {
        this.bufferLimit = bufferLimit;
        this.replayBuffer = new ArrayList<>();
        this.random = new Random();
    }
    
    //add an experience to the replay buffer
    public void addExperience(Experience experience) {
        //check for duplicates
        for (Experience existingExperience : replayBuffer) {
            if (existingExperience.equals(experience)) {
                return; //experience already exists, skip adding
            }
        }

        if (replayBuffer.size() < bufferLimit) {
            replayBuffer.add(experience);
        } else {
            //replace the oldest experience with a new one (circular buffer)
            replayBuffer.set(random.nextInt(bufferLimit), experience);
        }
    }

    //sample an experience from the buffer
    public Experience sampleExperience() {
        if (replayBuffer.isEmpty()) {
            return null;
        }

        int index = random.nextInt(replayBuffer.size());
        return replayBuffer.get(index);
    }

    public int getBufferSize() {
        return replayBuffer.size();
    }
}
