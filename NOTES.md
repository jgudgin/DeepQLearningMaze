# Notes

## Design decisions

### 2026-09-15: Action is an enum

- Decision: `Action` is an enum with the constants NORTH, SOUTH, EAST, WEST in that order.
- Reason: `ordinal()` gives each action its network output index, and `COUNT` sizes the output layer.
- Consequence: the enum's built-in `values()` replaces the hand-written `values()` from the old class.

### 2026-09-15: Actions are one-hot encoded

- Decision: `Action.convertToInput()` returns a one-hot array of length `Action.COUNT`, with the 1 at `index()`.
- Reason: this matches the encoding of the `Action` class before it became an enum.
- Consequence: reordering the enum constants changes the network input.

### 2026-09-15: x is the column and y is the row

- Decision: `State.x` is the maze column and `State.y` is the maze row.
- Reason: `State.updateSurroundings` checks `maze[y - 1][x]` for NORTH, and `MazeApp` draws `maze[row][col]`.
- Consequence: `State.getNextState` adds `Action.getDeltaCol()` to `x` and `Action.getDeltaRow()` to `y`.

### 2026-09-15: Every hidden layer is created in one loop

- Decision: the `QLearningNetwork` constructor creates each hidden layer in a single loop. The first layer takes `inputSize` inputs, and each later layer takes the previous layer's size.
- Reason: the earlier loop stopped one layer short and left the last hidden layer null.

### 2026-09-15: The agent chooses moves from network Q-values

- Decision: `Agent.predictQValues` calls `predict` once for each available action, with the current state and that action, and reads `output[action.index()]`.
- Reason: `QLearningNetwork.train` only adjusts `output[action.index()]` for the action in the input, so that output is the network's Q-value for the pair.
- Consequence: `tools/AgentPolicyProbe.java` checks that a network preferring EAST makes the agent pick EAST.

### 2026-09-15: development is the working branch

- Decision: `development` was merged into `main`, and new work happens on `development`.

### 2026-09-15: Softmax subtracts the largest Q-value

- Decision: `EpsilonSoft.softmax` subtracts the largest Q-value from every Q-value before dividing by tau and calling `Math.exp`.
- Reason: `Math.exp` overflows to Infinity above about 709 and underflows to 0 far below it. Either case turns the probabilities into NaN, and `selectActionFromProbs` then always returns the last action. The subtraction cancels in the softmax fraction, so the probabilities stay the same, and the largest term becomes e^0 = 1.
- Consequence: `tools/SoftmaxProbe.java` checks the chosen-action shares against softmax values worked out by hand, including Q-values of 1000 and -1000.

### 2026-09-15: The training error is prediction minus target

- Decision: `QLearningNetwork.backpropagate` computes the error as prediction minus target.
- Reason: `Layer.updateWeights` subtracts the learning rate times the gradient. For squared error, the gradient with respect to the prediction is prediction minus target. The earlier target minus prediction made every update move the prediction away from its target.
- Alternative not taken: making `Layer.updateWeights` add the gradient. That would contradict the update formula written in `Layer`.
- Consequence: `tools/TrainingDirectionProbe.java` trains 200 networks toward a fixed target of 10 and checks that none moves farther away.

### 2026-09-15: Backpropagation follows the full gradient path

- Decision: `QLearningNetwork.backpropagate` runs a forward pass on the experience, then works from the output layer back to the first hidden layer. For each layer it computes the gradient for the layer below from the current weights, updates the weights from the layer's inputs, and passes the gradient on. Each hidden layer applies the ReLU derivative once, in `backpropagate`, so `Hidden.calcNextGradients` no longer applies it.
- Decision: `QLearningNetwork.train` copies the prediction it gets from `predict`, because `predict` returns the output layer's shared array.
- Decision: `Layer.updateWeights` loops over every weight row, and `Agent` sets the network input size to `2 + Action.COUNT`.
- Reason: the hidden layers never received a gradient, the output weight gradients used the raw network input, and the hidden weight gradients used each layer's outputs. A network with the correct input size threw `ArrayIndexOutOfBoundsException` on its first training step.
- Consequence: `tools/GradientCheckProbe.java` compares one training step with a numerical gradient for every weight and bias, and runs the agent for 50 moves with training. `tools/UpdateWeightsProbe.java` checks the weight update loop and the agent's input size.

## Known bugs

Entries marked (unverified) come from reading the code. The others were confirmed by running it.

### Training

- `QLearningNetwork.train` scales the target change by `alpha`, and `backpropagate` uses `alpha` again as the step size. (unverified)
- At the agent's learning rate of 0.1, training steps can jump past their target. `tools/TrainingDirectionProbe.java` reports this at 0.1 without failing. In one diagnostic run, 24 of the 97 networks that started above 0 ended farther from a target of 10 after 20 steps, and all 24 had crossed the target. At 0.01, none did. The unscaled coordinate inputs probably make the steps larger. (cause unverified)
- `QLearningNetwork.backpropagate` applies the ReLU derivative to the output layer, which has no activation. A prediction at or below 0 gets a zero gradient and never changes. `tools/TrainingDirectionProbe.java` shows this: every network that started at or below 0 stayed unchanged after 20 training steps.
- `Agent.move` stores the state after the move as both the current and the next state of each experience.
- Training never treats the goal as a terminal state. (unverified)

### Policy and rewards

- `EpsilonSoft.softmax` still produces NaN probabilities when a Q-value is NaN or infinite, and it divides by zero when tau is 0. (unverified)
- `EpsilonSoft.selectActionFromProbs` uses `Math.random()` instead of the `random` field, so seeding the field does not make runs repeatable. (unverified)
- `Agent.calculateReward` gives a dead end a positive reward about 10 points higher than a normal move. Its comment says dead ends should lose points. (unverified)

### Experience replay

- `ExperienceReplay.addExperience` checks for duplicates with `Experience.equals`, which `Experience` does not override. The check never matches.
- `ExperienceReplay.addExperience` replaces a random entry when the buffer is full. Its comment says it replaces the oldest. (unverified)
- `MazeApp.startGameLoop` resets the agent's `State` object in place at the end of an episode. This rewrites the state stored in the last experience.

### Maze and app

- `MazeApp.carvePath` never carves row `GRID_SIZE - 2` or column `GRID_SIZE - 2`, so the right and bottom edges have a double wall. (unverified)
- `MazeApp.carvePath` labels its directions wrongly, creates a new unseeded `Random` on every call and uses a biased shuffle. (unverified)
- `MazeApp.startGameLoop` checks the episode count before incrementing it, so training stops at the 101st goal. (unverified)
- `QLearningNetwork.backpropagate` creates an unused `Scanner` on every call. `Agent` has an unused `Scanner` field and unused locals in `move` and `trainWithBatch`.
- The `QLearningNetwork` constructor throws when `hiddenSizes` is empty. (unverified)

### Repository

- `nbproject/private/` is committed and contains absolute paths under `/Users/jgudgin`.
- The repository has no README. `nbproject/project.properties` points at a `test/` folder that does not exist.
- `.gitignore` does not list `output/` or `dist/`.

## Open questions

- Should the network be Q(s) or Q(s,a)? The input includes the action, and the output holds one value per action. A Q(s) design would drop the action from the input and remove `Action.convertToInput()`.
- Should coordinates be scaled before they enter the network? `State.convertToInput` passes raw values from 0 to 15.
- Is the epsilon decay rate intended? At 0.999 every 20 moves, epsilon needs about 46,000 valid moves to fall from 1.0 to 0.1.
- What should a normal step and a dead end be worth in `Agent.calculateReward`?
