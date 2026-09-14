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

### 2026-09-15: The output layer gradient is the error

- Decision: `QLearningNetwork.backpropagate` uses the error as the output layer gradient, with no ReLU derivative.
- Reason: `Output.forward` applies no activation, so the derivative of each output with respect to its weighted sum is 1. The ReLU derivative gave every prediction at or below 0 a zero gradient, so those networks never learned.
- Consequence: `tools/GradientCheckProbe.java` also checks a network whose prediction starts below 0. `tools/TrainingDirectionProbe.java` expects every network to move closer at its checked learning rate.

### 2026-09-15: The learning rate is applied once, and the agent uses 0.01

- Decision: `QLearningNetwork.train` sets the chosen action's target to `r + gamma * max Q(s', a')` and no longer blends it with the prediction using `alpha`. `alpha` is only the gradient step size in `Layer.updateWeights`.
- Decision: `Agent` sets the learning rate to 0.01.
- Reason: deep Q-learning regresses the network output onto the Q-learning target and applies the learning rate once, through gradient descent. The blended target scaled every step by `alpha` a second time, so at 0.1 each step was 10 times smaller than intended. A learning rate of 0.01 keeps the step size training used before this change.
- Consequence: `tools/GradientCheckProbe.java` checks the gradient of `1/2 * (Q - r)^2` for a terminal experience.

### 2026-09-15: Experiences store the state before and after the move

- Decision: `Agent.move` keeps the state it acted from as `previousState` and stores `new Experience(previousState, action, reward, currentState)`.
- Reason: Q-learning learns from `(s, a, r, s')`, where `s` is the position the agent acted from and `s'` is the position the action led to. The agent stored the position after the move as both states, so the network trained on the wrong state-action pair.
- Consequence: `tools/ExperienceStateProbe.java` makes two moves and checks both states of each stored experience.

### 2026-09-15: A move into the goal is stored as terminal

- Decision: when a move reaches the goal, `Agent.move` stores `null` as the experience's next state. `Agent.isAtGoal` decides whether a state is the goal, and `Agent.calculateReward` uses the same check.
- Reason: the episode ends at the goal, so the target for that move should be the reward alone. `QLearningNetwork.train` already treats a null next state as terminal. The agent stored the goal state instead, so the target added `gamma * max Q(goal)`.
- Alternative not taken: an explicit terminal flag on `Experience`. That would change the constructor and every place that builds an experience.
- Consequence: no stored experience references the goal state object, so the in-place reset in `MazeApp.startGameLoop` no longer changes a stored experience. `tools/GoalTerminalProbe.java` checks the terminal experience and the reset.

### 2026-09-15: Coordinates are scaled to between 0 and 1

- Decision: `State.convertToInput` divides `x` and `y` by the largest grid index, `MazeApp.GRID_SIZE - 1`, which is 15. `MazeApp.GRID_SIZE` is readable inside the package so `State` can use it.
- Reason: raw coordinates up to 15 made training steps overshoot their targets. A scratchpad comparison of 200 networks trained for 20 steps toward a target of 10 at learning rate 0.01 found 10 overshooting with raw coordinates and none with scaled coordinates.
- Alternatives not taken: lowering the learning rate to 0.001, which also stopped the overshoot but trained more slowly, and Huber loss, which stops runaway steps but barely moves toward a large reward in 20 steps.
- Consequence: `tools/TrainingDirectionProbe.java` checks direction at the agent's learning rate of 0.01 and reports 0.1. `tools/CompileFixProbe.java` expects the input `{1/15, 1/15, 0, 0, 1, 0}` for column 1 row 1 EAST.

## Known bugs

Entries marked (unverified) come from reading the code. The others were confirmed by running it.

### Training

- The goal reward of 5000 makes squared-error training run away. In a scratchpad run of 200 networks trained for 20 steps toward a target of 5000 at learning rate 0.01, 163 overshot with raw coordinates and 103 overshot with scaled coordinates. No probe in `tools/` covers this yet.

### Policy and rewards

- `EpsilonSoft.softmax` still produces NaN probabilities when a Q-value is NaN or infinite, and it divides by zero when tau is 0. (unverified)
- `EpsilonSoft.selectActionFromProbs` uses `Math.random()` instead of the `random` field, so seeding the field does not make runs repeatable. (unverified)
- `Agent.calculateReward` gives a dead end a positive reward about 10 points higher than a normal move. Its comment says dead ends should lose points. (unverified)

### Experience replay

- `ExperienceReplay.addExperience` checks for duplicates with `Experience.equals`, which `Experience` does not override. The check never matches.
- `ExperienceReplay.addExperience` replaces a random entry when the buffer is full. Its comment says it replaces the oldest. (unverified)
- `MazeApp.startGameLoop` resets the agent's `State` object in place at the end of an episode. No stored experience references that object today, and `tools/GoalTerminalProbe.java` shows no experience changes after the reset. Any future code that stores the agent's current state object would be rewritten by the reset.

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
- Is the epsilon decay rate intended? At 0.999 every 20 moves, epsilon needs about 46,000 valid moves to fall from 1.0 to 0.1.
- What should the goal, a normal step and a dead end be worth in `Agent.calculateReward`? The goal reward of 5000 makes training run away (see Known bugs), so smaller rewards may be needed.
