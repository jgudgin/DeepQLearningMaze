# Notes

## State coordinates

- Assumption: `State.x` is the maze row and `State.y` is the maze column.
- Source: `State.getSurroundings` checks `maze[x - 1][y]` for NORTH and `maze[x + 1][y]` for SOUTH.
- Consequence: `State.getNextState` adds `Action.getDeltaRow()` to `x` and `Action.getDeltaCol()` to `y`.

## Action encoding

- Decision: `Action.convertToInput()` returns a one-hot array of length `Action.COUNT`, with the 1 at `index()`.
- The order is NORTH, SOUTH, EAST, WEST. This matches the encoding of the `Action` class before it became an enum.
- Open question: the network takes the action as input and also outputs one value per action. A Q(s) design would drop the action from the input and remove this method.

## Coordinates after merging development

- Decision: `State.x` is the maze column and `State.y` is the maze row. This decision replaces the row assumption in "State coordinates".
- Source: `State.updateSurroundings` checks `maze[y - 1][x]` for NORTH, and `MazeApp` draws `maze[row][col]`.
- Consequence: `State.getNextState` adds `Action.getDeltaCol()` to `x` and `Action.getDeltaRow()` to `y`.
- `Action` stays an enum. The enum's built-in `values()` replaces the `values()` method from the development branch.
- `QLearningNetwork` uses the hidden layer loop from the development branch. That loop creates every hidden layer.

## Agent action selection

- Decision: `Agent.move` asks the network for a Q-value for each available action before it selects a move.
- For each action, the agent calls `predict` with its current state and that action, then reads `output[action.index()]`.
- Source: `QLearningNetwork.train` only adjusts `output[action.index()]` for the action in the input, so that output is the network's Q-value for the pair.
- The agent queries the state before the move. Training still stores the state after the move.
- Known risk: `EpsilonSoft.softmax` overflows once a Q-value passes about 354 at tau 0.5. The probabilities then become NaN and the policy returns the last action.
