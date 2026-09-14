# Notes

## State coordinates

- Assumption: `State.x` is the maze row and `State.y` is the maze column.
- Source: `State.getSurroundings` checks `maze[x - 1][y]` for NORTH and `maze[x + 1][y]` for SOUTH.
- Consequence: `State.getNextState` adds `Action.getDeltaRow()` to `x` and `Action.getDeltaCol()` to `y`.

## Action encoding

- Decision: `Action.convertToInput()` returns a one-hot array of length `Action.COUNT`, with the 1 at `index()`.
- The order is NORTH, SOUTH, EAST, WEST. This matches the encoding of the `Action` class before it became an enum.
- Open question: the network takes the action as input and also outputs one value per action. A Q(s) design would drop the action from the input and remove this method.
