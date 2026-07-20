export const DQN_INPUT_NAME = 'observation';
export const DQN_OUTPUT_NAME = 'q_values';
export const DQN_CHANNELS = 4;
export const DQN_BOARD_CELLS = 64;

export function encodeDqnObservation(board, player, legalIndices) {
  if (!Array.isArray(board) || board.length !== DQN_BOARD_CELLS) {
    throw new Error('DQN board must contain exactly 64 cells.');
  }
  if (player !== 1 && player !== -1) {
    throw new Error('DQN player must be black (1) or white (-1).');
  }

  const observation = new Float32Array(DQN_CHANNELS * DQN_BOARD_CELLS);
  let occupied = 0;
  for (let index = 0; index < DQN_BOARD_CELLS; index += 1) {
    const value = board[index];
    if (value !== 0 && value !== 1 && value !== -1) {
      throw new Error('DQN board contains an invalid cell value.');
    }
    if (value !== 0) occupied += 1;
    if (value === player) observation[index] = 1;
    if (value === -player) observation[DQN_BOARD_CELLS + index] = 1;
  }

  for (const index of legalIndices) {
    if (!Number.isInteger(index) || index < 0 || index >= DQN_BOARD_CELLS) {
      throw new Error('DQN legal action must be an integer between 0 and 63.');
    }
    observation[DQN_BOARD_CELLS * 2 + index] = 1;
  }

  const progress = occupied / DQN_BOARD_CELLS;
  observation.fill(progress, DQN_BOARD_CELLS * 3, DQN_BOARD_CELLS * 4);
  return observation;
}

export function selectLegalAction(qValues, legalIndices) {
  if (!qValues || qValues.length !== DQN_BOARD_CELLS) {
    throw new Error('DQN output must contain exactly 64 Q-values.');
  }
  if (!Array.isArray(legalIndices) || legalIndices.length === 0) {
    return null;
  }

  let bestIndex = legalIndices[0];
  let bestValue = Number(qValues[bestIndex]);
  for (let offset = 1; offset < legalIndices.length; offset += 1) {
    const index = legalIndices[offset];
    const value = Number(qValues[index]);
    if (value > bestValue || (value === bestValue && index < bestIndex)) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return bestIndex;
}
