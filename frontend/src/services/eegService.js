const movements = ["LEFT", "RIGHT", "UP", "DOWN", "CENTER"];

export function getFakePrediction() {
  const movement =
    movements[Math.floor(Math.random() * movements.length)];

  const confidence = Math.floor(Math.random() * 15) + 85;

  return {
    movement,
    confidence,
  };
}