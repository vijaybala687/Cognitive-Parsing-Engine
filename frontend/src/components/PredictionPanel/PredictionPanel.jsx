import "./PredictionPanel.css";

function PredictionPanel({ prediction = "LEFT", confidence = 98, status = "ACTIVE", onSendTrigger, onReset }) {
  const handleInject = (direction) => {
    if (onSendTrigger) {
      onSendTrigger(direction);
    }
  };

  const handleReset = () => {
    if (onReset) {
      onReset();
    }
  };

  return (
    <aside className="predictionPanel">
      <h2>Live Prediction</h2>

      <div className="prediction-card">
        <span>Prediction</span>
        <h1>{prediction}</h1>
      </div>

      <div className="prediction-card">
        <span>Confidence</span>
        <h1>{confidence}%</h1>
      </div>

      <div className="prediction-card">
        <span>Status</span>
        <h1>{status}</h1>
      </div>

      <div className="prediction-card">
        <span>Model</span>
        <h1>Hybrid EEGNet ONNX</h1>
      </div>

      <div className="testing-panel">
        <h3>Model Testing</h3>

        <div className="button-grid">
          <button onClick={() => handleInject("LEFT")}>LEFT</button>
          <button onClick={() => handleInject("RIGHT")}>RIGHT</button>
          <button onClick={() => handleInject("UP")}>UP</button>
          <button onClick={() => handleInject("DOWN")}>DOWN</button>
          <button className="reset-btn" onClick={handleReset}>RESET</button>
        </div>
      </div>
    </aside>
  );
}

export default PredictionPanel;